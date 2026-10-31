import json
import re
from bisect import bisect_right
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from .asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX, api_key
except ImportError:
    from asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX, api_key

# Chat-completions transport. The docs also describe POST /translate, but the live
# gateway returns 404 for it; this route is the one the console quickstart uses.
API_URL = "https://api.bodhan.ai/v1/chat/completions"
MODEL = "indic-translate"
TARGET_SCRIPT = "codemix"  # native script, with common English words kept in English; not offered for "en"
TIMEOUT = 120  # seconds per request; a full batch takes longer than one ASR chunk

# The gateway allows about 16 requests per 60 s for this model (the 17th in a burst gets a 429 with
# Retry-After), whatever the per-key headers say. Stay under it: at most RATE_LIMIT sends per RATE_WINDOW.
RATE_LIMIT = 15
RATE_WINDOW = 61  # seconds; the real window is 60, plus a second of margin
_sent = deque()  # monotonic times of recent sends, shared by every job
_sent_lock = threading.Lock()

# Batching: several chunks per request saves rate limit and gives the model context.
# The context window is 32K tokens for input + output. 4000 characters stays far below
# it even for Indic scripts, which take more tokens per character than English.
MAX_BATCH_CHARS = 700
MAX_BATCH_CHUNKS = 30
MARKER = re.compile(r"^\s*\[(\d+)\]\s*(.*)$")

# A chunk longer than MAX_BATCH_CHARS (after the post-ASR merge) is cut at punctuation and its pieces translated apart
SENTENCE_END = set("।॥.?!۔؟᱾᱿꯫…")
CLAUSE = set(",;:،؛—–")
CLOSERS = set("\"'”’)]")

# Target languages accepted by the Bodhan translation API (bhb and bho are refused)
LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "brx": "Bodo",
    "doi": "Dogri",
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}


class FatalTranslationError(RuntimeError):
    """An error every request would hit (bad key, unsupported language), so the whole job stops."""


class BatchError(RuntimeError):
    """The batch failed in a way a smaller batch may fix (too large, truncated, markers lost)."""


def translate_chunks(
    jdir: Path, chunks: list[dict], source: str, target: str, progress=None, only: set[int] | None = None
) -> dict:
    """Fill chunk["translation"] for every chunk with text and no translation yet (or a failed one); chunks
    already translated are kept, so a retry only sends the rest. The caller clears translations made for
    another target. Returns request stats. `progress()` is called after each batch, so the caller can
    publish translations as they arrive. `only`: chunk ids, to translate just those (an edited segment).
    chunks.json is written even when a fatal error stops the run."""
    todo = [
        c for c in chunks
        if c.get("text") and c.get("translation") is None and (only is None or c["id"] in only)
    ]
    for c in todo:
        c.pop("translation_error", None)
    stats = {"requests": 0, "rate_limits": {}, "attempts": 0, "raw_dir": jdir / "translation_raw"}

    try:
        if source == target:
            for c in todo:
                c["translation"] = c["text"]
        else:
            stats["raw_dir"].mkdir(exist_ok=True)
            key = api_key("translate")
            langs = {"target_language_code": target}
            if target != "en":  # the API rejects codemix for English
                langs["target_script"] = TARGET_SCRIPT
            if source in LANGUAGES:  # bhb / bho can be transcribed but aren't translation codes
                langs["source_language_code"] = source
            pieces = {c["id"]: [{"chunk": c, "text": t} for t in _split_text(c["text"])] for c in todo}
            for batch in _batches([p for ps in pieces.values() for p in ps]):
                _translate_batch(batch, langs, key, stats)
                _assemble(todo, pieces)
                if progress:
                    progress()
    finally:
        for c in chunks:
            c.setdefault("translation", None)
            map_sub_translations(c)
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    stats["failed"] = sum(1 for c in chunks if c.get("translation_error"))
    return stats


OVERFLOW_CHARS = 50  # the dimmed start of the next segment shown after a segment cut mid-sentence


def map_sub_translations(chunk: dict):
    """Give each short segment of a merged chunk its slice of the merged translation: "translation", and
    "translation_original" (the translation before the check, when there is one) and "translation_reviewed" (the
    check's edit, when there is one). A segment that ends
    mid-sentence also gets "translation_overflow": the start of what follows it, up to the next punctuation
    mark or OVERFLOW_CHARS. Call it whenever the chunk's translation changes."""
    subs = chunk.get("sub_chunks")
    if not subs:
        return
    tr = chunk.get("translation")
    original = chunk.get("translation_original")
    if tr is None:
        pieces, cuts = [None] * len(subs), None
    else:
        pieces, cuts = _slices(subs, tr)
    origs = _slices(subs, original)[0] if original is not None else [None] * len(subs)
    reviewed = (chunk.get("review") or {}).get("reviewed")
    edits = _slices(subs, reviewed)[0] if reviewed is not None else [None] * len(subs)
    for k, s in enumerate(subs):
        s["translation"] = pieces[k]
        s["translation_original"] = origs[k]
        s["translation_reviewed"] = edits[k]
        s["translation_overflow"] = None
        text = (s.get("text") or "").strip()
        if cuts is not None and k < len(subs) - 1 and text and not _ends_sentence(text):
            s["translation_overflow"] = _overflow(tr[cuts[k]:].lstrip())


def _slices(subs: list[dict], tr: str):
    """Each short segment's slice of `tr` (None where empty), and where each slice ends in `tr`."""
    if len(subs) == 1:
        cuts = [len(tr)]
    else:
        have = [s.get("translation") for s in subs]
        if tr == " ".join(t for t in have if t):  # the segments' own translations, joined: keep their cuts
            cuts, pos = [], 0
            for t in have:
                if t:
                    pos += len(t)
                    cuts.append(pos)
                    pos += 1  # the space that joins it to the next
                else:
                    cuts.append(max(pos - 1, 0))
        else:
            cuts = _aligned_cuts(subs, tr)
    cuts[-1] = len(tr)
    starts = [0] + cuts[:-1]
    return [tr[a:b].strip() or None for a, b in zip(starts, cuts)], cuts


def _aligned_cuts(subs: list[dict], tr: str) -> list[int]:
    """Where each short segment ends in `tr`, given the source text the segments make up. The source is cut
    at the end of each segment, and that position is matched in `tr` by sentence (see _aligner)."""
    parts, spans, pos = [], [], 0
    for s in subs:
        text = (s.get("text") or "").strip()
        if not text:
            spans.append(None)
            continue
        if parts:
            pos += 1  # the space merge_chunks joins the segments with
        spans.append(pos + len(text))
        parts.append(text)
        pos += len(text)
    at = _aligner(" ".join(parts), tr)
    cuts, last = [], 0
    for end in spans:
        if end is not None:
            last = max(last, at(end))
        cuts.append(last)
    return cuts


def _aligner(src: str, tgt: str):
    """A function from a position in `src` to the matching position in `tgt`. The sentences of both texts pair
    up in order, sentence j of `src` with the target sentence at the same share of the count, so the alignment
    starts again at every sentence end. Inside a sentence the position keeps its share of the sentence, moved
    to the nearest space so no word is cut."""
    S, T = _bounds(src), _bounds(tgt)
    n, m = len(S) - 1, len(T) - 1

    def target_bound(j):
        return T[round(j * m / n)]

    def at(p):
        j = bisect_right(S, p) - 1
        if j >= n:
            return T[-1]
        if p == S[j]:
            return target_bound(j)
        lo, hi = target_bound(j), target_bound(j + 1)
        share = (p - S[j]) / (S[j + 1] - S[j])
        return _snap(tgt, lo + round(share * (hi - lo)), lo, hi)

    return at


def _bounds(text: str) -> list[int]:
    """Where the sentences of `text` begin and end: 0, the end of each sentence (after its closing quotes or
    brackets), and the length of the text. Always at least two positions."""
    ends = {0, len(text)}
    for i, ch in enumerate(text):
        if ch in SENTENCE_END and not _is_inner_dot(text, i):
            end = i + 1
            while end < len(text) and text[end] in CLOSERS:
                end += 1
            ends.add(end)
    bounds = sorted(ends)
    return bounds if len(bounds) > 1 else bounds * 2


def _snap(text: str, pos: int, lo: int, hi: int) -> int:
    """`pos`, moved to the nearest space within [lo, hi], or `pos` itself when there is none."""
    for d in range(hi - lo + 1):
        for q in (pos - d, pos + d):
            if lo <= q <= hi and q < len(text) and text[q] == " ":
                return q
    return pos


def _ends_sentence(text: str) -> bool:
    """Whether the text ends with a sentence-ending mark, before any closing quotes or brackets."""
    t = text.rstrip().rstrip("".join(CLOSERS))
    return bool(t) and t[-1] in SENTENCE_END


def _overflow(rem: str) -> str | None:
    """The start of the next segment's translation, shown dimmed after a segment that ends mid-sentence: up to
    the first punctuation mark when it is within OVERFLOW_CHARS, else the first OVERFLOW_CHARS cut at a word."""
    for i, ch in enumerate(rem[:OVERFLOW_CHARS]):
        if (ch in SENTENCE_END or ch in CLAUSE) and not _is_inner_dot(rem, i):
            end = i + 1
            while end < len(rem) and rem[end] in CLOSERS:
                end += 1
            if end <= OVERFLOW_CHARS:
                return rem[:end].strip() or None
            break
    if len(rem) <= OVERFLOW_CHARS:
        return rem.strip() or None
    head = rem[:OVERFLOW_CHARS]
    space = head.rfind(" ")
    if space > 0:
        head = head[:space]
    return head.rstrip() + "…"


def _split_text(text: str, max_chars: int = MAX_BATCH_CHARS) -> list[str]:
    """`text` as pieces of at most max_chars, cut after the last sentence-ending mark that fits, else the last
    clause mark, else the last space (closing quotes / brackets stay on the left); a text that fits is kept whole."""
    pieces = []
    text = text.strip()
    while len(text) > max_chars:
        cut = _cut_point(text, max_chars)
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return pieces or [text]


def _cut_point(text, max_chars) -> int:
    for marks in (SENTENCE_END, CLAUSE):
        cuts = []
        for i in range(min(max_chars, len(text))):
            if text[i] in marks and not _is_inner_dot(text, i):
                c = i + 1
                while c < len(text) and text[c] in CLOSERS and c < max_chars:
                    c += 1
                if c <= max_chars and text[:c].strip() and text[c:].strip():
                    cuts.append(c)
        if cuts:
            return max(cuts)
    spaces = [i for i in range(1, min(max_chars, len(text) - 1) + 1) if text[i].isspace() and text[:i].strip()]
    return max(spaces) if spaces else max_chars


def _is_inner_dot(text, i) -> bool:
    """A "." that doesn't end a sentence: a decimal point (3.5), or after a lone Latin letter (an initial)."""
    if text[i] != ".":
        return False
    if 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
        return True
    return i > 0 and text[i - 1].isascii() and text[i - 1].isalpha() and (i == 1 or not text[i - 2].isalpha())


def _batches(items):
    """Consecutive items packed up to MAX_BATCH_CHARS / MAX_BATCH_CHUNKS."""
    batch, size = [], 0
    for item in items:
        n = len(item["text"])
        if batch and (size + n > MAX_BATCH_CHARS or len(batch) >= MAX_BATCH_CHUNKS):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += n
    if batch:
        yield batch


def _assemble(chunks, pieces):
    """Join each chunk's translated pieces (`pieces[chunk id]`), in order, into chunk["translation"]; a piece that
    failed fails the chunk."""
    for c in chunks:
        ps = pieces[c["id"]]
        if c.get("translation") is not None:
            continue
        error = next((p["error"] for p in ps if p.get("error")), None)
        if error:
            c["translation_error"] = error
        elif all(p.get("translation") is not None for p in ps):
            c["translation"] = " ".join(p["translation"] for p in ps)


def _translate_batch(batch, langs, key, stats):
    """Translate a batch in one request; on BatchError split it in half and retry each half."""
    if len(batch) == 1:
        text = _one_line(batch[0]["text"])
    else:
        text = "\n\n".join(f"[{i}] {_one_line(p['text'])}" for i, p in enumerate(batch, 1))

    try:
        out, truncated = _request(text, langs, key, stats)
        if truncated:
            raise BatchError("translation was truncated")
        parts = [out.strip()] if len(batch) == 1 else _parse(out, len(batch))
    except FatalTranslationError:
        raise
    except BatchError as exc:
        if len(batch) == 1:
            batch[0]["error"] = str(exc)
            return
        mid = len(batch) // 2
        _translate_batch(batch[:mid], langs, key, stats)
        _translate_batch(batch[mid:], langs, key, stats)
        return
    except Exception as exc:  # retries exhausted or a request error splitting won't fix
        for p in batch:
            p["error"] = str(exc)
        return

    for p, part in zip(batch, parts):
        p["translation"] = part


def _one_line(text):
    return " ".join(text.split())


def _parse(out, n):
    """Map "[i] ..." lines back to n chunks. Raises BatchError unless every marker 1..n appears once."""
    parts: dict[int, list[str]] = {}
    current = None
    for line in out.splitlines():
        m = MARKER.match(line)
        if m:
            current = int(m.group(1))
            if current in parts:
                raise BatchError(f"marker [{current}] repeated in translation")
            parts[current] = [m.group(2).strip()]
        elif line.strip():
            if current is None:
                raise BatchError("text before the first marker in translation")
            parts[current].append(line.strip())
    if sorted(parts) != list(range(1, n + 1)):
        raise BatchError(f"expected markers 1..{n}, got {sorted(parts)}")
    return [" ".join(parts[i]).strip() for i in range(1, n + 1)]


def _save_raw(stats, payload, res=None, error=None):
    """Write one attempt (request body + raw response) to translation_raw/<attempt>.json. No API key is stored."""
    stats["attempts"] += 1
    record = {
        "attempt": stats["attempts"],
        "time": datetime.now(timezone.utc).isoformat(),
        "url": API_URL,
        "request": payload,
    }
    if res is not None:
        record["status"] = res.status_code
        record["headers"] = dict(res.headers)
        try:
            record["body"] = res.json()
        except ValueError:
            record["body"] = res.text
    else:
        record["error"] = error
    path = stats["raw_dir"] / f"{stats['attempts']:04d}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))


def _throttle():
    """Block until another request fits in the rate window, then count it. Retries count too."""
    with _sent_lock:
        while True:
            now = time.monotonic()
            while _sent and now - _sent[0] >= RATE_WINDOW:
                _sent.popleft()
            if len(_sent) < RATE_LIMIT:
                _sent.append(now)
                return
            time.sleep(_sent[0] + RATE_WINDOW - now)


def _request(text, langs, key, stats):
    """POST one translation. Returns (translation, truncated)."""
    error = "unknown error"
    payload = {"model": MODEL, "messages": [{"role": "user", "content": text}], **langs}
    for attempt in range(RETRIES + 1):
        wait = min(BACKOFF * 2**attempt, MAX_WAIT)
        _throttle()
        try:
            res = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
            _save_raw(stats, payload, error=error)
        else:
            _save_raw(stats, payload, res=res)
            stats["requests"] += 1
            for h in ("X-Ratelimit-Api_key-Limit-Requests", "X-Ratelimit-Api_key-Limit-Max_parallel_requests"):
                if h in res.headers:
                    stats["rate_limits"][h] = res.headers[h]

            try:
                body = res.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}

            if res.status_code == 200 and body.get("choices"):
                choice = body["choices"][0]
                truncated = bool(body.get("truncated")) or choice.get("finish_reason") == "length"
                return choice["message"]["content"], truncated

            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            code = err.get("code", "")
            message = err.get("message") or body.get("detail") or res.text[:300]
            error = f"Bodhan translate {res.status_code} {code}: {message}"
            if res.status_code == 401 or code == "unsupported_language":
                raise FatalTranslationError(error)
            if res.status_code == 413:
                raise BatchError(error)
            if res.status_code != 429 and res.status_code < 500:
                raise RuntimeError(error)
            try:  # honour the server's Retry-After (seconds) when it sends one, up to RETRY_AFTER_MAX
                wait = max(min(float(res.headers.get("Retry-After", "")), RETRY_AFTER_MAX), wait)
            except ValueError:
                pass
        if attempt < RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)
