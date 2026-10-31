import difflib
import json
import os
import re
import time
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path

import requests

API_URL = "https://api.bodhan.ai/v1/audio/transcriptions"
MODEL = "indic-transcribe"
KEY_FILE = Path(__file__).resolve().parent.parent / "bodhan.txt"

RETRIES = 5  # extra attempts on 429 / 5xx / network errors
BACKOFF = 5  # seconds before the first retry, doubled each attempt
MAX_WAIT = 60  # cap on a single retry wait
RETRY_AFTER_MAX = 300  # cap on a wait the server asks for (Retry-After); above MAX_WAIT so we don't retry early
TIMEOUT = 60  # seconds per request

MERGE_MAX_GAP = 0.8  # seconds: same-speaker chunks closer than this are rejoined after ASR
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PUNCT_MODEL = "google/gemini-3.5-flash-lite"
PUNCT_WINDOW = 25  # chunks per transcript-cleanup request ...
PUNCT_CONTEXT = 2  # ... plus this many read-only neighbours on each side
PUNCT_TIMEOUT = 180
PUNCT_MAX_TOKENS = 16000  # reasoning tokens count toward this too
PUNCT_MAX_NEW = 0.3  # at most this share of a chunk's words may be new (misheard-word fixes), at least PUNCT_MIN_NEW
PUNCT_MIN_NEW = 3
SLOW_PAUSE_S = 1.0  # a merged chunk is slow only with a pause between its pieces at least this long ...
SLOW_PAUSE_SHARE = 0.2  # ... silence between its pieces at least this share of its span ...
SLOW_MIN_SPAN_S = 6.0  # ... and a span at least this long (pauses, not words/s: it doesn't depend on the language)

PUNCT_PROMPT = """You clean raw speech-to-text transcripts in {language} before they are translated and dubbed.
The video is titled: {title}

You get a JSON list of segments in time order with: id, speaker, text and editable (slow segments also have
slow=true and "parts": each piece's text, pause_after_s and slow, true where the speaker paused a long time after
it). Segments with editable=false are context only: never return them. Consecutive segments with the same speaker are parts of one
continuous turn, split at short pauses.

For every editable segment:
1. Misheard words: fix only words the recognizer clearly misheard: a word or phrase that sounds like what was
   said but makes no sense here (e.g. "great end descent" -> "gradient descent", "refine your gas" -> "refine
   your guess"), a misspelled or wrongly split/joined word, a technical term, name or number written wrongly, or
   a word in the wrong script. Use the title and the neighbouring segments to tell what was meant. When unsure,
   leave the word alone.
2. Punctuation: add natural sentence-ending punctuation (e.g. "।" for Hindi and Indic scripts that use danda,
   "." for English/Latin, "۔" for Urdu, plus "?" and "!") and clause punctuation (",") where the speaker
   naturally pauses or ends a sentence. Fix broken contractions (e.g. "didn t" -> "didn't", "that s" ->
   "that's"). If a sentence runs on into the next segment of the same speaker, do NOT end the first one with a
   sentence-ending mark.
   - Slow speech (editable segments with slow=true): the speaker pauses for a long time mid-sentence (e.g. while
     writing on a board), and the dub needs stopping points there so text-to-speech pauses with the speaker
     instead of rushing ahead. Only where a part has slow=true (a long pause after it):
     (a) Put a comma (",") after that part only if a written comma could stand there. Put NO mark when the part
         ends in an article, determiner, preposition, auxiliary or conjunction, or when the pause splits a name
         or noun phrase: never "the, model", "a set of, samples", "bounded by the, supremum", "realization, of,
         x". Skipping a pause is fine; a wrong comma is not.
     (b) If the text before that pause is a complete clause and the text after starts a new one (often with
         "and", "so", "but" or "then"), end the sentence there instead ("." / "।").
     Elsewhere in a slow segment, punctuate as usual: do not add commas where the speaker did not pause. A part
     boundary is not a sentence start: never capitalize the word after a comma.
3. Disfluencies that hurt dubbing: remove hesitation sounds ("uh", "um", "er", "ah", "hmm"), immediate stutters
   and false starts (e.g. "what what do I mean" -> "what do I mean", "why why is why am I" -> "why am I", "it will
   capture it will learn" -> "it will learn"), and filler tics ("right", "okay", "you know", "so basically") tacked
   onto clauses; lecturers often say "right" after almost every sentence: delete it, unless it is a real word of
   the sentence ("on the right side", "the right answer").

Keep numbers and symbols as the speaker said them, in words ("ninety nine percent", "zero and one"): never turn
them into digits or symbols ("99%", "0 and 1"), since the text is read aloud.
Do NOT paraphrase, reorder, translate, shorten, or fix the speaker's own grammar or style: spoken language is
fine as it is.
Return every editable segment with its id and full cleaned text.
Reply with JSON only: {{"items": [{{"id": <id>, "text": "<cleaned text>"}}]}}."""

# Languages accepted by the Bodhan speech-to-text API
LANGUAGES = {
    "as": "Assamese",
    "bn": "Bengali",
    "bhb": "Bhili",
    "bho": "Bhojpuri",
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


# Unicode script(s) each language is written in, as the first word of unicodedata.name(); Latin is always
# allowed too, since transcripts are often code-mixed with English words
SCRIPTS = {
    "as": ("BENGALI",),
    "bn": ("BENGALI",),
    "bhb": ("DEVANAGARI",),
    "bho": ("DEVANAGARI",),
    "brx": ("DEVANAGARI",),
    "doi": ("DEVANAGARI",),
    "en": (),
    "gu": ("GUJARATI",),
    "hi": ("DEVANAGARI",),
    "kn": ("KANNADA",),
    "ks": ("ARABIC", "DEVANAGARI"),
    "kok": ("DEVANAGARI",),
    "mai": ("DEVANAGARI",),
    "ml": ("MALAYALAM",),
    "mni": ("BENGALI", "MEETEI"),
    "mr": ("DEVANAGARI",),
    "ne": ("DEVANAGARI",),
    "or": ("ORIYA",),
    "pa": ("GURMUKHI",),
    "sa": ("DEVANAGARI",),
    "sat": ("OL", "DEVANAGARI"),
    "sd": ("ARABIC", "DEVANAGARI"),
    "ta": ("TAMIL",),
    "te": ("TELUGU",),
    "ur": ("ARABIC",),
}
JUNK_MIN_SCRIPT = 0.8  # below this share of letters in the language's script(s), a transcript is junk
JUNK_LOOP = re.compile(r"(.{1,10}?)\1{9,}", re.S)  # a 1-10 char unit repeated 10+ times in a row


def _junk_reason(text: str, language: str) -> str | None:
    """Why `text` looks like a degenerate ASR output (a repetition loop, or letters in the wrong script),
    or None when it looks like a real transcript."""
    if JUNK_LOOP.search(text):
        return "repetition loop"
    letters = [ch for ch in text if unicodedata.category(ch)[0] in "LM"]
    if not letters:
        return None
    allowed = ("LATIN",) + SCRIPTS.get(language, ())
    ok = sum(1 for ch in letters if unicodedata.name(ch, "").split(" ")[0] in allowed)
    if ok / len(letters) < JUNK_MIN_SCRIPT:
        return f"only {ok / len(letters):.0%} of letters in the {LANGUAGES.get(language, language)} script"
    return None


class FatalASRError(RuntimeError):
    """An error that will fail every chunk (bad key, no credit), so the whole job stops."""


def api_key(service: str) -> str:
    """Bodhan key for `service` ("asr" or "translate"): env BODHAN_<SERVICE>_KEY, else the
    "<service>:<key>" line in bodhan.txt."""
    env = f"BODHAN_{service.upper()}_KEY"
    key = os.environ.get(env, "").strip()
    if not key and KEY_FILE.exists():
        for line in KEY_FILE.read_text().splitlines():
            label, sep, value = line.partition(":")
            if sep and label.strip().lower() == service:
                key = value.strip()
                break
    if not key:
        raise RuntimeError(f"No Bodhan {service} key: set {env} or add a '{service}:<key>' line to {KEY_FILE.name}")
    return key


def merge_chunks(
    chunks: list[dict], max_gap: float = MERGE_MAX_GAP, joins: set[int] | None = None, commas: set[int] | None = None
) -> list[dict]:
    """Rejoin consecutive same-speaker chunks less than `max_gap` seconds apart (the mini-pause cuts made to keep
    each ASR request short), or, given `joins` (ids of the chunks whose cut to the next is joined; cuts.check_cuts),
    exactly across those cuts. Returns new chunks with ids 0..M-1; each keeps its parts, as they are, in
    "sub_chunks". The merged text is made of the parts' texts, with a comma after a part in `commas` (unless its
    text already ends in a mark); it counts as punctuated only if all of them are."""
    commas = commas or set()
    groups: list[list[dict]] = []
    for c in chunks:
        prev = groups[-1][-1] if groups else None
        if prev and prev["speaker"] == c["speaker"] and (
            prev["id"] in joins if joins is not None else c["start"] - prev["end"] < max_gap
        ):
            groups[-1].append(c)
        else:
            groups.append([c])

    merged = []
    for new_id, g in enumerate(groups):
        parts = [x for x in g if x.get("text") and x["text"].strip()]
        texts = [x["text"].strip() for x in parts]
        for k, x in enumerate(parts[:-1]):
            if x["id"] in commas and unicodedata.category(texts[k][-1])[0] in "LMN":
                texts[k] += ","
        errors = [x["error"] for x in g if x.get("error")]
        chunk = {
            "id": new_id,
            "speaker": g[0]["speaker"],
            "start": g[0]["start"],
            "end": g[-1]["end"],
            "cut_start": g[0]["cut_start"],
            "cut_end": g[-1]["cut_end"],
            "text": " ".join(texts) if texts else (None if errors else ""),
            "text_raw": " ".join(x.get("text_raw", x["text"]).strip() for x in g if x.get("text")),
            "punctuated": all(x.get("punctuated") for x in g if x.get("text")),
            "sub_chunks": [dict(x) for x in g],  # the short segments as they are: the UI shows them
        }
        if not texts and errors:
            chunk["error"] = "; ".join(errors)
        merged.append(chunk)
    mark_slow(merged)
    return merged


def mark_slow(chunks: list[dict]):
    """Mark slow speech, in place, from the pauses between a merged chunk's pieces (a word rate would depend on the
    language): the chunk gets "wps" (logged only), "pause_share" (silence between pieces / span) and "slow" (span
    >= SLOW_MIN_SPAN_S, a pause >= SLOW_PAUSE_S and pause_share >= SLOW_PAUSE_SHARE); each piece gets "wps",
    "pause_after" and "slow" (its chunk is slow and the pause after it is >= SLOW_PAUSE_S)."""
    for c in chunks:
        subs = [s for s in c.get("sub_chunks") or [] if s.get("text")]
        gaps = [round(float(b["start"] - a["end"]), 3) for a, b in zip(subs, subs[1:])]
        span = max(float(c["end"] - c["start"]), 1e-3)
        words = len((c.get("text_raw") or c.get("text") or "").split())
        c["wps"] = round(words / span, 2)
        c["pause_share"] = round(sum(max(g, 0.0) for g in gaps) / span, 3)
        c["slow"] = bool(
            words and span >= SLOW_MIN_SPAN_S and gaps and max(gaps) >= SLOW_PAUSE_S
            and c["pause_share"] >= SLOW_PAUSE_SHARE
        )
        for k, s in enumerate(subs):
            dur = max(float(s["end"] - s["start"]), 1e-3)
            s["wps"] = round(len((s.get("text_raw") or s.get("text") or "").split()) / dur, 2)
            s["pause_after"] = gaps[k] if k < len(gaps) else None
            s["slow"] = c["slow"] and k < len(gaps) and gaps[k] >= SLOW_PAUSE_S


class PunctWindowError(RuntimeError):
    """The reply was unusable in a way a smaller window may fix (too large, truncated, not JSON)."""


def punctuate_chunks(jdir: Path, chunks: list[dict], language: str, title: str = "", progress=None) -> dict:
    """Clean each chunk's text via OpenRouter (PUNCT_MODEL): punctuation, capitalization, misheard words, fillers
    and stutters; PUNCT_WINDOW chunks per request with PUNCT_CONTEXT read-only neighbours each side, the video's
    `title` as the topic. The text before is kept in chunk["text_raw"]; a reply that changes too much (_light_edit)
    is dropped. Chunks done are marked "punctuated", so a retry only sends the rest; chunks the user edited
    (text_edited) are left alone. A chunk the model fails on keeps its text. `progress()` is called after each
    window. chunks.json is written at the end. Raises RuntimeError if there is no OpenRouter key."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip() or api_key("openrouter")
    if any("sub_chunks" in c and "slow" not in c for c in chunks):
        mark_slow(chunks)
    order = [c for c in chunks if c.get("text")]
    todo = [c for c in order if not c.get("punctuated") and not c.get("text_edited")]
    stats = {"model": PUNCT_MODEL, "requests": 0, "attempts": 0, "changed": 0, "dropped": 0, "failed": 0, "raw_dir": jdir / "punctuate_raw"}
    ctx = {
        "system": PUNCT_PROMPT.format(language=LANGUAGES.get(language, language), title=title or "(unknown)"),
        "key": key,
        "order": order,
        "pos": {c["id"]: i for i, c in enumerate(order)},
    }
    stats["slow_chunks"] = sum(1 for c in chunks if c.get("slow"))
    stats["slow_sub_chunks"] = sum(1 for c in chunks for s in c.get("sub_chunks") or [] if s.get("slow"))
    stats["raw_dir"].mkdir(exist_ok=True)
    try:
        for i in range(0, len(todo), PUNCT_WINDOW):
            _punctuate_window(todo[i : i + PUNCT_WINDOW], ctx, stats)
            if progress:
                progress()
    finally:
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return {k: v for k, v in stats.items() if k != "raw_dir"}


def _punctuate_window(window, ctx, stats):
    """Clean one window in a request; on PunctWindowError split it in half and retry each half."""
    order, pos = ctx["order"], ctx["pos"]
    first, last = pos[window[0]["id"]], pos[window[-1]["id"]]
    editable = {c["id"] for c in window}
    items = [_punct_item(c, c["id"] in editable) for c in order[max(first - PUNCT_CONTEXT, 0) : last + PUNCT_CONTEXT + 1]]
    messages = [
        {"role": "system", "content": ctx["system"]},
        {"role": "user", "content": json.dumps(items, ensure_ascii=False, indent=1)},
    ]
    try:
        out = _punct_parse(_punct_request(messages, ctx["key"], stats))
    except PunctWindowError:
        if len(window) > 1:
            mid = len(window) // 2
            _punctuate_window(window[:mid], ctx, stats)
            _punctuate_window(window[mid:], ctx, stats)
        else:
            stats["failed"] += 1
        return
    except Exception:  # retries exhausted, or an error splitting won't fix: the texts stay as they are
        stats["failed"] += len(window)
        return
    for c in window:
        text = " ".join(str(out.get(c["id"]) or "").split())
        if not text:
            stats["failed"] += 1
        elif not _light_edit(c["text"], text):
            stats["dropped"] += 1
        else:
            if text != c["text"]:
                c.setdefault("text_raw", c["text"])
                c["text"] = text
                stats["changed"] += 1
            c["punctuated"] = True


def _punct_item(c: dict, editable: bool) -> dict:
    """One segment as the cleanup model sees it; an editable slow one also gets its parts with the pause after each
    (slow: a long one), so the model can put stopping points where the speaker paused."""
    item = {"id": c["id"], "speaker": c["speaker"], "text": c["text"], "editable": editable}
    if editable and c.get("slow"):
        item["slow"] = True
        subs = [s for s in c.get("sub_chunks") or [] if s.get("text")]
        if len(subs) > 1:
            item["parts"] = [
                {
                    "text": (s.get("text_raw") or s["text"]).strip(),
                    "pause_after_s": round(float(subs[k + 1]["start"] - s["end"]), 2) if k + 1 < len(subs) else None,
                    "slow": bool(s.get("slow")),
                }
                for k, s in enumerate(subs)
            ]
    return item


def _light_edit(a: str, b: str) -> bool:
    """Whether `b` is a cleaned version of `a`, not a rewrite: case and punctuation don't count, words may be
    dropped (fillers, stutters) but not most of them, and only a few may be new (misheard-word fixes)."""
    def words(s):
        cleaned = "".join(ch.lower() if ch.isalnum() or unicodedata.category(ch).startswith("M") else " " for ch in s)
        return cleaned.split()

    wa, wb = words(a), words(b)
    if not wb or (len(wa) > 6 and len(wb) < len(wa) * 0.5):
        return False
    same = sum(m.size for m in difflib.SequenceMatcher(None, wa, wb, autojunk=False).get_matching_blocks())
    return len(wb) - same <= max(PUNCT_MIN_NEW, PUNCT_MAX_NEW * len(wa))


def _punct_parse(content: str) -> dict[int, str]:
    """{id: text} from the reply. Raises PunctWindowError if it isn't the JSON asked for."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        data = json.loads(text)
    except ValueError:
        a, b = text.find("{"), text.rfind("}")  # some providers ignore response_format and add prose
        try:
            data = json.loads(text[a : b + 1]) if a != -1 and b > a else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise PunctWindowError(f"reply is not the expected JSON: {content[:200]}")
    return {i["id"]: i.get("text") for i in data["items"] if isinstance(i, dict) and isinstance(i.get("id"), int)}


def _punct_save_raw(stats, payload, res=None, error=None):
    """Write one attempt (request body + raw response) to punctuate_raw/<attempt>.json. No API key is stored."""
    stats["attempts"] += 1
    record = {
        "attempt": stats["attempts"],
        "time": datetime.now(timezone.utc).isoformat(),
        "url": OPENROUTER_URL,
        "request": payload,
    }
    if res is not None:
        record["status"] = res.status_code
        try:
            record["body"] = res.json()
        except ValueError:
            record["body"] = res.text
    else:
        record["error"] = error
    (stats["raw_dir"] / f"{stats['attempts']:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))


def _punct_request(messages, key, stats) -> str:
    """POST one transcript-cleanup request. Returns the reply's text."""
    error = "unknown error"
    payload = {
        "model": PUNCT_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": PUNCT_MAX_TOKENS,
        "response_format": {"type": "json_object"},  # plain JSON mode, as cuts._openrouter
    }
    for attempt in range(RETRIES + 1):
        wait = min(BACKOFF * 2**attempt, MAX_WAIT)
        try:
            res = requests.post(
                OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=PUNCT_TIMEOUT
            )
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
            _punct_save_raw(stats, payload, error=error)
        else:
            _punct_save_raw(stats, payload, res=res)
            stats["requests"] += 1
            try:
                body = res.json()
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}

            if res.status_code == 200 and body.get("choices"):
                choice = body["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise PunctWindowError("reply was cut off at the output limit")
                return (choice.get("message") or {}).get("content") or ""

            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            error = f"OpenRouter {res.status_code}: {err.get('message') or res.text[:300]}"
            if res.status_code in (401, 402, 403):
                raise FatalASRError(error)
            if res.status_code == 413:
                raise PunctWindowError(error)
            if res.status_code != 429 and res.status_code < 500:
                raise RuntimeError(error)
            try:  # honour the server's Retry-After (seconds) when it sends one, up to RETRY_AFTER_MAX
                wait = max(min(float(res.headers.get("Retry-After", "")), RETRY_AFTER_MAX), wait)
            except ValueError:
                pass
        if attempt < RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)


def _cut_chunks(jdir: Path, chunks: list[dict]) -> list[Path]:
    """Write each chunk's [cut_start, cut_end] of vocals_16k.wav to chunks/<id>.wav."""
    out_dir = jdir / "chunks"
    out_dir.mkdir(exist_ok=True)

    with wave.open(str(jdir / "vocals_16k.wav"), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    width = params.sampwidth * params.nchannels
    sr = params.framerate

    paths = []
    for c in chunks:
        a, b = int(c["cut_start"] * sr), int(c["cut_end"] * sr)
        path = out_dir / f"{c['id']:04d}.wav"
        with wave.open(str(path), "wb") as o:
            o.setnchannels(params.nchannels)
            o.setsampwidth(params.sampwidth)
            o.setframerate(sr)
            o.writeframes(frames[a * width : b * width])
        paths.append(path)
    return paths


def _transcribe_one(path: Path, language: str, key: str) -> str:
    error = "unknown error"
    for attempt in range(RETRIES + 1):
        wait = min(BACKOFF * 2**attempt, MAX_WAIT)
        try:
            with path.open("rb") as f:
                res = requests.post(
                    API_URL,
                    headers={"Authorization": f"Bearer {key}"},
                    data={"model": MODEL, "language": language},
                    files={"file": (path.name, f, "audio/wav")},
                    timeout=TIMEOUT,
                )
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
        else:
            if res.status_code == 200:
                text = res.json()["text"]
                junk = _junk_reason(text, language)
                if not junk:
                    return text
                error = f"junk transcript ({junk}): {text[:80]!r}"  # retried like a 5xx
            else:
                error = f"Bodhan API {res.status_code}: {res.text[:300]}"
                if res.status_code in (401, 402, 403):
                    raise FatalASRError(error)
                if res.status_code != 429 and res.status_code < 500:
                    raise RuntimeError(error)
                try:  # honour the server's Retry-After (seconds) when it sends one, up to RETRY_AFTER_MAX
                    wait = max(min(float(res.headers.get("Retry-After", "")), RETRY_AFTER_MAX), wait)
                except ValueError:
                    pass
        if attempt < RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)
