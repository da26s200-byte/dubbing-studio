"""The pause-split chunking (studio flow): every speaker turn is cut at each silence
of at least SPLIT_PAUSE_S (and, past the ASR's limit, as diarization._split does); each piece is transcribed with
one ASR attempt (the caller tries failed pieces once more, ASR_RETRY_DELAY_S later); then Gemma (BOUNDARY_MODEL)
reads the transcripts in order and judges every cut between two pieces of one speaker: kept, unless translating
the two sides separately would hurt the translation. Then a segment still shorter than SHORT_CHARS is joined, with a
comma, to a near neighbour by the same speaker; when both neighbours are near, Gemma picks the side it belongs
with (short_joins). The pieces across the joined cuts are merged into one chunk (asr.merge_chunks with `joins`),
so the dub can pause wherever the speaker paused."""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from . import asr, diarization, review, translate
except ImportError:
    import asr
    import diarization
    import review
    import translate

SPLIT_PAUSE_S = 0.5  # a silence at least this long inside a speaker's turn cuts it
MIN_PIECE_S = 0.2  # pieces of speech shorter than this between cuts are dropped (clicks, breaths)
ASR_RETRY_DELAY_S = 180  # a piece whose ASR failed is tried once more, this long after it failed, once all are done
DEFAULT_TARGET = "hi"  # the target language the cut check assumes for a run transcribed without one
SHORT_CHARS = 100  # after the cut check, a segment with fewer characters than this is joined (with a comma) ...
SHORT_GAP_S = 1.5  # ... to a same-speaker neighbour at most this far away; Gemma picks which when both are (short_joins)

BOUNDARY_MODEL = "google/gemma-4-26b-a4b-it"  # Gemma 4 MoE (26B, 4B active)
BOUNDARY_WINDOW = 40  # pieces per request ...
BOUNDARY_CONTEXT = 3  # ... plus this many read-only neighbours on each side
BOUNDARY_TIMEOUT = 180
BOUNDARY_REASONING = {"enabled": False}  # off: Gemma thinking at any effort (or budget) second-guesses until it runs out of tokens; the reason in the reply comes before the verdict instead
BOUNDARY_MAX_TOKENS = 16000  # reasoning tokens count toward this too
BOUNDARY_PROMPT = """You prepare a speech transcript in {source} for machine translation into {target}. The audio was \
cut at every silence of at least {pause} s (or, where "forced" is true, because it was too long for speech \
recognition), and each piece will be translated ON ITS OWN. Pieces are kept apart wherever possible, so the dub \
can pause where the speaker paused; two pieces are joined only if translating them separately would hurt the \
translation. The transcripts have no punctuation yet, so a missing full stop means nothing.

"pieces" are the pieces in order. Judge EVERY cut listed in "cuts": a cut is between piece "after" and the next \
piece by the same speaker; "joined" is the end of the one and the start of the next, with "|" at the cut.

For each cut, ask: if the left side and the right side are each translated into {target} on their own, will both \
come out grammatical, with the same meaning, and with the words in the same order as translating them together?
- "keep" if yes. Typically: the cut is between two sentences, or between two clauses that each have their own \
subject and verb, even when the right side starts with a conjunction ("and", "but", "so", "when", "because", \
"और", "लेकिन", "क्योंकि", "जब") or the left side ends with one; or the cut is after a discourse word ("so", \
"okay", "now").
- "join" if no. Typically: one side has no verb of its own; a verb is cut from its object or complement; a \
preposition or postposition from its noun; a subject from its verb; an auxiliary or modal from its main verb \
("have to | decide", "करने | की कोशिश"); a relative or complement clause from the word it belongs to; a fixed \
pair is split ("instead of X | Y", "not only | but also", "between A | and B"); or a name, number or list is split. \
Also "join" whenever {target} would have to move a word across the cut to say it naturally: in verb-final \
languages (most Indic languages) the verb, auxiliaries and postpositions go at the end of the clause, and \
relative and adverbial clauses go before what they modify, so a clause split anywhere but at its edge must be \
joined.

Examples ("joined" -> verdict: reason), with the {target} problem they avoid shown for Hindi:
- "we trained the model on this data | and then it learned the function" -> keep: two clauses, each with its own \
subject and verb; "and" only starts the second.
- "we tried the bigger network and | the results did not improve" -> keep: "and" ends a complete clause and a new \
clause starts; nothing moves.
- "when you add more layers | the training gets slower" -> keep: the "when" clause is complete ("जब आप और layers \
जोड़ते हैं" | "तो training धीमी हो जाती है"), the order stays.
- "this is why the loss goes down | now let us look at the gradient" -> keep: two sentences.
- "right | so let us start with vectors" -> keep: a discourse word, then a new sentence.
- "मैंने model को train किया | और फिर उसने pattern सीख लिया" -> keep: two clauses, each with its own verb.
- "the model fails in cases where | the training data has no labels" -> join: "where" opens a relative clause that \
continues on the right; alone, the left ends on a dangling "जहाँ".
- "most engineers, instead of deriving the formula, | decided to look it up" -> join: the subject is on the left and \
its verb on the right; alone, "decided to look it up" has no subject.
- "we minimise the loss using | gradient descent" -> join: "using" is cut from its object, and the verb must come \
last ("gradient descent का इस्तेमाल करके loss को कम करते हैं").
- "you will need to | retrain the model" -> join: a modal cut from its main verb ("model को फिर से train करना \
पड़ेगा" is one verb group).
- "this result is important for | understanding why the model fails" -> join: the complement must move before \
"important" ("model क्यों fail होता है यह समझने के लिए यह result ज़रूरी है").
- "the first convolutional | layer of the network" -> join: a noun phrase is split.
- "हम इस data को | अलग-अलग groups में बाँटने की कोशिश करते हैं" -> join: the object is cut from its verb.

Give the reason first, naming the word or phrase that would be stranded or moved (or the two clauses for \
"keep"), then the verdict. Return every cut exactly once.
Reply with JSON only: {{"cuts": [{{"after": <id>, "reason": "<why>", "verdict": "keep" | "join"}}]}}."""

JOIN_WORDS = 8  # words from each side of a cut shown in "joined"

SHORT_PROMPT = """You prepare a speech transcript in {source} for dubbing. Each segment is dubbed on its own, and \
the segments listed in "segments" are too short for that (fewer than {short} characters), so each will be joined \
to the segment just before it or the one just after it ("previous": the end of the segment before it; "next": the \
start of the segment after it; each with the pause between). The transcripts have no punctuation yet.

For each segment, choose the side it belongs with by meaning: the side whose sentence or thought it completes \
or continues. A segment that finishes a thought ("as we saw last time", "in this case", a trailing object or \
phrase) joins "previous"; one that opens a thought ("so in this class", "now", "the next thing is", or a clause \
that leads into the next sentence, like "while doing it" or "when we do that") joins "next". Only when neither \
side fits better, choose the shorter pause.

Give the reason first, then the choice. Return every segment exactly once.
Reply with JSON only, "id" copied from the segment: {{"segments": [{{"id": <id>, "reason": "<why>", "join": "previous" | \
"next"}}]}}."""


def split_pieces(jdir: Path, turns) -> list[dict]:
    """Same-speaker turns less than SPLIT_PAUSE_S apart are joined; each of the resulting turns is cut at every
    silence of at least SPLIT_PAUSE_S inside it (vocals_16k.wav's envelope, the diarization's silence level), and
    a piece still longer than the ASR's limit is cut as the studio does (diarization._split). Each piece gets
    pause_after (the silence to the next piece, None for the last), forced_after (cut for length, not at a
    pause) and the ASR cut with diarization.PAD around it."""
    env, duration = diarization._load_envelope(jdir / "vocals_16k.wav")
    quiet = env < diarization._silence_threshold(env)
    long_pauses = [
        (a * diarization.FRAME_S, b * diarization.FRAME_S)
        for a, b in diarization._quiet_runs(quiet)
        if (b - a) * diarization.FRAME_S >= SPLIT_PAUSE_S
    ]
    short_pauses = diarization.find_pauses(env)

    joined = []
    for speaker, start, end in turns:
        if joined and joined[-1][0] == speaker and start - joined[-1][2] < SPLIT_PAUSE_S:
            joined[-1][2] = end
        else:
            joined.append([speaker, start, end])

    raw = []  # (speaker, start, end, forced cut after)
    for speaker, start, end in joined:
        cuts = [(a, b) for a, b in long_pauses if start < a and b < end]
        bounds = [start] + [x for a, b in cuts for x in (a, b)] + [end]
        for k in range(0, len(bounds), 2):
            a, b = bounds[k], bounds[k + 1]
            if b - a < MIN_PIECE_S:
                continue
            parts = diarization._split(a, b, env, short_pauses)
            raw += [(speaker, x, y, i < len(parts) - 1) for i, (x, y) in enumerate(parts)]

    pieces = []
    for i, (speaker, start, end, forced) in enumerate(raw):
        lo = start - diarization.PAD
        if i > 0:
            lo = max(lo, (raw[i - 1][2] + start) / 2)
        hi = end + diarization.PAD
        if i + 1 < len(raw):
            hi = min(hi, (end + raw[i + 1][1]) / 2)
        pieces.append({
            "id": i,
            "speaker": speaker,
            "start": round(float(start), 3),
            "end": round(float(end), 3),
            "cut_start": round(float(max(lo, 0.0)), 3),
            "cut_end": round(float(min(hi, duration)), 3),
            "pause_after": round(float(raw[i + 1][1] - end), 3) if i + 1 < len(raw) else None,
            "forced_after": forced,
            "text": None,
        })
    (jdir / "pieces.json").write_text(json.dumps(pieces, indent=2))
    return pieces


class RateLimited(RuntimeError):
    """Bodhan answered 429; `wait` is the Retry-After it sent, if any."""

    def __init__(self, message, wait=None):
        super().__init__(message)
        self.wait = wait


def transcribe_pieces(jdir: Path, pieces: list[dict], language: str, progress=None):
    """One ASR attempt per piece, no immediate retry: a junk transcript or server error is recorded as the piece's
    error (with failed_at, its time) and the next piece goes on; the caller decides when to try again. A rate limit
    (429) is not the piece's failure: the pass pauses (Retry-After, else asr.BACKOFF doubling to asr.MAX_WAIT) and
    sends it again, up to asr.RETRIES times, before it counts as failed. Counts each piece's attempts. A bad key
    or no credit (FatalASRError) stops the run."""
    key = asr.api_key("asr")
    for p, path in zip(pieces, asr._cut_chunks(jdir, pieces)):
        p["asr_attempts"] = p.get("asr_attempts", 0) + 1
        for throttled in range(asr.RETRIES + 1):
            try:
                p["text"], p["error"] = _asr_once(path, language, key), None
                p.pop("failed_at", None)
            except asr.FatalASRError:
                raise
            except RateLimited as exc:
                p["text"], p["error"], p["failed_at"] = None, str(exc), time.time()
                if throttled < asr.RETRIES:
                    time.sleep(min(exc.wait or asr.BACKOFF * 2**throttled, asr.RETRY_AFTER_MAX))
                    continue
            except Exception as exc:
                p["text"], p["error"], p["failed_at"] = None, str(exc), time.time()
            break
        if progress:
            progress()


def _asr_once(path: Path, language: str, key: str) -> str:
    """One Bodhan ASR request (as asr._transcribe_one, without its retries). Raises on a junk transcript."""
    try:
        with path.open("rb") as f:
            res = requests.post(
                asr.API_URL,
                headers={"Authorization": f"Bearer {key}"},
                data={"model": asr.MODEL, "language": language},
                files={"file": (path.name, f, "audio/wav")},
                timeout=asr.TIMEOUT,
            )
    except requests.RequestException as exc:
        raise RuntimeError(f"request failed: {exc}")
    if res.status_code == 429:
        try:
            wait = float(res.headers.get("Retry-After", ""))
        except ValueError:
            wait = None
        raise RateLimited(f"Bodhan API 429: {res.text[:300]}", wait)
    if res.status_code != 200:
        error = f"Bodhan API {res.status_code}: {res.text[:300]}"
        raise (asr.FatalASRError if res.status_code in (401, 402, 403) else RuntimeError)(error)
    text = res.json()["text"]
    junk = asr._junk_reason(text, language)
    if junk:
        raise RuntimeError(f"junk transcript ({junk}): {text[:80]!r}")
    return text


def _raw(p: dict) -> str:
    """A piece's transcript as ASR gave it (before punctuation), else its text."""
    return p.get("text_raw") or p.get("text") or ""


def check_cuts(jdir: Path, pieces: list[dict], language: str, target: str):
    """BOUNDARY_MODEL gives a verdict on every same-speaker cut between transcribed pieces: keep, or join
    (translating the two sides into `target` separately would hurt the translation), BOUNDARY_WINDOW pieces per request with BOUNDARY_CONTEXT
    read-only neighbours. Sets piece["cut_verdict"] and piece["cut_reason"] on the piece before each cut judged.
    Returns ({ids of the pieces before a cut to join}, stats). A window that fails merges nothing
    (stats["failed"]); a cut the reply leaves out stays (stats["unjudged"])."""
    key = review.api_key()
    system = BOUNDARY_PROMPT.format(
        source=asr.LANGUAGES.get(language, language), target=translate.LANGUAGES.get(target, target), pause=SPLIT_PAUSE_S
    )
    raw_dir = jdir / "cut_check_raw"
    raw_dir.mkdir(exist_ok=True)
    stats = {"model": BOUNDARY_MODEL, "reasoning": BOUNDARY_REASONING, "requests": 0, "reasoning_tokens": 0, "failed": 0, "unjudged": 0, "dropped": []}
    merges = set()
    for w in range(0, len(pieces), BOUNDARY_WINDOW):
        lo, hi = w, min(w + BOUNDARY_WINDOW, len(pieces))
        first = max(lo - BOUNDARY_CONTEXT, 0)
        items = [
            {
                "id": p["id"],
                "speaker": p["speaker"],
                "text": _raw(p),
                "pause_after_s": p.get("pause_after"),
                "forced": bool(p.get("forced_after")),
            }
            for p in pieces[first : hi + BOUNDARY_CONTEXT]
        ]
        todo = {
            i: {
                "after": i,
                "pause_s": pieces[i].get("pause_after"),
                "forced": bool(pieces[i].get("forced_after")),
                "joined": " ".join(_raw(pieces[i]).split()[-JOIN_WORDS:]) + " | "
                + " ".join(_raw(pieces[i + 1]).split()[:JOIN_WORDS]),
            }
            for i in range(lo, min(hi, len(pieces) - 1))
            if pieces[i + 1]["speaker"] == pieces[i]["speaker"] and pieces[i].get("text") and pieces[i + 1].get("text")
        }
        if not todo:
            continue
        user = {"pieces": items, "cuts": list(todo.values())}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=1)}]
        try:
            reply = _openrouter(messages, key, raw_dir, stats)
        except Exception as exc:
            stats["failed"] += 1
            stats.setdefault("errors", []).append(str(exc))
            continue
        judged = set()
        for m in reply.get("cuts") or []:
            i = m.get("after") if isinstance(m, dict) else None
            if i not in todo or i in judged or m.get("verdict") not in ("keep", "join"):
                stats["dropped"].append({"cut": m, "why": "not a listed cut, judged twice, or no verdict"})
                continue
            judged.add(i)
            pieces[i]["cut_verdict"] = m["verdict"]
            pieces[i]["cut_reason"] = str(m.get("reason") or "").strip() or None
            if m["verdict"] == "join":
                merges.add(i)
        stats["unjudged"] += len(todo) - len(judged)
    stats["merged_cuts"] = len(merges)
    return merges, stats


def short_joins(jdir: Path, pieces: list[dict], joins: set[int], language: str) -> tuple[set[int], dict]:
    """After the joins in `joins` (ids of the pieces whose cut to the next is joined; check_cuts), the cuts to join
    as well so that no segment has fewer than SHORT_CHARS characters: such a segment is joined to a neighbour by
    the same speaker at most SHORT_GAP_S away. With only one such neighbour, to that one; with two, BOUNDARY_MODEL
    picks the side it belongs with by meaning (SHORT_PROMPT, one request per round for all of them; if it fails,
    the nearer side). Repeated in rounds until no short segment can be joined. Segments without text are left as
    they are. Sets piece["short_reason"] on the piece before each cut added (cleared everywhere else). Returns
    ({ids of those pieces}, stats)."""
    for p in pieces:
        p.pop("short_reason", None)
    added: set[int] = set()
    stats = {"requests": 0, "reasoning_tokens": 0, "asked": 0, "failed": 0, "rounds": 0}
    system = SHORT_PROMPT.format(source=asr.LANGUAGES.get(language, language), short=SHORT_CHARS)
    raw_dir = jdir / "short_check_raw"
    raw_dir.mkdir(exist_ok=True)
    key = None
    while True:
        groups: list[list[dict]] = []
        for p in pieces:
            if groups and groups[-1][-1]["id"] in joins | added:
                groups[-1].append(p)
            else:
                groups.append([p])
        texts = [" ".join(_raw(p) for p in g).strip() for g in groups]
        options = {}  # short segment k -> {"before" | "after": (id of the piece before the cut, gap)}
        for k, g in enumerate(groups):
            if not texts[k] or len(texts[k]) >= SHORT_CHARS:
                continue
            for side, j in (("before", k - 1), ("after", k + 1)):
                if not 0 <= j < len(groups) or not texts[j] or groups[j][0]["speaker"] != g[0]["speaker"]:
                    continue
                a, b = (groups[j], g) if j < k else (g, groups[j])
                gap = b[0]["start"] - a[-1]["end"]
                if gap <= SHORT_GAP_S:
                    options.setdefault(k, {})[side] = (a[-1]["id"], gap)
        if not options:
            return added, stats
        stats["rounds"] += 1
        choices = {}  # k -> (side, reason)
        two = [k for k, o in options.items() if len(o) == 2]
        if two:
            stats["asked"] += len(two)
            items = [
                {
                    "id": k,
                    "text": texts[k],
                    "previous": {"text": " ".join(texts[k - 1].split()[-JOIN_WORDS * 2 :]), "pause_s": round(options[k]["before"][1], 2)},
                    "next": {"text": " ".join(texts[k + 1].split()[: JOIN_WORDS * 2]), "pause_s": round(options[k]["after"][1], 2)},
                }
                for k in two
            ]
            messages = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps({"segments": items}, ensure_ascii=False, indent=1)}]
            try:
                key = key or review.api_key()
                by_text = {texts[k]: k for k in two}
                for m in _openrouter(messages, key, raw_dir, stats).get("segments") or []:
                    k = m.get("id", m.get("segment")) if isinstance(m, dict) else None
                    k = k if k in two else by_text.get(str(k).strip())  # it sometimes gives the text instead
                    if k in two and m.get("join") in ("previous", "next"):
                        choices[k] = ("before" if m["join"] == "previous" else "after", str(m.get("reason") or "").strip())
            except Exception as exc:
                stats["failed"] += 1
                stats.setdefault("errors", []).append(str(exc))
        for k, o in options.items():
            n = len(texts[k])
            if k in choices:
                side, why = choices[k]
                reason = f"short segment ({n} characters), joined to the {'previous' if side == 'before' else 'next'} one: {why}"
            elif len(o) == 1:
                side = next(iter(o))
                reason = f"short segment ({n} characters), joined to its only neighbour within {SHORT_GAP_S} s"
            else:  # the model didn't choose
                side = min(o, key=lambda x: o[x][1])
                reason = f"short segment ({n} characters), joined to its nearer neighbour ({o[side][1]:.1f} s)"
            pid = o[side][0]
            added.add(pid)
            next(p for p in pieces if p["id"] == pid)["short_reason"] = reason


def _openrouter(messages, key, raw_dir: Path, stats) -> dict:
    """One structured-output request to OpenRouter, with the studio's retries; each attempt is saved to raw_dir.
    Returns the parsed JSON."""
    payload = {
        "model": BOUNDARY_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": BOUNDARY_MAX_TOKENS,
        "reasoning": BOUNDARY_REASONING,
        # plain JSON mode: a strict json_schema made Gemini flash-lite skip thinking; Gemma is free to reply in JSON mode
        "response_format": {"type": "json_object"},
    }
    error = "unknown error"
    for attempt in range(asr.RETRIES + 1):
        wait = min(asr.BACKOFF * 2**attempt, asr.MAX_WAIT)
        record = {"time": datetime.now(timezone.utc).isoformat(), "request": payload}
        try:
            res = requests.post(asr.OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=BOUNDARY_TIMEOUT)
        except requests.RequestException as exc:
            error = record["error"] = f"request failed: {exc}"
        else:
            stats["requests"] += 1
            try:
                body = res.json()
            except ValueError:
                body = {}
            record.update(status=res.status_code, body=body or res.text)
            if res.status_code == 200 and isinstance(body, dict) and body.get("choices"):
                usage = (body.get("usage") or {}).get("completion_tokens_details") or {}
                stats["reasoning_tokens"] += usage.get("reasoning_tokens") or 0
                if body["choices"][0].get("finish_reason") == "length":  # the same request would run out again
                    (raw_dir / f"{stats['requests']:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
                    raise RuntimeError(f"reply cut off at {BOUNDARY_MAX_TOKENS} tokens (reasoning ran on)")
                content = (body["choices"][0].get("message") or {}).get("content") or ""
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                try:
                    data = json.loads(text[text.find("{") : text.rfind("}") + 1])
                except ValueError:
                    data = None
                if isinstance(data, dict):
                    (raw_dir / f"{stats['requests']:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
                    return data
                error = f"reply is not JSON: {content[:200]}"
            else:
                error = f"OpenRouter {res.status_code}: {res.text[:300]}"
                if res.status_code in (401, 402, 403) or (res.status_code != 429 and res.status_code < 500):
                    (raw_dir / f"{stats['requests']:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
                    raise RuntimeError(error)
        (raw_dir / f"fail-{stats['requests']:04d}-{attempt}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        if attempt < asr.RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)
