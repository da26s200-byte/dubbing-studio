"""Translation check (studio flow, its own stage after translation): an LLM on OpenRouter reads each segment's source and
translation in windows and rewrites the ones that would run too long after TTS, read wrongly where a sentence
was cut across segments, have bad punctuation, are wrong, untranslated or too literal, or use the wrong
grammatical gender for the speaker (each segment is sent with its speaker's detected gender). Edits are applied at
once; the original translation is kept so each edit can be undone."""

import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from . import asr, gender, translate
    from .asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX
except ImportError:
    import asr
    import gender
    import translate
    from asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.5-flash-lite"
TIMEOUT = 180  # seconds per request
MAX_OUTPUT_TOKENS = 8000  # a reply cut off at this length counts as failed (the window is split)

WINDOW = 25  # at most this many segments to check per request ...
CONTEXT = 2  # ... plus this many read-only neighbours on each side, so sentences cut at the edges are seen

# Length budget: the translation should fit its slot at TARGET_SPEED, below the dub's MAX_SPEED cap
TARGET_SPEED = 1.4
DEFAULT_CHARS_PER_SEC = 14.0  # TTS speaking rate in characters per second, before any speed-up
CHARS_PER_SEC: dict[str, float] = {}  # per-language overrides; tune from dubbed runs (tts_duration vs len(translation))

ISSUES = ["too_long", "punctuation", "broken_meaning", "mistranslation", "too_literal", "untranslated", "wrong_gender"]

PROMPT = """You check a machine translation from {source} into {target} that will be dubbed over a video.
Each segment is one speaker's line in a fixed time slot; its translation is read aloud by text-to-speech and
must fit that slot.

You get a JSON list of segments in time order with: id, speaker, gender (the speaker's gender detected from
their voice: "female", "male" or "unknown"), slot_s (slot length in seconds), source (the original words, from
speech recognition, so it may have small errors), translation, chars (length of the translation), max_chars
(the longest translation that fits the slot at a comfortable pace) and editable.
Segments with editable=false are context only: never return them.

Check every editable segment for these issues:
- too_long: chars is above max_chars. Shorten it to at most max_chars: drop filler, use shorter natural
  wording, but keep everything that matters to the meaning.
- untranslated: left wholly or partly in the source language, empty, or not a translation of the source.
- mistranslation: the meaning is wrong, or the text is garbled.
- too_literal: translated word for word, so each word may be right but the intended meaning, idiom or tone
  is lost. Rewrite it the way a native {target} speaker would actually say it, keeping the meaning of the
  source, not its wording. Judge this from the source and the neighbouring segments.
- broken_meaning: a sentence cut across segment boundaries reads wrongly, e.g. a dangling clause, or the
  verb ended up in the next segment so neither half makes sense. Fix it by rewriting this segment and, if
  needed, the neighbouring segment of the same speaker, moving words between them so each reads naturally
  in order. Never move words between different speakers or into a non-editable segment.
- punctuation: missing, wrong or excessive punctuation that would make the speech pause in the wrong place.
  Segments with slow=true intentionally have extra commas and sentence breaks so the dub pauses with a slow
  speaker: keep those stopping points in the translation (only fix a mark that breaks a tight two-word unit).
- wrong_gender: where the speaker talks about themselves (I, me, my, we when it plainly means the speaker),
  the gendered forms in {target} (verb endings, adjectives, participles, self-references) must match the
  segment's gender, e.g. in Hindi a female speaker says "main ja rahi hoon", a male "main ja raha hoon".
  Fix only those forms, keeping the rest of the wording. Do not flag it:
  - when gender is "unknown";
  - for words about other people (he, she, you, they, names): those follow the source and the context,
    not the speaker's gender;
  - when the translation has no gendered form referring to the speaker, or the source makes clear the
    speaker is quoting or playing someone else.

Rules:
- Keep the style of the existing translation: the same script and register, and keep common English words
  in English where the translation already does.
- Keep names, numbers and terms consistent with the neighbouring segments.
- Leave segments that are fine alone. Return only the segments you change.
- For each one give all the issues it had and its full new translation. Do not explain your changes.

Reply with JSON only: {"edits": [{"id": <id>, "issues": [<issue>, ...], "translation": "<new text>"}]}.
If nothing needs changing, reply {"edits": []}."""

# The user message and the JSON reply format (SCHEMA, sent as response_format) are fixed. The system prompt
# above is the default; a run can use its own, edited in the UI. {source} and {target} are replaced by the
# language names.
MAX_PROMPT_CHARS = 20000


def render_prompt(prompt: str, source: str, target: str) -> str:
    return prompt.replace("{source}", source).replace("{target}", target)


def edited(chunk: dict) -> bool:
    """The user rewrote this segment's transcript or translation: the check leaves it alone."""
    return bool(chunk.get("text_edited") or chunk.get("translation_edited"))


def reset_chunks(chunks: list[dict]):
    """Undo a check: put back the translation from before it, and forget its results, so the check can run
    again from scratch."""
    for c in chunks:
        if c.get("translation_original") is not None:
            set_translation(c, c["translation_original"])
        for k in ("translation_original", "review", "review_error"):
            c.pop(k, None)


SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "issues": {"type": "array", "items": {"type": "string", "enum": ISSUES}},
                    "translation": {"type": "string"},
                },
                "required": ["id", "issues", "translation"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["edits"],
    "additionalProperties": False,
}


class FatalReviewError(RuntimeError):
    """An error every request would hit (bad key, no credit), so the check stops."""


class WindowError(RuntimeError):
    """The reply was unusable in a way a smaller window may fix (truncated, not JSON)."""


def api_key() -> str:
    """OpenRouter key: env OPENROUTER_API_KEY, else an "openrouter:<key>" line in bodhan.txt."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    try:
        return asr.api_key("openrouter")
    except RuntimeError:
        raise RuntimeError(
            f"No OpenRouter key: set OPENROUTER_API_KEY or add an 'openrouter:<key>' line to {asr.KEY_FILE.name}"
        )


def max_chars(chunk: dict, lang: str) -> int:
    """The longest translation that fits the chunk's slot at TARGET_SPEED."""
    rate = CHARS_PER_SEC.get(lang, DEFAULT_CHARS_PER_SEC)
    return max(int(math.floor((chunk["end"] - chunk["start"]) * TARGET_SPEED * rate)), 1)


def set_translation(chunk: dict, text: str):
    """Change a chunk's translation. A TTS clip made before tts_text was stored is marked as made from the old
    text, so the next dub redoes it. The short segments get their slices of the new text."""
    if chunk.get("tts_file"):
        chunk.setdefault("tts_text", chunk.get("translation"))
    chunk["translation"] = text
    translate.map_sub_translations(chunk)  # the short segments' slices follow


def review_chunks(
    jdir: Path, chunks: list[dict], source: str, target: str, progress=None, prompt: str | None = None
) -> dict:
    """Check every chunk with a translation and no review yet (or a failed one), WINDOW at a time. Sets
    chunk["review"] = {issues, reason, model, changed, reviewed} (reviewed: the edited text, when changed) and,
    on the first edit, chunk["translation_original"]; or chunk["review_error"]. `progress()` is called after
    each window. `prompt` is the system prompt to use (default PROMPT). chunks.json is written even when the
    check stops early. Segments the user edited (text_edited / translation_edited) are never checked.
    Returns stats."""
    todo = [c for c in chunks if c.get("translation") and "review" not in c and not edited(c)]
    for c in todo:
        c.pop("review_error", None)
    stats = {"model": MODEL, "requests": 0, "attempts": 0, "dropped": [], "error": None, "raw_dir": jdir / "review_raw"}
    if not todo:
        return _finish(jdir, chunks, stats)

    ctx = {
        "source": asr.LANGUAGES.get(source, source),
        "target": translate.LANGUAGES.get(target, target),
        "lang": target,
        "prompt": prompt or PROMPT,
        "key": api_key(),
        "genders": gender.load(jdir),  # {speaker: "female" | "male"}; read now, so a corrected gender is used
        # neighbours are looked up among all translated chunks, reviewed or not
        "order": [c for c in chunks if c.get("translation")],
    }
    ctx["pos"] = {c["id"]: i for i, c in enumerate(ctx["order"])}
    stats["raw_dir"].mkdir(exist_ok=True)

    try:
        for i in range(0, len(todo), WINDOW):
            _review_window(todo[i : i + WINDOW], ctx, stats)
            if progress:
                progress()
    except FatalReviewError as exc:
        stats["error"] = str(exc)
        for c in todo:
            if "review" not in c and "review_error" not in c:
                c["review_error"] = str(exc)
    return _finish(jdir, chunks, stats)


def _finish(jdir, chunks, stats):
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    if stats["dropped"]:
        (stats["raw_dir"] / "dropped.json").write_text(json.dumps(stats["dropped"], indent=2, ensure_ascii=False))
    reviewed = [c["review"] for c in chunks if c.get("review")]
    return {
        "model": stats["model"],
        "requests": stats["requests"],
        "flagged": sum(1 for r in reviewed if r["issues"]),
        "changed": sum(1 for r in reviewed if r["changed"]),
        "failed": sum(1 for c in chunks if c.get("review_error")),
        "error": stats["error"],
    }


def _review_window(window, ctx, stats):
    """Check one window in a request; on WindowError split it in half and check each half."""
    try:
        edits = _parse(_request(_messages(window, ctx), ctx["key"], stats))
    except FatalReviewError:
        raise
    except WindowError as exc:
        if len(window) == 1:
            window[0]["review_error"] = str(exc)
            return
        mid = len(window) // 2
        _review_window(window[:mid], ctx, stats)
        _review_window(window[mid:], ctx, stats)
        return
    except Exception as exc:  # retries exhausted or a request error splitting won't fix
        for c in window:
            c["review_error"] = str(exc)
        return
    _apply(window, edits, stats)


def _messages(window, ctx):
    """The prompt and the window's segments, with CONTEXT read-only neighbours on each side."""
    order, pos = ctx["order"], ctx["pos"]
    first, last = pos[window[0]["id"]], pos[window[-1]["id"]]
    editable = {c["id"] for c in window}
    segments = []
    for c in order[max(first - CONTEXT, 0) : last + CONTEXT + 1]:
        segments.append(
            {
                "id": c["id"],
                "speaker": c["speaker"],
                "gender": ctx["genders"].get(c["speaker"], "unknown"),
                "slot_s": round(c["end"] - c["start"], 2),
                "source": c.get("text") or "",
                "translation": c["translation"],
                "chars": len(c["translation"]),
                "max_chars": max_chars(c, ctx["lang"]),
                "editable": c["id"] in editable,
                "slow": bool(c.get("slow")),
            }
        )
    return [
        {"role": "system", "content": render_prompt(ctx["prompt"], ctx["source"], ctx["target"])},
        {"role": "user", "content": json.dumps(segments, ensure_ascii=False, indent=1)},
    ]


def _parse(content):
    """The reply's edits list. Raises WindowError if it isn't the JSON asked for."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    try:
        data = json.loads(text)
    except ValueError:
        a, b = text.find("{"), text.rfind("}")  # some providers ignore response_format and add prose
        try:
            data = json.loads(text[a : b + 1]) if a != -1 and b > a else None
        except ValueError:
            data = None
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
        raise WindowError(f"reply is not the expected JSON: {content[:200]}")
    return data["edits"]


def _apply(window, edits, stats):
    """Apply a window's valid edits; every chunk in the window gets its review, edited or not."""
    by_id = {c["id"]: c for c in window}
    done = set()
    for e in edits:
        c = by_id.get(e.get("id")) if isinstance(e, dict) else None
        issues = [i for i in (e.get("issues") or []) if i in ISSUES] if c else []
        text = " ".join(str(e.get("translation") or "").split()) if c else ""
        why = None
        if c is None:
            why = "id not in the window's editable segments"
        elif c["id"] in done:
            why = "segment edited twice"
        elif not issues:
            why = "no known issue given"
        elif not text:
            why = "empty translation"
        if why:
            stats["dropped"].append({"edit": e, "why": why})
            continue
        done.add(c["id"])
        old = c["translation"]
        review = {"issues": issues, "reason": str(e.get("reason") or "").strip() or None, "model": MODEL}
        if "too_long" in issues and len(text) > len(old):
            # flagged, but the rewrite is longer: keep the translation, note why
            stats["dropped"].append({"edit": e, "why": "too_long rewrite is longer than the original"})
            c["review"] = {**review, "changed": False, "reviewed": None, "note": "rewrite was longer; not applied"}
            continue
        changed = text != old
        c["review"] = {**review, "changed": changed, "reviewed": text if changed else None}
        if changed:
            c.setdefault("translation_original", old)
            set_translation(c, text)
    for c in window:
        if c["id"] not in done:
            c["review"] = {"issues": [], "reason": None, "model": MODEL, "changed": False, "reviewed": None}


def _save_raw(stats, payload, res=None, error=None):
    """Write one attempt (request body + raw response) to review_raw/<attempt>.json. No API key is stored."""
    stats["attempts"] += 1
    record = {"attempt": stats["attempts"], "time": datetime.now(timezone.utc).isoformat(), "url": API_URL, "request": payload}
    if res is not None:
        record["status"] = res.status_code
        try:
            record["body"] = res.json()
        except ValueError:
            record["body"] = res.text
    else:
        record["error"] = error
    (stats["raw_dir"] / f"{stats['attempts']:04d}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))


def _request(messages, key, stats) -> str:
    """POST one check. Returns the reply's text."""
    error = "unknown error"
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": {"name": "review", "strict": True, "schema": SCHEMA}},
    }
    for attempt in range(RETRIES + 1):
        wait = min(BACKOFF * 2**attempt, MAX_WAIT)
        try:
            res = requests.post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=TIMEOUT)
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
            _save_raw(stats, payload, error=error)
        else:
            _save_raw(stats, payload, res=res)
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
                    raise WindowError("reply was cut off at the output limit")
                return (choice.get("message") or {}).get("content") or ""

            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            error = f"OpenRouter {res.status_code}: {err.get('message') or res.text[:300]}"
            if res.status_code in (401, 402, 403):
                raise FatalReviewError(error)
            if res.status_code == 413:
                raise WindowError(error)
            if res.status_code != 429 and res.status_code < 500:
                raise RuntimeError(error)
            try:  # honour the server's Retry-After (seconds) when it sends one, up to RETRY_AFTER_MAX
                wait = max(min(float(res.headers.get("Retry-After", "")), RETRY_AFTER_MAX), wait)
            except ValueError:
                pass
        if attempt < RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)
