"""Voice samples: one sentence per language, written by Gemma (SAMPLE_MODEL), read aloud by every voice, so the
studio can play how a voice sounds in the target language before dubbing. Texts are kept in
storage/voice_samples/texts.json and clips in storage/voice_samples/<lang>/<voice>.wav. A missing text or clip is
made the first time it's asked for; `python -m backend.samples` makes them all ahead of time (each language with
the voices it offers)."""

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

try:
    from . import asr, review, translate, tts
except ImportError:
    import asr
    import review
    import translate
    import tts

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage" / "voice_samples"
TEXTS_FILE = STORAGE_DIR / "texts.json"
BUILD_WORKERS = 4  # TTS requests at a time when making the whole matrix

SAMPLE_CHARS = 100  # about this long
SAMPLE_MODEL = "google/gemma-4-26b-a4b-it"  # Gemma 4 MoE (26B, 4B active)
SAMPLE_TIMEOUT = 120
SAMPLE_PROMPT = """Write a short sample text in {language}, written in {script}, for listening to a text-to-speech \
voice. About {chars} characters long (between {low} and {high}). One or two natural, everyday sentences, the way \
a native speaker would say them: varied sounds, a number or a name is welcome, and end with a question or an \
exclamation if it fits. No English words, no transliteration, no quotes around it.
Reply with JSON only: {{"text": "..."}}"""

# Script names for the prompt, by the first entry of asr.SCRIPTS (Unicode names)
SCRIPT_NAMES = {
    "ARABIC": "Perso-Arabic script",
    "BENGALI": "Bengali-Assamese script",
    "DEVANAGARI": "Devanagari",
    "GUJARATI": "Gujarati script",
    "GURMUKHI": "Gurmukhi",
    "KANNADA": "Kannada script",
    "MALAYALAM": "Malayalam script",
    "OL": "Ol Chiki",
    "ORIYA": "Odia script",
    "TAMIL": "Tamil script",
    "TELUGU": "Telugu script",
}

_texts_lock = threading.Lock()
_clip_locks: dict[tuple[str, str], threading.Lock] = {}  # one request per clip, however many ask at once


def load_texts() -> dict:
    try:
        return json.loads(TEXTS_FILE.read_text())
    except (OSError, ValueError):
        return {}


def text(lang: str) -> str:
    """The sample sentence in `lang`, written by Gemma the first time."""
    with _texts_lock:
        texts = load_texts()
        if not texts.get(lang):
            texts[lang] = generate_text(lang)
            STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            TEXTS_FILE.write_text(json.dumps(texts, indent=2, ensure_ascii=False))
        return texts[lang]


def clip(lang: str, voice: str) -> Path:
    """`voice` reading the `lang` sample sentence (made the first time). Raises ValueError for an unknown
    language or voice."""
    if lang not in translate.LANGUAGES:
        raise ValueError(f"unknown language {lang!r}")
    if voice not in tts.ALL_VOICES:
        raise ValueError(f"unknown voice {voice!r}")
    path = STORAGE_DIR / lang / f"{voice}.wav"
    with _clip_locks.setdefault((lang, voice), threading.Lock()):
        if not path.exists():
            audio = tts._synthesize(text(lang), voice, json.dumps({"lang": lang}), asr.api_key("tts"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(audio)
    return path


def generate_text(lang: str) -> str:
    """A new sample text in `lang`: one OpenRouter request to SAMPLE_MODEL, with the studio's retries."""
    scripts = asr.SCRIPTS.get(lang) or ()
    script = SCRIPT_NAMES.get(scripts[0], scripts[0].title()) if scripts else "Latin script"
    prompt = SAMPLE_PROMPT.format(
        language=translate.LANGUAGES[lang], script=script, chars=SAMPLE_CHARS,
        low=int(SAMPLE_CHARS * 0.8), high=int(SAMPLE_CHARS * 1.2),
    )
    payload = {
        "model": SAMPLE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.9,
        "max_tokens": 1000,
        "reasoning": {"enabled": False},
        "response_format": {"type": "json_object"},
    }
    key = review.api_key()
    error = "unknown error"
    for attempt in range(asr.RETRIES + 1):
        wait = min(asr.BACKOFF * 2**attempt, asr.MAX_WAIT)
        try:
            res = requests.post(asr.OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=SAMPLE_TIMEOUT)
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
        else:
            if res.status_code == 200:
                content = (((res.json().get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content)
                try:
                    out = str(json.loads(content[content.find("{") : content.rfind("}") + 1]).get("text") or "").strip()
                except (ValueError, AttributeError):
                    out = ""
                if out:
                    return out
                error = f"reply has no text: {content[:200]}"
            else:
                error = f"OpenRouter {res.status_code}: {res.text[:300]}"
                if res.status_code != 429 and res.status_code < 500:
                    raise RuntimeError(error)
        if attempt < asr.RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)


def build(langs=None, voices=None):
    """Make every missing text and clip (all languages, each with the voices it offers, tts.preferred, by default),
    printing progress."""
    langs = langs or list(translate.LANGUAGES)
    for lang in langs:
        print(f"{lang}: {text(lang)}", flush=True)
    todo = [
        (l, v) for l in langs for v in (voices or [v for g in tts.preferred(l) for v in g])
        if not (STORAGE_DIR / l / f"{v}.wav").exists()
    ]
    print(f"{len(todo)} clips to make", flush=True)

    def one(job):
        try:
            clip(*job)
            return None
        except Exception as exc:
            return f"{job[0]}/{job[1]}: {exc}"

    failed = 0
    with ThreadPoolExecutor(BUILD_WORKERS) as pool:
        for i, err in enumerate(pool.map(one, todo), 1):
            if err:
                failed += 1
                print("FAILED", err, flush=True)
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {failed} failed", flush=True)


if __name__ == "__main__":
    import sys

    build(sys.argv[1:] or None)
