import io
import json
import time
import wave
from itertools import zip_longest
from pathlib import Path

import numpy as np
import requests

try:
    from . import gender
    from .asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX, api_key
except ImportError:
    import gender
    from asr import BACKOFF, MAX_WAIT, RETRIES, RETRY_AFTER_MAX, api_key

API_URL = "https://api.bodhan.ai/v1/audio/speech"
MODEL = "indic-speak"
TIMEOUT = 120  # seconds per request

# The API reads text over this many characters badly, so longer text is split and its clips joined
MAX_TTS_CHARS = 200
MIN_SPLIT_CHARS = 40  # minimum left-piece length when searching backward for punctuation
SENTENCE_PAUSE_S = 0.6  # silence inserted at sentence-ending punctuation splits
CLAUSE_PAUSE_S = 0.3  # silence inserted at clause punctuation splits
WORD_PAUSE_S = 0.15  # silence inserted at splits between words (every join gets at least this much)
SENTENCE_END = set("।॥.?!۔؟᱾᱿꯫…")  # danda, double danda, Latin, Urdu, Ol Chiki, Meitei Mayek
CLAUSE = set(",;:،؛—–")
CLOSERS = set("\"'”’)]")  # quotes / brackets right after a mark stay with the left piece
# Joining pieces: edge silence is always trimmed (as dubbing's cadence trim); then a fixed pause is put between
# them, by the kind of split
SILENCE_DB = -35.0  # frames this far below the clip's 95th-percentile RMS are silent
SILENCE_FRAME = 0.01  # seconds per RMS frame
EDGE_PAD_S = 0.02  # silence kept at each trimmed edge

FALLBACK_LANG = "hi"  # voice pool for target languages without their own voices (English)

# Voices by the language they were recorded in: (female, male). Any voice can read any language.
VOICES = {
    "as": (["Prastuti"], ["Ankur"]),
    "bn": (["Ishita"], ["Sourav"]),
    "brx": (["Gwrbw"], ["Sansuma"]),
    "doi": (["Preeti"], ["Sham"]),
    "gu": (["Dhara"], ["Parth"]),
    "hi": (["Kavya", "Suhani"], ["Amit"]),
    "kn": (["Deepika"], ["Adarsh"]),
    "kok": (["Anjali"], ["Sandeep"]),
    "ks": (["Zoon"], ["Ishfaq"]),
    "mai": (["Vaidehi"], ["Madhukar"]),
    "ml": (["Lakshmi"], ["Kiran"]),
    "mni": (["Thoibi"], ["Chaoba"]),
    "mr": (["Anagha"], ["Chinmay"]),
    "ne": (["Srijana"], ["Sagar"]),
    "or": (["Itishree"], ["Akash"]),
    "pa": (["Kaur"], ["Manpreet"]),
    "sa": (["Bharati"], ["Aryaman"]),
    "sat": (["Phulmani"], ["Sibu"]),
    "sd": (["Moomal"], ["Rano"]),
    "ta": (["Anitha"], ["Arun"]),
    "te": (["Sravani"], ["Vamsi"]),
    "ur": (["Saba"], ["Zaid"]),
}

# The voices offered for each target language, best first: (female, male), as ranked in voices.md (by ear for Hindi:
# Suhani and Amit, then Gujarati, Urdu, Punjabi, Maithili, Dogri, Sindhi; Kavya is left out). Assamese, Bengali and
# Odia voices stand in for each other, as do the Dravidian ones; the Hindi ranks are everyone's fallback. Past these,
# every other voice of the gender follows in REST_ORDER
PREFERENCE = {
    "hi": (["Suhani", "Dhara", "Saba", "Vaidehi", "Preeti"], ["Amit", "Parth", "Manpreet", "Sham", "Rano"]),
    "en": (["Suhani", "Dhara", "Saba", "Vaidehi", "Preeti"], ["Amit", "Parth", "Manpreet", "Sham", "Rano"]),
    "gu": (["Suhani", "Dhara", "Saba", "Vaidehi", "Preeti"], ["Amit", "Parth", "Manpreet", "Sham", "Rano"]),
    "ur": (["Saba", "Suhani", "Dhara", "Vaidehi", "Preeti"], ["Zaid", "Amit", "Parth", "Manpreet", "Sham"]),
    "pa": (["Kaur", "Suhani", "Dhara", "Saba", "Vaidehi"], ["Manpreet", "Amit", "Parth", "Sham", "Rano"]),
    "mai": (["Vaidehi", "Suhani", "Dhara", "Saba", "Preeti"], ["Madhukar", "Amit", "Parth", "Manpreet", "Sham"]),
    "doi": (["Preeti", "Suhani", "Dhara", "Saba", "Vaidehi"], ["Sham", "Amit", "Parth", "Manpreet", "Rano"]),
    "sd": (["Moomal", "Saba", "Suhani", "Dhara", "Vaidehi"], ["Rano", "Amit", "Parth", "Manpreet", "Sham"]),
    "ks": (["Zoon", "Saba", "Suhani", "Dhara", "Vaidehi"], ["Ishfaq", "Manpreet", "Amit", "Parth", "Rano"]),
    "mr": (["Anagha", "Suhani", "Dhara", "Saba", "Vaidehi"], ["Chinmay", "Amit", "Parth", "Manpreet", "Sham"]),
    "kok": (["Anjali", "Anagha", "Suhani", "Dhara", "Saba"], ["Sandeep", "Chinmay", "Amit", "Parth", "Manpreet"]),
    "ne": (["Srijana", "Vaidehi", "Suhani", "Dhara", "Saba"], ["Sagar", "Madhukar", "Amit", "Parth", "Manpreet"]),
    "sa": (["Bharati", "Suhani", "Dhara", "Saba", "Vaidehi"], ["Aryaman", "Amit", "Parth", "Manpreet", "Sham"]),
    "as": (["Prastuti", "Ishita", "Itishree", "Suhani", "Dhara"], ["Ankur", "Sourav", "Akash", "Amit", "Parth"]),
    "bn": (["Ishita", "Prastuti", "Itishree", "Suhani", "Dhara"], ["Sourav", "Ankur", "Akash", "Amit", "Parth"]),
    "or": (["Itishree", "Ishita", "Prastuti", "Suhani", "Dhara"], ["Akash", "Sourav", "Ankur", "Amit", "Parth"]),
    "mni": (["Thoibi", "Prastuti", "Ishita", "Itishree", "Suhani"], ["Chaoba", "Ankur", "Sourav", "Akash", "Amit"]),
    "brx": (["Gwrbw", "Prastuti", "Ishita", "Itishree", "Suhani"], ["Sansuma", "Ankur", "Sourav", "Akash", "Amit"]),
    "sat": (["Phulmani", "Itishree", "Ishita", "Prastuti", "Suhani"], ["Sibu", "Akash", "Sourav", "Ankur", "Amit"]),
    "ta": (["Anitha", "Lakshmi", "Sravani", "Deepika", "Suhani"], ["Arun", "Kiran", "Vamsi", "Adarsh", "Amit"]),
    "ml": (["Lakshmi", "Anitha", "Deepika", "Sravani", "Suhani"], ["Kiran", "Arun", "Adarsh", "Vamsi", "Amit"]),
    "te": (["Sravani", "Deepika", "Anitha", "Lakshmi", "Suhani"], ["Vamsi", "Adarsh", "Arun", "Kiran", "Amit"]),
    "kn": (["Deepika", "Sravani", "Lakshmi", "Anitha", "Suhani"], ["Adarsh", "Vamsi", "Kiran", "Arun", "Amit"]),
}

# The order of the voices past a language's PREFERENCE: the Hindi ranks, the other north and west Indo-Aryan voices,
# then the east, north-east and Dravidian ones; Kavya (bad on Hindi) last
REST_ORDER = [
    "Suhani", "Dhara", "Saba", "Vaidehi", "Preeti", "Amit", "Parth", "Manpreet", "Sham", "Rano",
    "Zaid", "Kaur", "Madhukar", "Moomal", "Zoon", "Ishfaq", "Anagha", "Chinmay", "Anjali", "Sandeep",
    "Srijana", "Sagar", "Bharati", "Aryaman",
    "Ishita", "Sourav", "Prastuti", "Ankur", "Itishree", "Akash",
    "Thoibi", "Chaoba", "Gwrbw", "Sansuma", "Phulmani", "Sibu",
    "Anitha", "Arun", "Lakshmi", "Kiran", "Sravani", "Vamsi", "Deepika", "Adarsh",
    "Kavya",
]
# Voices offered per gender for a target language (the best of `preferred`): more for the main languages
MAIN_LANGS = {"hi", "ta", "te", "ml", "bn", "mr"}
SHOWN_MAIN = 8
SHOWN = 4

# Speaking styles the API accepts in `instructions` (console.bodhan.ai/api-docs); none sent = neutral reading
STYLES = [
    "news",
    "AIR style news",
    "TV style news",
    "educational lecture",
    "single person narration audiobook",
    "children's stories",
    "advertisements",
    "Customer Care",
    "happy",
    "sad",
    "anger",
    "fear",
    "surprise",
    "disgust",
]


class FatalTTSError(RuntimeError):
    """An error every request would hit (bad key, no credit), so the whole job stops."""


ALL_VOICES = [v for f, m in VOICES.values() for v in f + m]


def preferred(lang) -> tuple[list[str], list[str]]:
    """(female, male): the voices offered for target `lang`, best first (its PREFERENCE, then REST_ORDER), the
    first SHOWN_MAIN of each gender for MAIN_LANGS and SHOWN for the others."""
    top = PREFERENCE.get(lang) or PREFERENCE[FALLBACK_LANG]
    n = SHOWN_MAIN if lang in MAIN_LANGS else SHOWN
    out = []
    for k, ranked in enumerate(top):
        voices = {v for pair in VOICES.values() for v in pair[k]}
        out.append((ranked + [v for v in REST_ORDER if v in voices and v not in ranked])[:n])
    return out[0], out[1]


def options() -> dict:
    """Voices per language, for the UI: each target language's preferred voices, and every voice by the language
    it was recorded in."""
    return {
        "preferred": {lang: dict(zip(("female", "male"), preferred(lang))) for lang in PREFERENCE},
        "voices": {lang: {"female": f, "male": m} for lang, (f, m) in VOICES.items()},
        "fallback_lang": FALLBACK_LANG,
        "styles": STYLES,
    }


def by_talk_time(chunks) -> list:
    """The speakers of `chunks`, the one who speaks longest first (ties by speaker number): the order voices are
    handed out in, so the main speakers get the best ones. A merged chunk counts its parts' speech, not the pauses
    between them."""
    talk: dict = {}
    for c in chunks:
        for part in c.get("sub_chunks") or [c]:
            talk[c["speaker"]] = talk.get(c["speaker"], 0.0) + part["end"] - part["start"]
    return sorted(talk, key=lambda s: (-talk[s], s))


def assign_voices(speakers, lang, chosen=None, genders=None):
    """A voice per speaker, handed out in the order of `speakers` (by_talk_time: the longest speaker first). Speakers in `chosen` ({speaker: voice}) keep that voice. Speakers of known gender
    in `genders` ({speaker: "female" | "male"}) get the best unused voice of that gender in `preferred(lang)`
    (reused in that order once all of them are taken). The others get the best unused voices,
    female and male alternating."""
    chosen = {s: v for s, v in (chosen or {}).items() if s in speakers and v}
    genders = genders or {}
    female, male = preferred(lang)
    out, taken = dict(chosen), set(chosen.values())
    reused = {"female": 0, "male": 0}
    for s in speakers:
        g = genders.get(s)
        if s in chosen or g not in ("female", "male"):
            continue
        pool = female if g == "female" else male
        out[s] = next((v for v in pool if v not in taken), None)
        if out[s] is None:
            out[s] = pool[reused[g] % len(pool)]
            reused[g] += 1
        taken.add(out[s])
    pool = [v for pair in zip_longest(female, male) for v in pair if v]
    free = [v for v in pool if v not in taken] or pool
    i = 0
    for s in speakers:
        if s not in out:
            out[s] = free[i % len(free)]
            i += 1
    return {s: out[s] for s in speakers}


def suggest_voices(jdir: Path, speakers, lang, chosen=None) -> dict:
    """What the UI shows per speaker, as {"1": voice}: `assign_voices` using the run's detected genders, except
    that a speaker whose gender couldn't be told (and who has no chosen voice) gets None: the user picks."""
    details = gender.load_details(jdir)
    voices = assign_voices(speakers, lang, chosen, gender.load(jdir))
    return {
        str(s): None if details.get(s, {}).get("gender") == "unknown" and s not in (chosen or {}) else v
        for s, v in voices.items()
    }


def synthesize_chunks(
    jdir: Path,
    chunks: list[dict],
    lang: str,
    voices: dict | None = None,
    progress=None,
    style: str | None = None,
    reuse: bool = False,
    only: set[int] | None = None,
) -> dict:
    """Read each chunk's translation aloud into tts/<id>.wav. `voices` maps speaker -> voice (speakers left
    out are assigned automatically); `style` is one of STYLES, or None for the neutral reading. With `reuse`,
    clips already made from the same text with the same voice and style are kept (a clip from before tts_text
    was stored counts as made from the current text), so a retry only requests the failed or missing
    ones. `only`: chunk ids, to read just those (an edited segment); the others are left as they are.
    `progress()` is called once after the reset and then after each chunk with a translation. chunks.json
    is written even when a fatal error stops the run. Returns stats."""
    key = api_key("tts")
    out_dir = jdir / "tts"
    out_dir.mkdir(exist_ok=True)

    voices = assign_voices(by_talk_time(chunks), lang, voices, gender.load(jdir))
    instructions = json.dumps({"lang": lang, **({"style": style} if style else {})})

    for c in chunks:
        if only is not None and c["id"] not in only:
            continue
        # a job can be re-voiced; drop the previous synthesis results first, unless reusable
        if (
            reuse
            and c.get("tts_file")
            and (jdir / c["tts_file"]).exists()
            and c.get("tts_voice") == voices[c["speaker"]]
            and "tts_style" in c
            and c["tts_style"] == style
            and c.get("tts_text", c.get("translation")) == c.get("translation")
        ):
            continue
        for k in ("tts_error", "tts_voice", "tts_duration", "tts_style", "tts_text"):
            c.pop(k, None)
        c["tts_file"] = None
    if progress:
        progress()

    try:
        for c in chunks:
            text = c.get("translation")
            if not text or c.get("tts_file") or (only is not None and c["id"] not in only):
                continue
            voice = voices[c["speaker"]]
            try:
                audio = _synthesize(text, voice, instructions, key)
            except FatalTTSError:
                raise
            except Exception as exc:
                c["tts_error"] = str(exc)
            else:
                path = out_dir / f"{c['id']:04d}.wav"
                path.write_bytes(audio)
                c["tts_file"] = f"tts/{path.name}"
                c["tts_voice"] = voice
                c["tts_style"] = style
                c["tts_text"] = text
                c["tts_duration"] = _duration(path)
            if progress:
                progress()
    finally:
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return {
        "failed": sum(1 for c in chunks if c.get("tts_error")),
        "voices": {str(s): v for s, v in voices.items()},
    }


def _duration(path: Path):
    try:
        with wave.open(str(path), "rb") as w:
            return round(w.getnframes() / w.getframerate(), 3)
    except (wave.Error, EOFError):
        return None


def _synthesize(text, voice, instructions, key) -> bytes:
    """WAV bytes for `text`. Text over MAX_TTS_CHARS is split (`_split`), each piece requested on its own and
    the clips joined back into one (`_join`). Any piece failing fails the whole text."""
    pieces = _split(text.strip())
    if len(pieces) == 1:
        return _request(text, voice, instructions, key)
    clips = [_request(p, voice, instructions, key) for p, _ in pieces]
    return _join(clips, [kind for _, kind in pieces[:-1]])


def _split(text) -> list[tuple[str, str]]:
    """Greedy pieces of at most MAX_TTS_CHARS: returns [(piece, split_kind), ...], where split_kind is how the
    split after the piece was made: 'sentence', 'clause' or 'word' (the last piece's is 'none')."""
    pieces = []
    text = text.strip()
    while len(text) > MAX_TTS_CHARS:
        cut, kind = _split_point(text)
        pieces.append((text[:cut].strip(), kind))
        text = text[cut:].strip()
    if text:
        pieces.append((text, "none"))
    return pieces


def _split_point(text) -> tuple[int, str]:
    """Where to cut the start of `text` (longer than MAX_TTS_CHARS) and the kind of cut: after the sentence-ending
    mark closest to MAX_TTS_CHARS, else after the closest clause mark (both at least MIN_SPLIT_CHARS in, closing
    quotes / brackets kept on the left); else at the last space that fits; else (no space) at MAX_TTS_CHARS."""
    for kind, marks in (("sentence", SENTENCE_END), ("clause", CLAUSE)):
        cuts = []
        for i in range(MIN_SPLIT_CHARS, MAX_TTS_CHARS):
            if text[i] in marks and not _is_inner_dot(text, i):
                c = i + 1
                while c < len(text) and text[c] in CLOSERS and c < MAX_TTS_CHARS:
                    c += 1
                if c <= MAX_TTS_CHARS and text[:c].strip() and text[c:].strip():
                    cuts.append(c)
        if cuts:
            return max(cuts), kind
    for i in range(MAX_TTS_CHARS, 0, -1):
        if text[i].isspace() and text[:i].strip() and text[i:].strip():
            return i, "word"
    return MAX_TTS_CHARS, "word"


def _is_inner_dot(text, i) -> bool:
    """A "." that doesn't end a sentence: a decimal point (3.5), or after a lone Latin letter (an initial)."""
    if text[i] != ".":
        return False
    if 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit():
        return True
    return i > 0 and text[i - 1].isascii() and text[i - 1].isalpha() and (i == 1 or not text[i - 2].isalpha())


def _join(clips, kinds) -> bytes:
    """One WAV from the pieces' clips. At every boundary (`kinds[k]`, between clip k and k+1) the trailing silence
    of the left clip and the leading silence of the right one are trimmed to EDGE_PAD_S. Then silence is put
    between the clips: SENTENCE_PAUSE_S after a sentence split, CLAUSE_PAUSE_S after a clause split and
    WORD_PAUSE_S after a split between words, so no two clips run into each other."""
    arrays, params = [], None
    for wav in clips:
        with wave.open(io.BytesIO(wav), "rb") as w:
            p = (w.getnchannels(), w.getsampwidth(), w.getframerate())
            frames = w.readframes(w.getnframes())
        if p[1] != 2:
            raise RuntimeError(f"Bodhan TTS returned {8 * p[1]}-bit audio, expected 16-bit")
        if params and p != params:
            raise RuntimeError(f"Bodhan TTS pieces differ in format: {params} vs {p}")
        params = p
        arrays.append(np.frombuffer(frames, dtype=np.int16).reshape(-1, p[0]).astype(np.float32))
    channels, _, sr = params
    for k in range(len(arrays) - 1):
        arrays[k] = _trim_silence(arrays[k], sr, end=True)
        arrays[k + 1] = _trim_silence(arrays[k + 1], sr, start=True)
    pauses = {"sentence": SENTENCE_PAUSE_S, "clause": CLAUSE_PAUSE_S}
    out = arrays[0]
    for k, x in enumerate(arrays[1:]):
        gap = np.zeros((int(round(sr * pauses.get(kinds[k], WORD_PAUSE_S))), channels), dtype=np.float32)
        out = np.concatenate([out, gap, x])
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(np.round(out), -32768, 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _trim_silence(x, sr, start=False, end=False):
    """`x` (samples × channels) with its leading and/or trailing silence cut to EDGE_PAD_S, at whole
    SILENCE_FRAME frames. Silent: frame RMS below the 95th-percentile frame's plus SILENCE_DB. A clip with no
    speech found is returned as it is."""
    hop = int(round(sr * SILENCE_FRAME))
    n = len(x) // hop
    if not n:
        return x
    rms = np.sqrt((x[: n * hop] ** 2).reshape(n, -1).mean(axis=1))
    top = float(np.percentile(rms, 95))
    idx = np.flatnonzero(rms >= top * 10 ** (SILENCE_DB / 20)) if top > 0 else []
    if not len(idx):
        return x
    pad = int(round(EDGE_PAD_S / SILENCE_FRAME))
    a = max(idx[0] - pad, 0) * hop if start else 0
    b = min(idx[-1] + 1 + pad, n) * hop if end else len(x)
    if end and b == n * hop:  # speech runs to the last whole frame: keep the partial frame after it too
        b = len(x)
    return x[a:b]


def _request(text, voice, instructions, key) -> bytes:
    """POST one TTS request. Returns the WAV bytes."""
    error = "unknown error"
    payload = {"model": MODEL, "input": text, "voice": voice, "instructions": instructions}
    for attempt in range(RETRIES + 1):
        wait = min(BACKOFF * 2**attempt, MAX_WAIT)
        try:
            res = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=TIMEOUT,
            )
        except requests.RequestException as exc:
            error = f"request failed: {exc}"
        else:
            if res.status_code == 200 and res.content[:4] == b"RIFF":
                return res.content
            error = f"Bodhan TTS {res.status_code}: {res.text[:300]}"
            if res.status_code == 200:
                raise RuntimeError(f"Bodhan TTS returned a non-WAV body: {res.text[:300]}")
            if res.status_code in (401, 402, 403):
                raise FatalTTSError(error)
            if res.status_code != 429 and res.status_code < 500:
                raise RuntimeError(error)
            try:  # honour the server's Retry-After (seconds) when it sends one, up to RETRY_AFTER_MAX
                wait = max(min(float(res.headers.get("Retry-After", "")), RETRY_AFTER_MAX), wait)
            except ValueError:
                pass
        if attempt < RETRIES:
            time.sleep(wait)
    raise RuntimeError(error)
