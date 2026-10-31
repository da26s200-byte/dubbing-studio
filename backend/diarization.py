import json
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch

MODEL_NAME = "nvidia/Nemotron-3-Diarization"
THRESHOLD = 0.5  # speaker probability above which a 10 ms frame counts as that speaker talking
FRAME_S = 0.02

# Cleanup / chunking thresholds (seconds)
MIN_SEG = 0.3  # drop diarization segments shorter than this
INTERJECTION = 0.5  # shorter segment sandwiched by one other speaker counts as a pause
PAD = 0.15  # padding added around each chunk for the ASR cut
MAX_AUDIO = 15.0  # longest audio sent in one ASR request, padding included
MAX_CHUNK = MAX_AUDIO - 2 * PAD  # runs longer than this are split; no chunk's speech span exceeds it
MIN_PAUSE = 0.27  # silent stretches at least this long are recorded as pauses, and are the cut options
MIN_VOICED = 0.2  # a turn with fewer of its frames above the silence threshold is dropped (stem bleed)
SILENCE_DB = -30.0  # frames this far below the file's 95th-percentile level are silence
SUSTAIN = 0.2  # window for the lowest-sustained-energy fallback
FALLBACK_WINDOW = 6.0  # a long piece is cut in its middle this-many seconds: at the longest pause, else the quietest point
# Face-based fixes (lipsync/faces.py): False = dry run, they are only recorded in diarization.json
ASD_APPLY = True

_model = None


def get_model():
    """(processor, model); loaded on first use and reused. Needs a `transformers` with Nemotron-3-Diarization;
    the weights come from the Hugging Face cache (HF_TOKEN is picked up from the environment if the repo needs it).
    bf16 on GPU, float32 on CPU."""
    global _model
    if _model is None:
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor

        cuda = torch.cuda.is_available()
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        model = AutoModelForAudioFrameClassification.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16 if cuda else torch.float32
        )
        _model = (processor, model.to(torch.device("cuda" if cuda else "cpu")).eval())
    return _model


def _run_model(data, sr):
    """Raw (speaker, start, end) segments; speakers are 0..7 in order of arrival, overlapping speech overlaps."""
    processor, model = get_model()
    inputs = processor(data, sampling_rate=sr).to(model.device, dtype=model.dtype)
    with torch.inference_mode():  # offline mode: the model cuts the file into its own streaming chunks
        logits = model(**inputs).logits
    segments = processor.extract_speaker_dict(logits.float(), inputs["attention_mask"], threshold=THRESHOLD)[0]
    return [(seg["Speaker"], seg["Start"], seg["End"]) for seg in segments]


def diarize_vocals(jdir: Path, vocals_path: Path, ffmpeg_bin: str):
    """Downsample vocals to 16 kHz mono, diarize, clean up. Returns [(speaker, start, end)]."""
    wav16 = jdir / "vocals_16k.wav"
    subprocess.run(
        [ffmpeg_bin, "-y", "-i", str(vocals_path), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav16)],
        check=True,
        capture_output=True,
    )

    data, sr = _read_wav(wav16)
    raw = _run_model(data, sr)

    # the model sees normalised features, so faint music bleed in the vocals stem can pass as speech
    flat = _flatten(raw)
    voiced = _drop_silent(flat, _load_envelope(wav16)[0])
    turns = _relabel(_clean(voiced))

    payload = {
        "num_speakers": len({t[0] for t in turns}),
        "dropped_silent": len(flat) - len(voiced),
        "turns": [{"speaker": s, "start": round(a, 3), "end": round(b, 3)} for s, a, b in turns],
    }
    try:
        from . import gender
    except ImportError:
        import gender
    try:  # only the voice suggestion depends on it, so a missing model must not stop the run
        payload["genders"] = {str(s): g for s, g in gender.classify_speakers(data, sr, turns).items()}
    except Exception as exc:
        payload["genders"] = {}
        payload["gender_error"] = str(exc)
    (jdir / "diarization.json").write_text(json.dumps(payload, indent=2))
    return turns


def apply_faces(jdir: Path, turns, error: str | None = None):
    """Record the face-based fixes of faces.json (lipsync/faces.py) in diarization.json and, if ASD_APPLY,
    use them: the turns become faces.json's, the audio-only ones are kept as `turns_raw`, genders are redone.
    `error`: the faces step failed, so the audio-only turns stay. Returns [(speaker, start, end)]."""
    path = jdir / "diarization.json"
    payload = json.loads(path.read_text())
    faces = None if error else json.loads((jdir / "faces.json").read_text())
    payload["asd"] = {
        "applied": bool(faces and ASD_APPLY and faces["fixes"]),
        "error": error,
        "fixes": faces["fixes"] if faces else [],
        "speaker_map": faces["speaker_map"] if faces else None,
    }
    if payload["asd"]["applied"]:
        payload["turns_raw"] = payload["turns"]
        payload["turns"] = faces["turns"]
        turns = [(t["speaker"], t["start"], t["end"]) for t in faces["turns"]]
        payload["num_speakers"] = len({t[0] for t in turns})
        try:
            from . import gender
        except ImportError:
            import gender
        if "gender_error" not in payload:
            try:
                data, sr = _read_wav(jdir / "vocals_16k.wav")
                payload["genders"] = {str(s): g for s, g in gender.classify_speakers(data, sr, turns).items()}
            except Exception as exc:
                payload["genders"] = {}
                payload["gender_error"] = str(exc)
    path.write_text(json.dumps(payload, indent=2))
    return turns


def _flatten(raw):
    """One label per moment: overlaps are cut at their midpoint, nested segments are dropped."""
    out = []
    for label, start, end in sorted(raw, key=lambda s: (s[1], s[2])):
        if out and start < out[-1][2]:
            if end <= out[-1][2]:
                continue
            mid = min(max((start + out[-1][2]) / 2, out[-1][1]), out[-1][2])
            out[-1] = (out[-1][0], out[-1][1], mid)
            start = mid
        out.append((label, start, end))
    return out


def _drop_silent(segs, env):
    """Drops segments where under MIN_VOICED of the frames are above the silence threshold."""
    loud = env >= _silence_threshold(env)
    keep = []
    for label, start, end in segs:
        frames = loud[int(start / FRAME_S) : max(int(end / FRAME_S), int(start / FRAME_S) + 1)]
        if len(frames) and frames.mean() >= MIN_VOICED:
            keep.append((label, start, end))
    return keep


def _clean(segs):
    segs = [s for s in segs if s[2] - s[1] >= MIN_SEG]
    keep = []
    for i, (label, start, end) in enumerate(segs):
        if end - start < INTERJECTION and 0 < i < len(segs) - 1 and segs[i - 1][0] == segs[i + 1][0] != label:
            continue
        keep.append((label, start, end))
    return keep


def _relabel(segs):
    """Speakers become 1..n in order of first appearance."""
    ids: dict = {}
    out = []
    for label, start, end in segs:
        ids.setdefault(label, len(ids) + 1)
        out.append((ids[label], start, end))
    return out


def _read_wav(path: Path):
    """Mono 16-bit PCM wav -> (float32 samples in [-1, 1], sample rate)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    return data, sr


def _load_envelope(path: Path):
    data, sr = _read_wav(path)
    hop = int(sr * FRAME_S)
    n = len(data) // hop
    if n == 0:
        return np.zeros(1, dtype=np.float32), len(data) / sr
    frames = data[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames**2).mean(axis=1))
    return np.convolve(rms, np.ones(5) / 5, mode="same"), len(data) / sr


def _quiet_runs(mask):
    """[start, end) frame indices of each consecutive True stretch."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _silence_threshold(env):
    """Envelope level below which a frame is silent: SILENCE_DB under the file's 95th percentile."""
    return np.percentile(env, 95) * 10 ** (SILENCE_DB / 20)


def find_pauses(env):
    """Every silent stretch of at least MIN_PAUSE in the whole file, as time-sorted [(start, end)]."""
    return [
        (a * FRAME_S, b * FRAME_S)
        for a, b in _quiet_runs(env < _silence_threshold(env))
        if (b - a) * FRAME_S >= MIN_PAUSE
    ]


def _quiet_point(env, lo, hi):
    """Time of the lowest SUSTAIN-averaged energy between lo and hi."""
    w = max(int(SUSTAIN / FRAME_S), 1)
    i0 = int(lo / FRAME_S)
    seg = env[i0 : min(int(hi / FRAME_S), len(env))]
    if len(seg) < w:
        return (lo + hi) / 2
    sustained = np.convolve(seg, np.ones(w) / w, mode="valid")
    t = (i0 + int(np.argmin(sustained)) + w // 2) * FRAME_S + FRAME_S / 2
    return min(max(t, lo), hi)


def _split(start, end, env, pauses):
    """Cut a run into chunks of at most MAX_CHUNK, recursively: a piece that's too long is cut in its middle
    FALLBACK_WINDOW, at the longest pause whose midpoint lies there (the pause belongs to neither half), else
    at the quietest point there; each half is cut again while it's too long."""
    if end - start <= MAX_CHUNK:
        return [(start, end)]
    mid = (start + end) / 2
    lo, hi = mid - FALLBACK_WINDOW / 2, mid + FALLBACK_WINDOW / 2
    inside = [(s, e) for s, e in pauses if start < s and e < end and lo <= (s + e) / 2 <= hi]
    if inside:
        a, b = max(inside, key=lambda p: (p[1] - p[0], -abs((p[0] + p[1]) / 2 - mid)))
    else:
        a = b = _quiet_point(env, lo, hi)
    return _split(start, a, env, pauses) + _split(b, end, env, pauses)


