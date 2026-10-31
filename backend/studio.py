"""Studio flow (/v2): upload → separate → diarize → cut each speaker's turns at their
pauses (cuts.py), then stop and wait for the user; on request, ASR, with every segment published as soon as its
transcript comes back (failed ones tried once more a few minutes later), then Gemma judges every cut between two
segments of one speaker and the cuts it joins are merged, the merged chunks are cleaned (punctuation, capitals,
misheard words, fillers; asr.punctuate_chunks), then (if a target
language was picked) batched translation, published batch by batch, then a check of the translations by an
LLM (review.py) that rewrites the ones that would run long or read wrongly (run again on request); on request, a dub: TTS per segment
with the voices the user picked, each chunk fitted on its own (dubbing.FIT_DESCRIPTION), mixed with the
background; on request, lip-sync of the dubbed video (lipsync/). Every finished step is
saved, so past runs reopen where they stopped without redoing anything.

Reuses the pipeline's components (trim / audio extraction / Mel-Band RoFormer separation from main.py,
diarization.py, cuts.py, asr.py, translate.py, review.py, tts.py, dubbing.py). Each run is a folder in storage/
with its run.json."""

import json
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from . import asr, cuts, diarization, dubbing, gender, portraits, review, translate, tts, youtube
except ImportError:
    import asr
    import cuts
    import diarization
    import dubbing
    import gender
    import portraits
    import review
    import translate
    import tts
    import youtube

router = APIRouter(prefix="/v2")

RUN_ID = re.compile(r"^[0-9a-f]{32}$")

# The furthest stage a run has reached. Redoing an earlier stage keeps it (and the results after it).
STAGES = ["diarized", "transcribed", "translated", "dubbed"]

# Statuses while a step runs in the background; a run in one of these can't start another step
RUNNING = {
    "uploaded", "trimming", "extracting_audio", "separating", "diarizing", "chunking", "transcribing", "translating",
    "reviewing", "synthesizing", "dubbing", "deleting", "lipsync_queued", "lipsync_analysing", "lipsyncing",
    "refreshing",
}

# Lip-sync (lipsync/run.sh: a memory-capped subprocess on a free GPU). One job at a time across the app, since
# it takes a whole GPU; the others wait as lipsync_queued.
LIPSYNC = Path(__file__).resolve().parent.parent / "lipsync"
_lipsync_slot = threading.Semaphore(1)

# Set by main.py: storage root, ffmpeg binary, and the pipeline's trim / extract / separate helpers
_cfg: dict = {}


def configure(storage_dir: Path, ffmpeg_bin: str, trim, extract_audio, separate):
    _cfg.update(storage_dir=storage_dir, ffmpeg_bin=ffmpeg_bin, trim=trim, extract_audio=extract_audio, separate=separate)
    youtube.configure(storage_dir)


# In-memory run records. A record is replaced, never changed in place, so a request can serialise one
# while a step's thread moves on. `version` goes up with every change.
runs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(run_id: str, /, **fields):
    with _lock:
        old = runs.get(run_id, {})
        runs[run_id] = {**old, **fields, "version": old.get("version", 0) + 1}


def _claim(run_id: str, status: str) -> bool:
    """Set the run to `status` unless a step is already running. Returns whether it was set."""
    with _lock:
        old = runs.get(run_id, {})
        if old.get("status") in RUNNING:
            return False
        runs[run_id] = {**old, "status": status, "error": None, "version": old.get("version", 0) + 1}
        return True


def _start(run_id: str, step, *args):
    """Run a step in a background thread; the UI follows it by polling GET /v2/runs/{id}."""

    def run():
        try:
            step(run_id, *args)
        except Exception as exc:
            _set(run_id, status="error", error=str(exc), failed_at=runs.get(run_id, {}).get("status"))

    threading.Thread(target=run, daemon=True).start()


def _run_dir(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise HTTPException(status_code=404, detail="no run with this id")
    return _cfg["storage_dir"] / run_id


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- run.json: what a run is, and how far it got ---


def _save_info(jdir: Path, info: dict):
    (jdir / "run.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))


def _read_info(jdir: Path) -> dict | None:
    """run.json of a studio run; None if the folder has none."""
    if (jdir / "run.json").exists():
        return json.loads((jdir / "run.json").read_text())
    return None


MERGED_KEYS = {"id", "speaker", "start", "end", "cut_start", "cut_end", "text", "text_raw", "punctuated", "error", "sub_chunks", "wps", "pause_share", "slow"}


def _segments(chunks: list[dict]) -> list[dict]:
    """What the UI shows: one row per chunk, i.e. per merged chunk once the cuts are checked (before that, per
    segment of the pause cuts), as it is translated, checked and dubbed. `parts` is how many segments were merged
    into it and `inner_pauses` the cuts merged away ({at, duration, reason}). Builds new dicts, so it doubles as a
    snapshot."""
    rows = []
    for c in chunks:
        subs = c.get("sub_chunks") or [c]
        rows.append({
            "id": c["id"],
            "parent_id": c["id"],
            "speaker": c["speaker"],
            "start": c["start"],
            "end": c["end"],
            "text": c.get("text"),
            "error": c.get("error"),
            "parts": len(subs),
            "wps": c.get("wps"),
            "pause_share": c.get("pause_share"),
            "slow": bool(c.get("slow")),
            "inner_pauses": [
                {"at": a["end"], "duration": round(b["start"] - a["end"], 3), "reason": a.get("short_reason") or a.get("cut_reason")}
                for a, b in zip(subs, subs[1:])
            ],
            "translation": c.get("translation"),
            "translation_overflow": None,
            "translation_error": c.get("translation_error"),
            "translation_original": c.get("translation_original"),
            "review": _sub_review(c, c),
            "review_error": c.get("review_error"),
            "tts": bool(c.get("tts_file")),
            "tts_voice": c.get("tts_voice"),
            "tts_duration": c.get("tts_duration"),
            "tts_error": c.get("tts_error"),
            "dub": c.get("dub"),
            "dub_error": c.get("dub_error"),
            "text_edited": any(x.get("text_edited") for x in [c, *subs]),
            "translation_edited": any(x.get("translation_edited") for x in [c, *subs]),
            "retranslate": bool(c.get("retranslate")),
            "refresh": bool(c.get("refresh")),
        })
    return rows


def _sub_review(c: dict, s: dict) -> dict | None:
    """The check's result for the merged chunk `c`, as seen from its segment `s`: "reviewed" is the segment's
    slice of the edit."""
    r = c.get("review")
    if not r or s is c:
        return r
    return {**r, "reviewed": s.get("translation_reviewed")}


def _find(chunks: list[dict], seg_id: int):
    """The chunk with the UI row id `seg_id`, twice (the row is the whole chunk, so it is also the segment edited).
    (None, None) if there is no such chunk."""
    c = next((c for c in chunks if c["id"] == seg_id), None)
    return c, c


def _unmerge(chunks: list[dict]) -> list[dict]:
    """The short segments of the merged chunks, with their texts, errors and edits, as before the merge."""
    return [dict(s) for c in chunks for s in c.get("sub_chunks", [c])]


def _merge_again(before: list[dict], shorts: list[dict], joins: set[int], commas: set[int] | None = None) -> list[dict]:
    """merge_chunks across the cuts in `joins` (a comma at those in `commas`), keeping what each merged chunk had in
    the earlier pass (translation, check, speech, dub). A chunk whose text changed is marked for translating again."""
    merged = asr.merge_chunks(shorts, joins=joins, commas=commas)
    old = {c["id"]: c for c in before if c.get("sub_chunks")}
    for c in merged:
        prev = old.get(c["id"])
        if prev is None:
            continue
        for k, v in prev.items():
            if k not in MERGED_KEYS:
                c.setdefault(k, v)
        if prev.get("text") != c.get("text") and prev.get("translation") is not None:
            for unit in [c, *c["sub_chunks"]]:
                unit["retranslate"] = unit["refresh"] = True
        if c.get("translation") is not None:
            translate.map_sub_translations(c)
    return merged


def _split_punctuated(c: dict):
    """A merged chunk's punctuated text cut back into its segments, by their share of its words (each segment's
    ASR text kept in text_raw). Left as it is when the chunk isn't punctuated or a segment would get no words."""
    subs = [s for s in c.get("sub_chunks") or [] if s.get("text")]
    if not (c.get("punctuated") and c.get("text") and subs):
        return
    words = c["text"].split()
    counts = [len((s.get("text_raw") or s["text"]).split()) for s in subs]
    total = sum(counts)
    if not total:
        return
    cum, bounds = 0, [0]
    for n in counts:
        cum += n
        bounds.append(round(cum * len(words) / total))
    if any(b <= a for a, b in zip(bounds, bounds[1:])):
        return
    for s, a, b in zip(subs, bounds, bounds[1:]):
        s.setdefault("text_raw", s["text"])
        s["text"] = " ".join(words[a:b])
        s["punctuated"] = True


def _upgrade(jdir: Path, chunks: list[dict]) -> list[dict]:
    """Chunks saved before the short segments were shown: each segment gets the fields the merge now copies
    (from chunks_asr_raw.json, else its chunk's), and its slice of the chunk's translation."""
    if not any(s.get("speaker") is None for c in chunks for s in c.get("sub_chunks") or []):
        return chunks
    raw_path = jdir / "chunks_asr_raw.json"
    raw = {r["id"]: r for r in json.loads(raw_path.read_text())} if raw_path.exists() else {}
    for c in chunks:
        for s in c.get("sub_chunks") or []:
            r = raw.get(s["id"], {})
            s.setdefault("speaker", r.get("speaker", c["speaker"]))
            s.setdefault("cut_start", r.get("cut_start", s["start"]))
            s.setdefault("cut_end", r.get("cut_end", s["end"]))
        if c.get("translation") is not None:
            translate.map_sub_translations(c)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return chunks


def _set_flag(c: dict, sub: dict, name: str, on: bool = True):
    """Mark the segment `sub` of chunk `c` with `name`; the chunk carries the mark while any of its segments does.
    Pass on=False to clear it."""
    if on:
        sub[name] = True
        c[name] = True
        return
    if sub is c:  # the whole chunk was edited: its segments' marks go too
        _clear_all(c, name)
        return
    sub.pop(name, None)
    if not any(s.get(name) for s in c.get("sub_chunks") or []):
        c.pop(name, None)


def _clear_all(c: dict, *names: str):
    """Clear the marks on the chunk and on each of its segments."""
    for unit in [c, *(c.get("sub_chunks") or [])]:
        for name in names:
            unit.pop(name, None)


def _reached(info: dict, stage: str) -> bool:
    return info["stage"] in STAGES and STAGES.index(info["stage"]) >= STAGES.index(stage)


def _advance(info: dict, stage: str):
    """Set the run's stage, unless it has already got further (a redo of an earlier stage)."""
    if not _reached(info, stage):
        info["stage"] = stage


def _mark_stale(info: dict, chunks: list[dict], *names: str):
    """Redoing a stage keeps the results after it, but they no longer match. Note the ones that exist:
    "translation", "dub", "lipsync". Each is cleared when its stage is run again."""
    exist = {
        "translation": any(c.get("translation") for c in chunks),
        "dub": bool(info.get("dub")),
        "lipsync": bool(info.get("lipsync")),
    }
    info["stale"] = sorted(set(info.get("stale", [])) | {n for n in names if exist[n]})


def _unstale(info: dict, *names: str):
    info["stale"] = sorted(set(info.get("stale", [])) - set(names))


def _speakers(run_id: str, chunks: list[dict]) -> list[dict]:
    """Per speaker: segment count, seconds of speech, the detected gender ("female", "male", "unknown";
    None when diarization.json has none) and the URL of their face (portraits.py; None without one)."""
    jdir = _run_dir(run_id)
    genders, faces = gender.load_details(jdir), portraits.load(jdir)
    out: dict[int, dict] = {}
    for c in _segments(chunks):
        s = out.setdefault(c["speaker"], {"speaker": c["speaker"], "segments": 0, "seconds": 0.0})
        s["segments"] += 1
        s["seconds"] += c["end"] - c["start"]
    return [
        {
            **s,
            "seconds": round(s["seconds"], 1),
            "gender": genders.get(s["speaker"], {}).get("gender"),
            "gender_confidence": genders.get(s["speaker"], {}).get("confidence"),
            "gender_manual": bool(genders.get(s["speaker"], {}).get("manual")),
            "face_url": f"/v2/runs/{run_id}/speakers/{s['speaker']}/face?v={faces[s['speaker']]}"
            if s["speaker"] in faces else None,
        }
        for s in sorted(out.values(), key=lambda s: s["speaker"])
    ]


def _asr_done(chunks: list[dict]) -> int:
    return sum(1 for x in _segments(chunks) if x["text"] is not None or x["error"])


def _translate_total(chunks: list[dict]) -> int:
    """Chunks translation works on: those with text (failed or empty transcripts are skipped)."""
    return sum(1 for x in _segments(chunks) if x["text"])


def _translate_done(chunks: list[dict]) -> int:
    return sum(1 for x in _segments(chunks) if x["text"] and (x["translation"] is not None or x["translation_error"]))


def _review_total(chunks: list[dict]) -> int:
    """Chunks the translation check works on: those with a translation that the user didn't edit."""
    return sum(1 for c in chunks if c.get("translation") and not review.edited(c))


def _review_done(chunks: list[dict]) -> int:
    return sum(
        1 for c in chunks
        if c.get("translation") and not review.edited(c) and (c.get("review") or c.get("review_error"))
    )


def _tts_total(chunks: list[dict]) -> int:
    """Chunks TTS works on: those with a translation."""
    return sum(1 for x in _segments(chunks) if x["translation"])


def _tts_done(chunks: list[dict]) -> int:
    return sum(1 for x in _segments(chunks) if x["translation"] and (x["tts"] or x["tts_error"]))


def _voices(jdir: Path, info: dict, chunks: list[dict]) -> dict:
    """A voice per speaker, as {"1": voice}: the last dub's voices, the others assigned automatically by the
    detected gender, the longest speaker first (None where the gender is unknown: the user picks)."""
    speakers = tts.by_talk_time(chunks)  # the longest speaker gets the best voice
    chosen = {int(s): v for s, v in (info.get("voices") or {}).items()}
    return tts.suggest_voices(jdir, speakers, info.get("target_language"), chosen)


def _record(run_id: str, info: dict, chunks: list[dict]) -> dict:
    """A run's full state after a finished step, or loaded from disk."""
    return {
        "run_id": run_id,
        "name": info["name"],
        "created": info["created"],
        "language": info["language"],
        "language_name": asr.LANGUAGES.get(info["language"], info["language"]),
        "range": info.get("range"),
        "stage": info["stage"],
        "stale": info.get("stale", []),
        "timings": info.get("timings", {}),
        "asr_failed": info.get("asr_failed"),
        "target_language": info.get("target_language"),
        "target_language_name": translate.LANGUAGES.get(info.get("target_language"), info.get("target_language")),
        "translation_failed": info.get("translation_failed"),
        "num_speakers": info.get("num_speakers"),
        "input_url": f"/download/{run_id}/input",
        "speakers": _speakers(run_id, chunks),
        "segments": _segments(chunks),
        "asr_done": _asr_done(chunks),
        "asr_total": len(_segments(chunks)),
        "translate_done": _translate_done(chunks),
        "translate_total": _translate_total(chunks),
        "review": info.get("review"),
        "review_prompt": info.get("review_prompt") or review.PROMPT,
        "review_prompt_custom": bool(info.get("review_prompt")),
        "review_done": _review_done(chunks),
        "review_total": _review_total(chunks),
        "voices": _voices(_run_dir(run_id), info, chunks) if info.get("target_language") else {},
        "tts_failed": info.get("tts_failed"),
        "tts_style": info.get("tts_style"),
        "tts_done": _tts_done(chunks),
        "tts_total": _tts_total(chunks),
        "dub": info.get("dub"),
        "dubbed_audio_url": f"/download/{run_id}/dubbed_audio" if info["stage"] == "dubbed" else None,
        "dubbed_video_url": f"/download/{run_id}/dubbed_video" if info["stage"] == "dubbed" else None,
        "lipsync": info.get("lipsync"),
        "lipsync_video_url": f"/download/{run_id}/lipsynced_video" if info.get("lipsync") else None,
        "refreshes": info.get("refreshes", 0),
    }


def _load(run_id: str):
    """A saved run's folder, run info and chunks. Puts the run in memory if it isn't. 404 if none."""
    jdir = _run_dir(run_id)
    info = _read_info(jdir) if jdir.is_dir() else None
    if info is None:
        raise HTTPException(status_code=404, detail="no saved run with this id")
    chunks = _upgrade(jdir, json.loads((jdir / "chunks.json").read_text()))
    if run_id not in runs:
        _set(run_id, **_record(run_id, info, chunks), status=info["stage"])
    return jdir, info, chunks


def _summary(record: dict) -> dict:
    return {
        "run_id": record["run_id"],
        "name": record.get("name"),
        "created": record.get("created"),
        "language": record.get("language"),
        "range": record.get("range"),
        "status": record.get("status"),
        "stage": record.get("stage"),
        "target_language": record.get("target_language"),
        "segments": len(record.get("segments") or []),
        "num_speakers": record.get("num_speakers"),
    }


# --- endpoints ---


@router.get("/runs")
def list_runs():
    """Past runs (saved after diarization), plus runs still being processed or failed before that,
    newest first."""
    out = {}
    for d in _cfg["storage_dir"].iterdir():
        if not (RUN_ID.match(d.name) and d.is_dir()):
            continue
        record = runs.get(d.name)
        if record and record.get("stage"):  # in memory: its live status
            out[d.name] = _summary(record)
            continue
        info = _read_info(d)
        if info:
            chunks = _upgrade(d, json.loads((d / "chunks.json").read_text()))
            out[d.name] = _summary({**_record(d.name, info, chunks), "status": info["stage"]})
    for run_id, record in runs.items():  # not saved yet: still before diarization finished, or failed there
        if run_id not in out:
            out[run_id] = _summary(record)
    return sorted(out.values(), key=lambda r: r["created"] or "", reverse=True)[:50]


@router.post("/youtube/info")
def youtube_info(url: str = Body(..., embed=True)):
    """Title, length and thumbnail of a YouTube link, without downloading it, so the part to download can be
    picked."""
    try:
        return youtube.probe(url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/youtube")
def fetch_youtube(
    url: str = Body(...),
    height: int = Body(youtube.MAX_HEIGHT),
    start: float | None = Body(None),
    end: float | None = Body(None),
):
    """Start downloading a YouTube video (at most 720p), or only start..end of it (seconds; at most 60 min),
    to start a run from. Poll GET /v2/youtube/{fetch_id}; when it's "ready", create the run with that
    fetch_id."""
    try:
        return youtube.start(url, height, _cfg["ffmpeg_bin"], start, end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/youtube/{fetch_id}")
def get_youtube(fetch_id: str):
    rec = youtube.get(fetch_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="no download with this id")
    return rec


@router.get("/youtube/{fetch_id}/video")
def get_youtube_video(fetch_id: str):
    path = youtube.video_path(fetch_id)
    if path is None:
        raise HTTPException(status_code=404, detail="the video isn't downloaded (yet)")
    return FileResponse(path, media_type="video/mp4")


@router.delete("/youtube/{fetch_id}")
def discard_youtube(fetch_id: str):
    if not youtube.discard(fetch_id):
        raise HTTPException(status_code=404, detail="no download with this id")
    return {"fetch_id": fetch_id, "deleted": True}


@router.post("/runs")
async def create_run(
    file: UploadFile | None = File(None),
    fetch_id: str | None = Form(None),
    language: str = Form(...),
    range_start: float | None = Form(None),
    range_end: float | None = Form(None),
):
    """Upload a video (or give the fetch_id of a finished YouTube download), the spoken language, and
    optionally the portion to work on. In the background: trim, extract audio, separate, diarize, chunk.
    Stops at status "diarized"."""
    if (file is None) == (fetch_id is None):
        raise HTTPException(status_code=400, detail="send either a file or a fetch_id")
    if fetch_id is not None and youtube.video_path(fetch_id) is None:
        raise HTTPException(status_code=400, detail="that download isn't ready (or doesn't exist)")
    if language not in asr.LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language: {language}")
    portion = None
    if range_start is not None or range_end is not None:
        start = max(range_start or 0.0, 0.0)
        if range_end is not None and range_end <= start:
            raise HTTPException(status_code=400, detail="the portion's end must be after its start")
        portion = [round(start, 3), None if range_end is None else round(range_end, 3)]
    try:
        asr.api_key("asr")  # fail now rather than after diarization
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    run_id = uuid.uuid4().hex
    jdir = _cfg["storage_dir"] / run_id
    jdir.mkdir(parents=True)

    # a trimmed upload is kept as original.*; input.* is always the video the run works on
    suffix = ".mp4" if file is None else Path(file.filename or "").suffix or ".mp4"
    upload_path = jdir / f"{'original' if portion else 'input'}{suffix}"
    source_url = source_section = None
    if file is None:
        fetched = youtube.take(fetch_id, upload_path)
        name, source_url, source_section = fetched["title"], fetched["url"], fetched.get("section")
    else:
        name = file.filename or "video"
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

    info = {
        "run_id": run_id,
        "name": name,
        "source_url": source_url,
        "source_section": source_section,
        "created": _now(),
        "language": language,
        "range": portion,
        "stage": None,
    }
    _set(
        run_id,
        **{k: info[k] for k in ("run_id", "name", "created", "language", "range", "stage")},
        language_name=asr.LANGUAGES[language],
        status="uploaded",
        segments=[],
        speakers=[],
        timings={},
    )
    _start(run_id, _diarize_step, jdir, upload_path, info)
    return runs[run_id]


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    """A run's live state: from memory while it's known, otherwise loaded from disk."""
    if run_id in runs:
        return runs[run_id]
    _load(run_id)
    return runs[run_id]


def _check_target(info: dict, target: str | None):
    """400 for an unknown target language; 500 if translation would need a key that isn't set."""
    if target is None:
        return
    if target not in translate.LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported target language: {target}")
    if target != info["language"]:  # same language: the text is copied, no requests
        try:
            asr.api_key("translate")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))


@router.post("/runs/{run_id}/transcribe")
def transcribe_run(run_id: str, body: dict = Body(default={})):
    """ASR for a diarized run, in the background. Each segment is published as soon as its transcript
    arrives. With {"target_language": code}, the transcripts are then translated in batches, each batch
    published as it returns. A run that's already transcribed is returned as it is (no requests), unless
    {"redo": true}: then every segment is transcribed again, optionally in another {"language": code}. The
    translation, dub and lip-sync are kept but marked outdated (`stale`) until they are run again."""
    jdir, info, _ = _load(run_id)
    redo = bool(body.get("redo"))
    if _reached(info, "transcribed") and not redo:
        return runs[run_id]
    language = body.get("language") or None
    if language is not None and language not in asr.LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language: {language}")
    target = body.get("target_language") or None
    _check_target(info, target)
    try:
        asr.api_key("asr")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not _claim(run_id, "transcribing"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    # known from the start, so the UI can show that translation follows
    if not redo or target:  # a redo without a target keeps the one it had
        _set(run_id, target_language=target, target_language_name=translate.LANGUAGES.get(target))
    _start(run_id, _asr_step, jdir, info, target, redo, language)
    return runs[run_id]


@router.post("/runs/{run_id}/translate")
def translate_run(run_id: str, body: dict = Body(default={})):
    """Translation alone, for a transcribed run: one that was transcribed without a target, or whose
    translation failed. Body {"target_language": code}. A translated run is returned as it is, unless
    {"redo": true}: then every segment is translated (and checked) again, optionally to another target.
    The dub and lip-sync are kept but marked outdated until they are run again."""
    jdir, info, _ = _load(run_id)
    redo = bool(body.get("redo"))
    if _reached(info, "translated") and not redo:
        return runs[run_id]
    if not _reached(info, "transcribed"):
        raise HTTPException(status_code=409, detail="transcribe the run first")
    target = body.get("target_language") or info.get("target_language")
    if not target:
        raise HTTPException(status_code=400, detail="target_language is required")
    _check_target(info, target)
    if not _claim(run_id, "translating"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    chunks = json.loads((jdir / "chunks.json").read_text())
    _start(run_id, _translate_step, jdir, info, chunks, target, redo)
    return runs[run_id]


@router.get("/review-prompt")
def default_review_prompt():
    """The system prompt the translation check uses unless a run has its own."""
    return {"prompt": review.PROMPT, "max_chars": review.MAX_PROMPT_CHARS}


@router.put("/runs/{run_id}/review-prompt")
def set_review_prompt(run_id: str, body: dict = Body(default={})):
    """Save this run's own system prompt for the translation check. {"prompt": text}; an empty prompt, or the
    default text, goes back to the default. {source} and {target} in it are replaced by the language names.
    Nothing is run: POST /review uses it."""
    jdir, info, _ = _load(run_id)
    prompt = str(body.get("prompt") or "").strip()
    if len(prompt) > review.MAX_PROMPT_CHARS:
        raise HTTPException(status_code=400, detail=f"the prompt is longer than {review.MAX_PROMPT_CHARS} characters")
    with _lock:  # held throughout, so no step can start meanwhile
        if runs.get(run_id, {}).get("status") in RUNNING:
            raise HTTPException(status_code=409, detail="a step is already running for this run")
        if not prompt or prompt == review.PROMPT.strip():
            info.pop("review_prompt", None)
        else:
            info["review_prompt"] = prompt
        _save_info(jdir, info)
    _set(run_id, review_prompt=info.get("review_prompt") or review.PROMPT, review_prompt_custom=bool(info.get("review_prompt")))
    return runs[run_id]


@router.post("/runs/{run_id}/review")
def review_run(run_id: str, body: dict = Body(default={})):
    """The translation check, again: it runs on its own after every translation, and this retries or redoes it.
    It reads the translations with the run's system prompt (see PUT /review-prompt) and applies its edits, each
    of which can be undone. Segments already checked keep their result, so this checks only the ones not checked yet or that
    failed. With {"redo": true} every edit is undone first and all segments are checked again. A dubbed run
    keeps its dub, marked outdated if the check changed any translation."""
    jdir, info, chunks = _load(run_id)
    if not _reached(info, "translated"):
        raise HTTPException(status_code=409, detail="translate the run first")
    if info.get("target_language") == info["language"]:
        raise HTTPException(status_code=409, detail="the translation is the transcript copied; there is nothing to check")
    try:
        review.api_key()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not _claim(run_id, "reviewing"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    _start(run_id, _review_only_step, jdir, info, chunks, bool(body.get("redo")))
    return runs[run_id]


def _edited_text(body: dict) -> str:
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="the text can't be empty")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="the text is too long (at most 5000 characters)")
    return text


def _edit_segment(run_id: str, seg_id: int, stage: str, edit):
    """Load the run, check it's editable, apply `edit(info, chunk, segment)` to the short segment `seg_id` and its
    merged chunk, and save chunks.json."""
    jdir, info, _ = _load(run_id)
    if not _reached(info, stage):
        raise HTTPException(status_code=409, detail=f"the run isn't {stage} yet")
    with _lock:  # held throughout, so no step can start meanwhile
        if runs.get(run_id, {}).get("status") in RUNNING:
            raise HTTPException(status_code=409, detail="a step is already running for this run")
        chunks = json.loads((jdir / "chunks.json").read_text())
        c, s = _find(chunks, seg_id)
        if c is None:
            raise HTTPException(status_code=404, detail="no segment with this id")
        edit(info, c, s)
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    _set(run_id, segments=_segments(chunks))
    return runs[run_id]


@router.put("/runs/{run_id}/segments/{seg_id}/text")
def edit_text(run_id: str, seg_id: int, body: dict = Body(default={})):
    """The user's own transcript for a segment: body {"text": ...}. Once the segment has a translation, it is
    marked `retranslate` and `refresh`: POST .../refresh translates it again (and redoes its dub and lip-sync).
    Edited segments are never sent to the translation check."""
    text = _edited_text(body)

    def edit(info, c, s):
        if text == s.get("text"):
            return
        s["text"] = text
        s.pop("error", None)
        _set_flag(c, s, "text_edited")
        if s is not c:  # the merged chunk is translated as one: its text is its segments'
            c["text"] = " ".join(x["text"].strip() for x in c["sub_chunks"] if x.get("text") and x["text"].strip())
            if not any(x.get("error") for x in c["sub_chunks"]):
                c.pop("error", None)
        # once translated (or the run is), it needs its own translation; else the translation step does it
        if c.get("translation") is not None or _reached(info, "translated"):
            _set_flag(c, s, "retranslate")
            _set_flag(c, s, "refresh")

    return _edit_segment(run_id, seg_id, "transcribed", edit)


@router.put("/runs/{run_id}/segments/{seg_id}/translation")
def edit_translation(run_id: str, seg_id: int, body: dict = Body(default={})):
    """The user's own translation for a segment: body {"text": ...}. Replaces the check's result and any
    pending re-translation. In a dubbed run the segment is marked `refresh`: POST .../refresh redoes its
    speech, its fit and its lip-sync."""
    text = _edited_text(body)

    def edit(info, c, s):
        if text == s.get("translation") and not s.get("retranslate"):
            return
        for k in ("translation_error", "review", "review_error", "translation_original"):
            c.pop(k, None)
        _set_flag(c, s, "retranslate", on=False)
        _set_flag(c, s, "translation_edited")
        if s is c:
            review.set_translation(c, text)
        else:  # the merged chunk is spoken as one: its translation is its segments', joined
            s["translation"] = text
            review.set_translation(c, " ".join(x["translation"] for x in c["sub_chunks"] if x.get("translation")))
        _set_flag(c, s, "refresh", on=info["stage"] == "dubbed")

    return _edit_segment(run_id, seg_id, "translated", edit)


@router.put("/runs/{run_id}/segments/{seg_id}/speaker")
def edit_speaker(run_id: str, seg_id: int, body: dict = Body(default={})):
    """Give a segment to another of the run's speakers: body {"speaker": n}. A merged chunk moves whole. In a
    dubbed run the segment is marked `refresh`: POST .../refresh redoes its speech with its new speaker's voice,
    and its lip-sync for both speakers (`moved_from` keeps the earlier ones, whose old clips there must go)."""
    speaker = body.get("speaker")
    _, _, chunks = _load(run_id)
    if not isinstance(speaker, int) or speaker not in {c["speaker"] for c in chunks}:
        raise HTTPException(status_code=400, detail="speaker must be one of the run's speakers")

    def edit(info, c, s):
        if c["speaker"] == speaker:
            return
        if info["stage"] == "dubbed":
            c["moved_from"] = sorted({*c.get("moved_from", []), c["speaker"]} - {speaker})
        for unit in [c, *(c.get("sub_chunks") or [])]:
            unit["speaker"] = speaker
        if info["stage"] == "dubbed":
            _set_flag(c, c, "refresh")

    _edit_segment(run_id, seg_id, "diarized", edit)
    _set(run_id, speakers=_speakers(run_id, json.loads((_run_dir(run_id) / "chunks.json").read_text())))
    return runs[run_id]


@router.post("/runs/{run_id}/segments/{seg_id}/refresh")
def refresh_segment(run_id: str, seg_id: int):
    """Redo one edited segment (marked `refresh`) and nothing else: its translation if its transcript was
    edited, then, in a dubbed run, its speech with the run's voices and style, its fit spliced into the dub, and, if the run was lip-synced, the lip-sync of its part. Status "refreshing"; the record's
    `refresh` = {segment, step} says which part is running."""
    jdir, info, chunks = _load(run_id)
    c, _ = _find(chunks, seg_id)  # the merged chunk is translated and dubbed as one, so it is refreshed whole
    if c is None:
        raise HTTPException(status_code=404, detail="no segment with this id")
    if not c.get("refresh"):
        raise HTTPException(status_code=409, detail="nothing to refresh: the segment wasn't edited since")
    try:
        if c.get("retranslate") and info["target_language"] != info["language"]:
            asr.api_key("translate")
        if info["stage"] == "dubbed":
            asr.api_key("tts")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not _claim(run_id, "refreshing"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    _set(run_id, refresh={"segment": seg_id, "step": None}, lipsync_progress=None)
    _start(run_id, _refresh_step, jdir, info, chunks, seg_id)
    return runs[run_id]


@router.post("/runs/{run_id}/speakers/{speaker}/gender")
def set_speaker_gender(run_id: str, speaker: int, body: dict = Body(default={})):
    """Corrects a speaker's gender: body {"gender": "female" | "male" | "unknown"}. Saved in diarization.json
    (marked `manual`), so the suggested voices and the next dub follow it; a dubbed run keeps its dub until it
    is dubbed again. "unknown" means the user has to pick that speaker's voice."""
    gender_value = body.get("gender")
    if gender_value not in ("female", "male", "unknown"):
        raise HTTPException(status_code=400, detail='gender must be "female", "male" or "unknown"')
    jdir, info, chunks = _load(run_id)
    if speaker not in {c["speaker"] for c in chunks}:
        raise HTTPException(status_code=404, detail="no speaker with this number")
    with _lock:  # held throughout, so no step can start meanwhile
        if runs.get(run_id, {}).get("status") in RUNNING:
            raise HTTPException(status_code=409, detail="a step is already running for this run")
        path = jdir / "diarization.json"
        data = json.loads(path.read_text())
        entry = data.setdefault("genders", {}).setdefault(str(speaker), {"confidence": None, "seconds": 0})
        entry.update(gender=gender_value, manual=True)
        path.write_text(json.dumps(data, indent=2))
    fields = {"speakers": _speakers(run_id, chunks)}
    if info.get("target_language"):
        kept = {k: v for k, v in (info.get("voices") or {}).items() if k != str(speaker)}  # re-suggested
        fields["voices"] = _voices(jdir, {**info, "voices": kept}, chunks)
    _set(run_id, **fields)
    return runs[run_id]


@router.get("/runs/{run_id}/speakers/{speaker}/face")
def speaker_face(run_id: str, speaker: int):
    """A speaker's face, a small square JPEG (portraits.py); 404 if they have none."""
    path = portraits.path(_run_dir(run_id), speaker)
    if path is None:
        raise HTTPException(status_code=404, detail="no face for this speaker")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "max-age=31536000, immutable"})


@router.post("/runs/{run_id}/dub")
def dub_run(run_id: str, body: dict = Body(default={})):
    """TTS for every translated segment with the chosen voices, then the speed-up-only dub (dubbing.build_speed_dub),
    in the background. Body {"voices": {"<speaker>": voice}, "style": one of tts.STYLES or null}; speakers left out
    get a voice automatically, and no style means the neutral reading. A dubbed run can be dubbed
    again (other voices): TTS is redone for segments whose voice or style changed, or that failed, then the
    dub is redone. POST /refit redoes only the fit."""
    jdir, info, chunks = _load(run_id)
    if info["stage"] not in ("translated", "dubbed"):
        raise HTTPException(status_code=409, detail="translate the run first")
    voices = {}
    for s, v in (body.get("voices") or {}).items():
        if v not in tts.ALL_VOICES:
            raise HTTPException(status_code=400, detail=f"unknown voice: {v}")
        try:
            voices[int(s)] = v
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid speaker: {s}")
    style = body.get("style") or None
    if style is not None and style not in tts.STYLES:
        raise HTTPException(status_code=400, detail=f"unknown speaking style: {style}")
    try:
        asr.api_key("tts")
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if not _claim(run_id, "synthesizing"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    _start(run_id, _dub_step, jdir, info, chunks, voices, style)
    return runs[run_id]


@router.post("/runs/{run_id}/refit")
def refit_run(run_id: str):
    """Redo only the fit and mix of a dubbed run from its stored TTS clips. No API requests."""
    jdir, info, chunks = _load(run_id)
    if info["stage"] != "dubbed":
        raise HTTPException(status_code=409, detail="dub the run first")
    if not _claim(run_id, "dubbing"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    _start(run_id, _fit_step, jdir, info, chunks)
    return runs[run_id]


@router.post("/runs/{run_id}/lipsync")
def lipsync_run(run_id: str):
    """Lip-sync a dubbed run in the background: find who is speaking on screen, sync their mouths to their dub
    with LatentSync, keep only parts that pass a SyncNet check, paste them back -> lipsynced.mp4. Progress is in
    the record's `lipsync_progress`. Redoes any earlier lip-sync of the run."""
    jdir, info, chunks = _load(run_id)
    if info["stage"] != "dubbed" or not (jdir / "dubbed.mp4").exists():
        raise HTTPException(status_code=409, detail="dub the run first")
    if not _claim(run_id, "lipsync_queued"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    _set(run_id, lipsync_progress=None)
    _start(run_id, _lipsync_step, jdir, info, chunks)
    return runs[run_id]


@router.delete("/runs/{run_id}")
def delete_run(run_id: str):
    """Delete a studio run: its folder in storage/ and its in-memory record. Not while a step is running (409)."""
    jdir = _run_dir(run_id)
    if not jdir.is_dir() and run_id not in runs:
        raise HTTPException(status_code=404, detail="no run with this id")
    # "deleting" is a running status, so no step can start while the folder is removed
    if not _claim(run_id, "deleting"):
        raise HTTPException(status_code=409, detail="a step is already running for this run")
    try:
        if jdir.is_dir():
            shutil.rmtree(jdir)
    except OSError as exc:
        _set(run_id, status="error", error=f"could not delete the run: {exc}")
        raise HTTPException(status_code=500, detail=f"could not delete the run: {exc}")
    with _lock:
        runs.pop(run_id, None)
    return {"run_id": run_id, "deleted": True}


# --- steps ---


def _diarize_step(run_id: str, jdir: Path, upload_path: Path, info: dict):
    video_path = upload_path
    if info["range"]:
        _set(run_id, status="trimming")
        video_path = jdir / "input.mp4"
        _cfg["trim"](upload_path, video_path, *info["range"])

    _set(run_id, status="extracting_audio", input_url=f"/download/{run_id}/input")
    audio_path = _cfg["extract_audio"](video_path, jdir)

    _set(run_id, status="separating")
    start = time.perf_counter()
    vocals_path = _cfg["separate"](audio_path, jdir)
    separation_s = time.perf_counter() - start

    _set(run_id, status="diarizing")
    start = time.perf_counter()
    turns = diarization.diarize_vocals(jdir, vocals_path, _cfg["ffmpeg_bin"])
    turns = _faces_step(run_id, jdir, video_path, turns)
    _set(run_id, status="chunking")
    chunks = cuts.split_pieces(jdir, turns)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    diarization_s = time.perf_counter() - start

    info.update(
        video=video_path.name,
        stage="diarized",
        num_speakers=len({t[0] for t in turns}),
        timings={"separation": round(separation_s, 2), "diarization": round(diarization_s, 2)},
    )
    _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status="diarized")


def _faces_step(run_id: str, jdir: Path, video_path: Path, turns):
    """Faces and Light-ASD on the original video (lipsync/faces.py, the lip-sync's GPU slot, still status
    "diarizing") -> faces.json, reused by the lip-sync pre-pass; its diarization fixes are recorded, and
    applied if diarization.ASD_APPLY, then each speaker's face is cropped (portraits.py). If it fails, the
    audio-only turns stay and no speaker has a face."""
    with _lipsync_slot:
        try:
            _lipsync_proc(run_id, jdir, "faces", video_path.name, log="faces.log")
        except Exception as exc:
            (jdir / "faces.json").unlink(missing_ok=True)
            return diarization.apply_faces(jdir, turns, error=str(exc))
    turns = diarization.apply_faces(jdir, turns)
    portraits.build(jdir, video_path, _cfg["ffmpeg_bin"])  # the final speakers' faces, for the UI
    return turns


def _asr_step(run_id: str, jdir: Path, info: dict, target: str | None, redo: bool = False, language: str | None = None):
    chunks = json.loads((jdir / "chunks.json").read_text())
    raw_path = jdir / "chunks_asr_raw.json"
    # the merged chunks of an earlier pass, whose translation and dub the new merge keeps
    before = chunks if any("sub_chunks" in c for c in chunks) else None
    if redo and raw_path.exists():  # a full redo starts again from the original diarization cuts
        chunks = json.loads(raw_path.read_text())
    elif before is not None:  # ASR works on the short segments
        chunks = _unmerge(chunks)
    if redo:  # every segment is sent again; what came after the transcript is kept, but outdated
        if language:
            info["language"] = language
        for c in chunks:
            c["text"] = None
            for k in ("error", "text_raw", "punctuated", "text_edited", "retranslate", "refresh", "wps", "pause_share", "pause_after", "slow"):
                c.pop(k, None)
        _mark_stale(info, chunks, "translation", "dub", "lipsync")
        _save_info(jdir, info)
        _set(run_id, language=info["language"], language_name=asr.LANGUAGES[info["language"]], stale=info["stale"])
    for c in chunks:  # a retry keeps the transcripts that arrived; only failed and missing segments are sent
        if c.get("error"):
            c["text"] = None
            c.pop("error")
    _set(run_id, segments=_segments(chunks), asr_done=_asr_done(chunks), asr_total=len(chunks), asr_note=None)

    def report():  # called after every chunk; the rows only change once the pieces are merged
        _set(run_id, asr_done=_asr_done(chunks))

    start = time.perf_counter()
    # one attempt per segment; the failed ones get one more, cuts.ASR_RETRY_DELAY_S after they failed
    cuts.transcribe_pieces(jdir, [c for c in chunks if c.get("text") is None], info["language"], progress=report)
    failed = [c for c in chunks if c.get("error")]
    if failed:
        retry_at = datetime.fromtimestamp(max(c["failed_at"] for c in failed) + cuts.ASR_RETRY_DELAY_S)
        _set(run_id, asr_note=f"{len(failed)} failed; one more try at {retry_at:%H:%M:%S}")
        for c in sorted(failed, key=lambda c: c["failed_at"]):
            time.sleep(max(c["failed_at"] + cuts.ASR_RETRY_DELAY_S - time.time(), 0))
            cuts.transcribe_pieces(jdir, [c], info["language"], progress=report)
    for c in chunks:
        c.pop("failed_at", None)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    if before is None or redo:  # the transcripts as they came back, for a redo
        raw_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))

    # Gemma: which cuts between a speaker's segments to join, so they are translated (and dubbed) as one
    _set(run_id, asr_note="Gemma is checking which cuts to join")
    merges, cut_stats = cuts.check_cuts(
        jdir, chunks, info["language"], target or info.get("target_language") or cuts.DEFAULT_TARGET
    )
    joins = {chunks[i]["id"] for i in merges}
    # then a segment still too short joins a near neighbour, with a comma (Gemma picks the side when both are near)
    _set(run_id, asr_note="Gemma is joining segments that are too short")
    shorts, short_stats = cuts.short_joins(jdir, chunks, joins, info["language"])
    info["cut_check"] = {
        **{k: v for k, v in cut_stats.items() if k != "dropped"},
        "short_joins": len(shorts), "short_check": short_stats,
    }
    joins |= shorts
    chunks = (
        asr.merge_chunks(chunks, joins=joins, commas=shorts) if before is None
        else _merge_again(before, chunks, joins, shorts)
    )
    _set(run_id, segments=_segments(chunks), asr_note="Cleaning up the transcript")
    texts = {c["id"]: c.get("text") for c in chunks}
    try:  # the merged chunks; then each segment gets its part of the cleaned text
        info["punctuation"] = asr.punctuate_chunks(jdir, chunks, info["language"], info.get("name") or "")
    except Exception as exc:  # no OpenRouter key or a failed request: the transcripts stay as they are
        info["punctuation"] = {"error": str(exc)}
    for c in chunks:  # a chunk of an earlier pass, already translated, whose text changed (as in _merge_again)
        if c.get("text") != texts[c["id"]] and c.get("translation") is not None:
            for unit in [c, *c.get("sub_chunks", [])]:
                unit["retranslate"] = unit["refresh"] = True
    for c in chunks:
        _split_punctuated(c)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    _set(run_id, asr_note=None)
    report()
    failed = sum(1 for c in chunks if c.get("error"))

    _advance(info, "transcribed")
    info["asr_failed"] = failed
    info["timings"] = {**info.get("timings", {}), "asr": round(time.perf_counter() - start, 2)}
    if target:  # kept even if translation then fails, so a retry offers the same language
        info["target_language"] = target
    _save_info(jdir, info)  # saved before translating, so a failed translation keeps the transcripts
    if target is None:
        _set(run_id, **_record(run_id, info, chunks), status=info["stage"])
        return
    _set(run_id, **_record(run_id, info, chunks), status="translating")
    _translate_step(run_id, jdir, info, chunks, target, redo)


def _translate_step(run_id: str, jdir: Path, info: dict, chunks: list[dict], target: str, redo: bool = False):
    clear = redo or target != info.get("target_language")
    if clear:  # the dub and lip-sync are kept, but no longer match the new translation
        _mark_stale(info, chunks, "dub", "lipsync")
        info.pop("review", None)  # the check was of the old translation
    for c in chunks:  # a retry keeps the translations that arrived (same target); failed ones are sent again
        if clear:
            if c.get("tts_file"):  # see review.set_translation
                c.setdefault("tts_text", c.get("translation"))
            c.pop("translation", None)
            _clear_all(c, "review", "review_error", "translation_original", "translation_edited", "retranslate", "refresh")
        c.pop("translation_error", None)
    info["target_language"] = target
    _save_info(jdir, info)  # saved first, so translations kept on disk after a failure match the target
    _set(
        run_id,
        target_language=target,
        target_language_name=translate.LANGUAGES[target],
        segments=_segments(chunks),
        translate_done=_translate_done(chunks),
        translate_total=_translate_total(chunks),
    )

    def report():  # called after every batch
        _set(run_id, segments=_segments(chunks), translate_done=_translate_done(chunks))

    start = time.perf_counter()
    stats = translate.translate_chunks(jdir, chunks, info["language"], target, progress=report)

    info["translation_failed"] = stats["failed"]
    info["translation_requests"] = stats["requests"]
    info["timings"] = {**info.get("timings", {}), "translation": round(time.perf_counter() - start, 2)}
    _save_info(jdir, info)
    _set(run_id, translation_failed=stats["failed"], timings=info["timings"])

    _advance(info, "translated")
    _unstale(info, "translation")
    _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks))

    # the translation check runs at once on what was translated; POST /review retries or redoes it
    before = {c["id"]: c.get("translation") for c in chunks}
    _review_step(run_id, jdir, info, chunks)
    if not clear and any(c.get("translation") != before[c["id"]] for c in chunks):  # see _review_only_step
        _mark_stale(info, chunks, "dub", "lipsync")
        _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status=info["stage"])


def _review_step(run_id: str, jdir: Path, info: dict, chunks: list[dict]):
    """The translation check. A failure to check doesn't change the translations: they stay as they are, and
    the error is in info["review"] (POST /review retries). Skipped when the "translation" is the transcript
    copied."""
    if info["target_language"] == info["language"]:
        return
    _set(run_id, status="reviewing", review_done=_review_done(chunks), review_total=_review_total(chunks))

    def report():  # called after every window
        _set(run_id, segments=_segments(chunks), review_done=_review_done(chunks))

    start = time.perf_counter()
    try:
        stats = review.review_chunks(
            jdir, chunks, info["language"], info["target_language"], progress=report, prompt=info.get("review_prompt")
        )
    except Exception as exc:  # no key, or a bug: the run is still translated
        stats = {"model": review.MODEL, "error": str(exc)}
    stats["custom_prompt"] = bool(info.get("review_prompt"))
    info["review"] = stats
    info["timings"] = {**info.get("timings", {}), "review": round(time.perf_counter() - start, 2)}
    _save_info(jdir, info)


def _review_only_step(run_id: str, jdir: Path, info: dict, chunks: list[dict], redo: bool = False):
    before = {c["id"]: c.get("translation") for c in chunks}
    if redo:  # undo every edit and forget the results, so all segments are checked again
        review.reset_chunks(chunks)
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
        _set(run_id, segments=_segments(chunks))
    _review_step(run_id, jdir, info, chunks)
    if any(c.get("translation") != before[c["id"]] for c in chunks):  # the dub and lip-sync were made from the old text
        _mark_stale(info, chunks, "dub", "lipsync")
        _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status=info["stage"])


def _dub_step(run_id: str, jdir: Path, info: dict, chunks: list[dict], voices: dict, style: str | None):
    def report():  # called once after the previous TTS results are cleared, then after every segment
        _set(run_id, segments=_segments(chunks), tts_done=_tts_done(chunks), tts_total=_tts_total(chunks))

    pending = {c["id"] for c in chunks if c.get("retranslate")}  # edited transcripts not translated again yet
    if pending:
        _retranslate(jdir, info, chunks, pending)
    start = time.perf_counter()
    stats = tts.synthesize_chunks(
        jdir, chunks, info["target_language"], voices, progress=report, style=style, reuse=True
    )
    for c in chunks:  # the whole dub is redone from speech that matches each segment's text
        if not c.get("retranslate"):
            _clear_all(c, "refresh")
        c.pop("moved_from", None)  # the lip-sync is dropped with the old dub
    info["voices"] = stats["voices"]
    info["tts_style"] = style
    info["tts_failed"] = stats["failed"]
    info["timings"] = {**info.get("timings", {}), "tts": round(time.perf_counter() - start, 2)}
    _fit_step(run_id, jdir, info, chunks)


def _fit_step(run_id: str, jdir: Path, info: dict, chunks: list[dict]):
    _set(run_id, status="dubbing")
    _clear_lipsync(jdir, info)  # it no longer matches the new dub
    start = time.perf_counter()
    dub = dubbing.build_speed_dub(jdir, chunks, jdir / info["video"], _cfg["ffmpeg_bin"])

    info["stage"] = "dubbed"
    info["dub"] = dub
    _unstale(info, "dub")
    info["timings"] = {**info.get("timings", {}), "dub": round(time.perf_counter() - start, 2)}
    _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status="dubbed")


def _clear_lipsync(jdir: Path, info: dict):
    shutil.rmtree(jdir / "lipsync", ignore_errors=True)
    (jdir / "lipsynced.mp4").unlink(missing_ok=True)
    info.pop("lipsync", None)
    _unstale(info, "lipsync")


def _lipsync_step(run_id: str, jdir: Path, info: dict, chunks: list[dict]):
    with _lipsync_slot:
        start = time.perf_counter()
        _clear_lipsync(jdir, info)
        _save_info(jdir, info)
        (jdir / "lipsync").mkdir()
        _set(run_id, status="lipsync_analysing", lipsync=None, lipsync_video_url=None)
        _lipsync_proc(run_id, jdir, "prepass")
        _set(run_id, status="lipsyncing")
        _lipsync_proc(run_id, jdir, "worker")

    info["lipsync"] = _lipsync_info(jdir)
    info["timings"] = {**info.get("timings", {}), "lipsync": round(time.perf_counter() - start, 2)}
    _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status="dubbed", lipsync_progress=None)


def _lipsync_info(jdir: Path) -> dict:
    """run.json's `lipsync`, from the pre-pass's plan.json and the worker's clips.json."""
    plan = json.loads((jdir / "lipsync" / "plan.json").read_text())
    summary = json.loads((jdir / "lipsync" / "clips.json").read_text())["summary"]
    return {
        "people": plan["people"],
        "speaker_person": plan["speaker_person"],
        **{k: summary[k] for k in ("ranges", "kept", "rejected", "skipped", "range_seconds", "kept_seconds")},
        "times": summary["times_s"],
        "peak_ram_mb": summary["peak_anon_mb"],
        "peak_vram_mb": summary["peak_vram_mb"],
    }


def _retranslate(jdir: Path, info: dict, chunks: list[dict], ids: set[int]) -> list[int]:
    """Translate again the merged chunks in `ids` (a transcript of theirs was edited). The segments whose
    translation the user wrote, and whose transcript wasn't edited since, keep it in place of their slice of the
    new translation. A chunk whose request fails keeps its old translation and its `retranslate` mark, with
    translation_error. Returns the ids that failed."""
    todo = [c for c in chunks if c["id"] in ids]
    old = {c["id"]: (c.get("translation"), [s.get("translation") for s in c.get("sub_chunks") or []]) for c in todo}
    for c in todo:
        if c.get("tts_file"):  # see review.set_translation
            c.setdefault("tts_text", c.get("translation"))
        c["translation"] = None
        for k in ("review", "review_error", "translation_original"):
            c.pop(k, None)
    try:
        translate.translate_chunks(jdir, chunks, info["language"], info["target_language"], only=ids)
    finally:
        for c in todo:
            subs = c.get("sub_chunks") or []
            translation, slices = old[c["id"]]
            if c.get("translation") is None:  # failed: the segments get their old slices back
                for s, t in zip(subs, slices):
                    s["translation"] = t
                review.set_translation(c, translation)
                continue
            # segments with their own translation, whose transcript wasn't edited since, keep it
            mine = {s["id"] for s, t in zip(subs, slices) if s.get("translation_edited") and not s.get("retranslate") and t}
            if mine:
                for s, t in zip(subs, slices):
                    if s["id"] in mine:
                        s["translation"] = t
                review.set_translation(c, " ".join(s["translation"] for s in subs if s.get("translation")))
            _clear_all(c, "retranslate")
            for s in subs:
                if s["id"] not in mine:
                    s.pop("translation_edited", None)
            if not mine:
                c.pop("translation_edited", None)
        (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return [c["id"] for c in todo if c.get("retranslate")]


def _refresh_step(run_id: str, jdir: Path, info: dict, chunks: list[dict], seg_id: int):
    """One edited segment, the rest of the run untouched: its translation (if its transcript was edited); in a
    dubbed run, its speech (the run's voices and style), its fit (with the dub's speed floor) spliced into the dub
    (dubbing.refit_segments; a merged block from an older dub it was part of is taken apart, each of its segments
    fitted alone);
    if the run was lip-synced, the pre-pass again (it reuses faces.json) and the worker on the ranges where
    the dub changed (--update), keeping the other clips. A failure leaves the segment marked `refresh`. `seg_id`
    is the short segment the user refreshed; its merged chunk is what is translated and dubbed again."""
    c, _ = _find(chunks, seg_id)
    cid = c["id"]
    start = time.perf_counter()

    def step(name):
        _set(run_id, refresh={"segment": seg_id, "step": name}, segments=_segments(chunks))

    if c.get("retranslate"):
        step("translating")
        if _retranslate(jdir, info, chunks, {cid}):
            raise RuntimeError(f"translation failed: {c.get('translation_error')}")

    if info["stage"] == "dubbed":
        step("synthesizing")
        voices = {int(s): v for s, v in (info.get("voices") or {}).items()}
        stats = tts.synthesize_chunks(
            jdir, chunks, info["target_language"], voices, style=info.get("tts_style"), reuse=True, only={cid}
        )
        info["tts_failed"] = stats["failed"]
        if not c.get("tts_file"):
            raise RuntimeError(f"speech failed: {c.get('tts_error')}")

        step("dubbing")
        floor = (info.get("dub") or {}).get("speed_floor") or 1.0
        fit = dubbing.refit_segments(jdir, chunks, {cid}, jdir / info["video"], _cfg["ffmpeg_bin"], floor)
        info["dub"] = {**info["dub"], **fit["summary"]}
        info["refreshes"] = info.get("refreshes", 0) + 1  # the dub's files changed (the UI's cache key)
        _save_info(jdir, info)
        _set(run_id, **_record(run_id, info, chunks))

        if info.get("lipsync"):  # until this is done, the segment's `refresh` mark says its lip-sync is old
            (jdir / "lipsync" / "progress.json").unlink(missing_ok=True)
            step("lipsync_queued")
            with _lipsync_slot:
                step("lipsync_analysing")
                _lipsync_proc(run_id, jdir, "prepass")
                step("lipsyncing")
                t0, t1 = fit["span"]
                spk = sorted({*fit["speakers"], *c.get("moved_from", [])})  # a moved segment's old clips go too
                _lipsync_proc(
                    run_id, jdir, "worker", "--update", f"{t0},{t1}", "--speakers", ",".join(map(str, spk))
                )
            info["lipsync"] = _lipsync_info(jdir)
            info["refreshes"] += 1

    _clear_all(c, "refresh")
    c.pop("moved_from", None)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    info["timings"] = {**info.get("timings", {}), "refresh": round(time.perf_counter() - start, 2)}
    _save_info(jdir, info)
    _set(run_id, **_record(run_id, info, chunks), status=info["stage"], refresh=None, lipsync_progress=None)


def _lipsync_proc(run_id: str, jdir: Path, step: str, *args: str, log: str | None = None):
    """Run one lip-sync step (lipsync/run.sh) to the end, publishing the worker's progress.json every 2 s.
    `log`: its log, relative to the run folder (default lipsync/<step>.log)."""
    log = log or f"lipsync/{step}.log"
    progress = jdir / "lipsync" / "progress.json"
    with open(jdir / log, "w") as f:
        proc = subprocess.Popen([str(LIPSYNC / "run.sh"), step, str(jdir), *args], stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            time.sleep(2)
            if progress.exists():
                try:
                    _set(run_id, lipsync_progress=json.loads(progress.read_text()))
                except (OSError, ValueError):
                    pass  # being replaced; the next read gets it
    if proc.returncode:
        text = (jdir / log).read_text(errors="replace")
        if "signal 9" in text:
            reason = "killed: it went over the lip-sync RAM cap"
        else:  # the last line that isn't /usr/bin/time's report
            lines = [l for l in text.splitlines() if l.strip() and not l.startswith("\t")]
            reason = lines[-1] if lines else f"exit code {proc.returncode}"
        raise RuntimeError(f"lip-sync {step} failed: {reason} (see {log})")
