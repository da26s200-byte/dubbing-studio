import json
import math
import subprocess
import wave
from pathlib import Path

import numpy as np
from scipy.ndimage import minimum_filter1d

DUB_SR = 24000  # TTS output rate; clips are fitted and laid out at this rate
OUT_SR = 44100  # final mix rate, the same as audio.wav / bg_noise.wav
GAIN_RANGE = (0.25, 4.0)  # loudness match to the original speaker is clamped to this
GAP_GAIN = 1.0  # the original vocals (laughter, cheers, reactions) outside every segment are mixed back at this gain; 0 = off
GAP_GUARD = 0.05  # s kept clear of each segment's span, so the original speech at its edges doesn't leak in
GAP_FADE = 0.05  # s: the mixed-back vocals fade in and out over this at the edges of a gap (< 2 * GAP_GUARD)
LIMIT_CEILING = 0.97  # the final mix never peaks above this ...
LIMIT_HOLD_S = 0.02  # ... its gain drops over this window around each peak (look-ahead) ...
LIMIT_SMOOTH_S = 0.01  # ... and is smoothed over this, so only the samples near a peak get quieter (< LIMIT_HOLD_S)


def _gap_mask(chunks, n: int) -> np.ndarray:
    """1 where no segment is (outside every chunk's cut_start..cut_end, widened by GAP_GUARD), else 0, with the
    edges faded over GAP_FADE; at OUT_SR, n samples. The span counts even where the dub ended early, so the
    original speech never plays in the silence a short clip leaves."""
    mask = np.ones(n, dtype=np.float32)
    for c in chunks:
        a = int(max(c.get("cut_start", c["start"]) - GAP_GUARD, 0) * OUT_SR)
        b = int((c.get("cut_end", c["end"]) + GAP_GUARD) * OUT_SR)
        mask[a:b] = 0.0
    w = max(int(GAP_FADE * OUT_SR), 1)
    csum = np.concatenate([[0.0], np.cumsum(mask, dtype=np.float64)])  # moving average over w samples
    lo, hi = np.clip(np.arange(n) - w // 2, 0, n), np.clip(np.arange(n) + w - w // 2, 0, n)
    return ((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def _limit(mix: np.ndarray) -> np.ndarray:
    """Look-ahead peak limiter on a (n, 2) mix at OUT_SR: the gain each sample needs to stay under LIMIT_CEILING,
    min-filtered over LIMIT_HOLD_S and averaged over LIMIT_SMOOTH_S. The average only sees values within the
    hold window, so it never exceeds the gain a peak needs; away from peaks the gain is 1."""
    if not len(mix):
        return mix
    need = np.minimum(1.0, LIMIT_CEILING / np.maximum(np.abs(mix).max(axis=1), 1e-9)).astype(np.float32)
    if need.min() >= 1.0:
        return mix
    gain = minimum_filter1d(need, size=max(int(LIMIT_HOLD_S * OUT_SR), 1), mode="nearest")
    w = max(int(LIMIT_SMOOTH_S * OUT_SR), 1)
    n = len(gain)
    csum = np.concatenate([[0.0], np.cumsum(gain, dtype=np.float64)])  # moving average over w samples
    lo, hi = np.clip(np.arange(n) - w // 2, 0, n), np.clip(np.arange(n) + w - w // 2, 0, n)
    gain = ((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)
    return mix * gain[:, None]


def _mix_and_mux(jdir: Path, out_dir: Path, track, video_path: Path | None, ffmpeg_bin: str, chunks: list[dict]):
    """24 kHz mono dub track -> 44.1 kHz stereo, mixed with bg_noise.wav and, outside every segment, the original
    vocals (laughter and reactions the speech-only dub would drop; GAP_GAIN): dubbed_vocals.wav (the dub alone),
    dubbed_audio.wav and, with a video, dubbed.mp4 with the translated (default) and the original audio track."""
    dub = _ffmpeg_pcm(ffmpeg_bin, track, DUB_SR, OUT_SR)
    dub = np.repeat(dub[:, None], 2, axis=1)
    bg = _decode(ffmpeg_bin, jdir / "bg_noise.wav", OUT_SR, 2)
    n = max(len(dub), len(bg))
    dub = np.pad(dub, ((0, n - len(dub)), (0, 0)))
    bg = np.pad(bg, ((0, n - len(bg)), (0, 0)))
    mix = dub + bg
    if GAP_GAIN:
        vocals = _decode(ffmpeg_bin, jdir / "vocals.wav", OUT_SR, 2)[:n]
        mix[: len(vocals)] += GAP_GAIN * vocals * _gap_mask(chunks, n)[: len(vocals), None]
    mix = _limit(mix)

    _write_wav(out_dir / "dubbed_vocals.wav", dub, OUT_SR)
    _write_wav(out_dir / "dubbed_audio.wav", mix, OUT_SR)

    if video_path:
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-v", "error",
                "-i", str(video_path), "-i", str(out_dir / "dubbed_audio.wav"),
                # the dub is track 0 and the default, so players that only play one track play the dub
                "-map", "0:v:0", "-map", "1:a:0", "-map", "0:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-metadata:s:a:0", "title=Translated", "-metadata:s:a:1", "title=Original",
                "-disposition:a:0", "default", "-disposition:a:1", "0",
                "-movflags", "+faststart",
                str(out_dir / "dubbed.mp4"),
            ],
            check=True,
            capture_output=True,
        )


# The studio flow's dub (build_speed_dub): every segment fitted on its own with cadence
MAX_SPEED = 1.65  # no segment is sped up beyond this; what still doesn't fit runs past its slot
SPEED_FLOOR = 1.1  # if any segment needs more than SPEED_FLOOR_FROM, every segment is sped up to at least this ...
SPEED_FLOOR_FROM = 1.35  # ... (its pauses are then stretched to fill what it no longer needs)
STRETCH_MAX = 1.5  # a stretched pause never grows beyond this
# cadence: pauses aligned to where the original speaker paused (frames of PAUSE_FRAME, level PAUSE_DB)
PAUSE_DB = -35.0  # TTS frames this far below the clip's 95th-percentile level are silent
PAUSE_FRAME = 0.01  # RMS frame for pause detection in TTS audio
EDGE_PAD_S = 0.02  # TTS edge silence kept when trimming
MIN_SPEECH_BRIDGE_S = 0.05  # silences shorter than this inside speech (plosives) count as speech
MIN_NOISE_SPIKE_S = 0.04  # speech shorter than this between silences counts as silence
ORIG_PAUSE_MIN_S = 0.20  # silences in the original at least this long are pauses to align to
TTS_PAUSE_CANDIDATE_S = 0.20  # silences in the TTS at least this long are pauses that can be aligned
UNMATCHED_TTS_PAUSE_THRESH_S = 0.25  # a TTS pause the original doesn't have, longer than this, ...
UNMATCHED_TTS_PAUSE_FLOOR_S = 0.15  # ... shrinks to this when its segment needs a speed-up
PROGRESS_TOLERANCE = 0.12  # a TTS and an original pause pair up when their share of speech before them differs by at most this
LOCAL_SPEED_MIN_RATIO = 0.85  # between aligned pauses, the speed a stretch needs stays within these ratios ...
LOCAL_SPEED_MAX_RATIO = 1.20  # ... of the segment's speed
ONSET_MAX_S = 0.40  # at most this much of the original's silence before its first word is kept before the dub
UNMATCHED_STRETCH_WEIGHT = 0.05  # leftover time barely stretches unmatched pauses ...
UNMATCHED_STRETCH_MAX_S = 0.30  # ... and never beyond this
MATCHED_PAUSE_SLACK_S = 0.30  # a matched pause grows to at most max(its length, the original's) + this (and STRETCH_MAX)
MATCH_MIN_RATIO = 0.4  # a TTS pause (not a boundary) pairs with an original pause only if at least this share of its length
GROWTH_SLACK_S = 0.30  # a matched pause (not a boundary) also grows to at most max(its length + this, ...
GROWTH_RATIO = 2.0  # ... this times its length), so a short gap between words isn't stretched into a full stop
SLOW_ORIG_PAUSE_S = 1.0  # in a slow segment (c["slow"]) that fits its slot, only original pauses at least this long (asr.SLOW_PAUSE_S) get the slow rules:
SLOW_MATCH_REF_MAX_S = 0.75  # ... _pairable caps o["dur"] at this when checking MATCH_MIN_RATIO, so a 0.3 s TTS comma pause can pair with it ...
SLOW_PAIR_TOLERANCE = 0.08  # ... if within this in speech progress (tighter than PROGRESS_TOLERANCE, as the length check is looser) ...
SLOW_STRETCH_MAX = 3.0  # ... and a matched pause may grow to the original's + slack, capped at this (no GROWTH_RATIO cap);
# all a slow segment's pauses together grow by at most the original's total pause time in it
SLOW_UNMATCHED_STRETCH_WEIGHT = 0.25  # in a slow segment, unmatched TTS pauses in the tail absorb more leftover time ...
SLOW_UNMATCHED_STRETCH_MAX_S = 1.20  # ... up to this cap
SPEED_PENALTY = 0.5  # alignment score lost per 1x between a stretch's speed and the segment's
FIT_DESCRIPTION = (
    f"Each segment on its own: its TTS clip's edge silence trimmed, matched in loudness to the original speaker, "
    f"sped up just enough to fit its slot (never slowed down, at most {MAX_SPEED}x; at least {SPEED_FLOOR}x when any "
    f"segment needs more than {SPEED_FLOOR_FROM}x), its pauses paired with the speaker's by speech progress and "
    "stretched so speech resumes where theirs does, pauses the speaker doesn't have shortened when it needs a speed-up."
)


def build_speed_dub(jdir: Path, chunks: list[dict], video_path: Path | None, ffmpeg_bin: str) -> dict:
    """Speed-up only (studio flow), FIT_DESCRIPTION: every chunk's TTS clip (tts/<id>.wav) prepared (_fit_prepare),
    the speed floor set from the fastest chunk, then each fitted alone (_fit_one), written to dub/<id>.wav,
    placed at its start and mixed with bg_noise. Sets chunk["dub"] = {slot, tts_duration, trimmed, needed_speed,
    vanilla_speed, speed, duration, gain, unit, unit_size, matched_pauses, compressed, stretched, restored, end_gap,
    onset} or chunk["dub_error"]. Makes no API requests."""
    dub_dir = jdir / "dub"
    dub_dir.mkdir(exist_ok=True)
    for old in dub_dir.glob("*.wav"):  # don't leave clips of an earlier fit
        old.unlink()

    orig, orig_sr = _read_wav(jdir / "vocals_16k.wav")
    track = np.zeros(int(math.ceil(len(orig) / orig_sr * DUB_SR)) + 1, dtype=np.float32)

    prepared = {}
    for c in chunks:
        c.pop("dub", None)
        c.pop("dub_error", None)
        if not c.get("tts_file"):
            continue
        try:
            prepared[c["id"]] = _fit_prepare(jdir, c, orig, orig_sr, ffmpeg_bin)
        except Exception as exc:
            c["dub_error"] = str(exc)

    fastest = min(max((f["plan"]["needed"] for f in prepared.values()), default=1.0), MAX_SPEED)
    floor = SPEED_FLOOR if fastest > SPEED_FLOOR_FROM else 1.0
    for c in chunks:
        if c["id"] not in prepared:
            continue
        try:
            y, c["dub"] = _fit_one(c, prepared[c["id"]], floor, ffmpeg_bin)
        except Exception as exc:
            c["dub_error"] = str(exc)
            continue
        _write_wav(dub_dir / f"{c['id']:04d}.wav", y, DUB_SR)
        at = min(int(round(float(c["start"]) * DUB_SR)), len(track))
        n = min(len(y), len(track) - at)
        track[at : at + n] += y[:n]

    _mix_and_mux(jdir, jdir, track, video_path, ffmpeg_bin, chunks)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return {**_dub_summary(chunks), "speed_floor": floor}


def _fit_prepare(jdir: Path, c: dict, orig, orig_sr, ffmpeg_bin: str) -> dict:
    """A chunk's TTS clip ready to fit: at DUB_SR, its edge silence trimmed to EDGE_PAD_S at whole frames,
    matched in loudness to the original speaker in its slot (gain within GAIN_RANGE), and its cadence plan.
    Raises if the clip can't be read, is empty or has no speech."""
    x, sr = _read_wav(jdir / c["tts_file"])
    if sr != DUB_SR:
        x = _ffmpeg_pcm(ffmpeg_bin, x, sr, DUB_SR)
    if not len(x):
        raise ValueError("TTS audio is empty")
    hop = int(round(DUB_SR * PAUSE_FRAME))
    mask = _voiced(x, DUB_SR)
    idx = np.flatnonzero(mask)
    if not len(idx):
        raise ValueError("TTS audio is silent")
    pad = _frames(EDGE_PAD_S)
    a, b = max(idx[0] - pad, 0), min(idx[-1] + 1 + pad, len(mask))
    trimmed = (len(x) - (b - a) * hop) / DUB_SR
    x, mask = x[a * hop : b * hop], mask[a:b]
    gain = 1.0
    if _level(x, DUB_SR) > 0:
        ref = _level(orig[int(c["start"] * orig_sr) : int(c["end"] * orig_sr)], orig_sr)
        gain = float(np.clip(ref / _level(x, DUB_SR), *GAIN_RANGE))
    x = (x * gain).astype(np.float32)
    plan = _cadence_plan([c], {c["id"]: {"x": x, "mask": mask}}, orig, orig_sr)
    return {"plan": plan, "gain": gain, "trimmed": trimmed, "length": len(x) / DUB_SR}


def _fit_one(c: dict, prep: dict, floor: float, ffmpeg_bin: str):
    """A prepared chunk fitted alone: sped up just enough to fit its slot, at least `floor`, at most MAX_SPEED,
    its pauses aligned (_cadence_fill). A slow chunk (c["slow"]) that already fits isn't held to the floor.
    Returns (audio, its "dub" record)."""
    plan = prep["plan"]
    if c.get("slow") and plan["needed"] <= 1.0:
        floor = 1.0
    speed = min(max(plan["needed"], floor), MAX_SPEED)
    y, segs, info = _cadence_fill(plan["block"], plan, speed, floor, ffmpeg_bin)
    return y, {
        "slot": round(float(c["end"] - c["start"]), 3),
        "tts_duration": round(prep["length"], 3),
        "trimmed": round(prep["trimmed"], 3),
        "needed_speed": round(plan["needed"], 3),
        "vanilla_speed": round(plan["needed"], 3),
        "speed": round(speed, 3),
        "duration": round(len(y) / DUB_SR, 3),
        "gain": round(prep["gain"], 3),
        "unit": c["id"],
        "unit_size": 1,
        "matched_pauses": segs[0]["matched"],
        "compressed": round(plan["compressed"][0], 3),
        **info,
    }


def _dub_summary(chunks: list[dict]) -> dict:
    dubs = [c["dub"] for c in chunks if c.get("dub")]
    speeds = sorted(d["speed"] for d in dubs)
    return {
        "fitted": len(speeds),
        "failed": sum(1 for c in chunks if c.get("dub_error")),
        "sped_up": sum(1 for v in speeds if v > 1.0),
        "min_speed": speeds[0] if speeds else None,
        "median_speed": float(np.median(speeds)) if speeds else None,
        "max_speed": speeds[-1] if speeds else None,
        "matched_pauses": sum(d.get("matched_pauses", 0) for d in dubs),
        "slow_fitted": sum(1 for d in dubs if d.get("slow")),
        "relaxed_pairs": sum(d.get("relaxed_pairs", 0) for d in dubs),
        "max_pause_added": max((d.get("max_pause_added", 0) for d in dubs), default=0),
    }


def refit_segments(
    jdir: Path, chunks: list[dict], ids, video_path: Path | None, ffmpeg_bin: str, floor: float = 1.0
) -> dict:
    """After a segment was edited (studio): the chunks in `ids` are fitted again as build_speed_dub does, with the
    dub's speed `floor` (a block of several segments from a dub made before every segment was fitted alone is
    taken apart, each of its segments fitted). The rest of the dub stays as it is: the track is rebuilt from
    dub/*.wav and mixed again. Returns {ids (the segments refitted), span ([t0, t1], where the dub changed: the
    old and the new placement), speakers, summary (the dub summary's counts, redone)}."""
    dub_dir = jdir / "dub"
    by_id = {c["id"]: c for c in chunks}
    units = {by_id[i]["dub"]["unit"] for i in ids if by_id[i].get("dub")}
    redo = [c for c in chunks if c["id"] in ids or (c.get("dub") and c["dub"]["unit"] in units)]
    spans = []
    for c in redo:
        spans.append((c["start"], c["end"]))
        d = c.get("dub")
        if d:  # where its old dub played, as in lipsync/prepass.py's speaking_ranges
            u0 = by_id[d["unit"]]["start"]
            ps = u0 + (c["start"] - u0) / d["speed"]
            spans.append((ps, ps + d["duration"]))
    for u in units:
        (dub_dir / f"{u:04d}.wav").unlink(missing_ok=True)

    orig, orig_sr = _read_wav(jdir / "vocals_16k.wav")
    for c in redo:
        c.pop("dub", None)
        c.pop("dub_error", None)
        if not c.get("tts_file"):
            continue
        try:
            y, dub = _fit_one(c, _fit_prepare(jdir, c, orig, orig_sr, ffmpeg_bin), floor, ffmpeg_bin)
        except Exception as exc:
            c["dub_error"] = str(exc)
            continue
        _write_wav(dub_dir / f"{c['id']:04d}.wav", y, DUB_SR)
        c["dub"] = dub
        spans.append((c["start"], c["start"] + dub["duration"]))

    # the track again: every unit's audio at its first segment's start
    track = np.zeros(int(math.ceil(len(orig) / orig_sr * DUB_SR)) + 1, dtype=np.float32)
    for c in chunks:
        if not c.get("dub") or c["dub"]["unit"] != c["id"]:
            continue
        y, _ = _read_wav(dub_dir / f"{c['id']:04d}.wav")
        at = min(int(round(float(c["start"]) * DUB_SR)), len(track))
        n = min(len(y), len(track) - at)
        track[at : at + n] += y[:n]
    _mix_and_mux(jdir, jdir, track, video_path, ffmpeg_bin, chunks)
    (jdir / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    return {
        "ids": [c["id"] for c in redo],
        "span": [round(min(a for a, _ in spans), 3), round(max(b for _, b in spans), 3)],
        "speakers": sorted({c["speaker"] for c in redo}),
        "summary": _dub_summary(chunks),
    }


def _frames(seconds):
    """Whole PAUSE_FRAME frames in `seconds`."""
    return int(round(seconds / PAUSE_FRAME))


def _voiced(x, sr):
    """Cadence: per PAUSE_FRAME frame, True where it's speech: RMS at least the 95th-percentile frame's plus
    PAUSE_DB. Silences shorter than MIN_SPEECH_BRIDGE_S inside speech count as speech (plosives), then
    speech shorter than MIN_NOISE_SPIKE_S counts as silence (noise)."""
    hop = int(round(sr * PAUSE_FRAME))
    n = len(x) // hop
    if not n:
        return np.zeros(0, dtype=bool)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1))
    top = float(np.percentile(rms, 95))
    voiced = rms >= top * 10 ** (PAUSE_DB / 20) if top > 0 else np.zeros(n, dtype=bool)
    for a, b in list(_runs(~voiced)):
        if a > 0 and b < n and b - a < _frames(MIN_SPEECH_BRIDGE_S):
            voiced[a:b] = True
    for a, b in list(_runs(voiced)):
        if b - a < _frames(MIN_NOISE_SPIKE_S):
            voiced[a:b] = False
    return voiced


def _cadence_pauses(voiced, min_s, bounds):
    """Cadence: the pauses of a unit's frame mask, in order, as {a, b (frames), dur, p, k, y}: one per boundary
    between its segments (`bounds`: (start, end) frames of each gap, widened over the silence around it; k =
    the boundary's index), then every other silent run of at least min_s not at its edges (k None). p = the
    share of the unit's speech before the pause; y = the duration of the pause it pairs with (_cadence_match)."""
    n = len(voiced)
    before = np.concatenate([[0], np.cumsum(voiced)])
    total = max(int(before[-1]), 1)
    taken = np.zeros(n, dtype=bool)
    spans = []
    for k, (a, b) in enumerate(bounds):
        while a > 0 and not voiced[a - 1]:
            a -= 1
        while b < n and not voiced[b]:
            b += 1
        taken[a:b] = True
        spans.append((a, b, k))
    spans += [
        (a, b, None)
        for a, b in _runs(~voiced)
        if a > 0 and b < n and b - a >= _frames(min_s) and not taken[a:b].any()
    ]
    return sorted(
        (
            {"a": int(a), "b": int(b), "dur": (b - a) * PAUSE_FRAME, "p": before[a] / total, "k": k, "y": None}
            for a, b, k in spans
        ),
        key=lambda q: q["a"],
    )


def _cadence_match(tpauses, opauses, tol=PROGRESS_TOLERANCE, slow=False):
    """Sets each TTS pause's y: for a boundary, the original's pause at the same boundary; for any other, the
    original pause closest in speech progress, if within `tol` and of similar length (_pairable; else None:
    unmatched)."""
    for q in tpauses:
        if q["k"] is not None:
            q["y"] = next((o["dur"] for o in opauses if o["k"] == q["k"]), None)
            continue
        near = [o for o in opauses if o["k"] is None and _pairable(q, o, tol, slow)]
        q["y"] = min(near, key=lambda o: _progress_diff(q, o))["dur"] if near else None
    return tpauses


def _cadence_plan(u, clips, orig, orig_sr):
    """Cadence, before the speed-up. A unit's trimmed clips (gain applied) joined with the original gaps between
    them as silence, at whole frames; its pauses and the original speaker's over the unit's span, both with
    their speech progress; and, when the unit is longer than its span, every unmatched TTS pause longer than
    UNMATCHED_TTS_PAUSE_THRESH_S shrunk to UNMATCHED_TTS_PAUSE_FLOOR_S (silence taken from its middle). Returns
    {block, voiced, owner (segment index per frame, -1 in a gap), tpauses, opauses, span, lead (the original's silence before its first word),
    needed (speed), compressed (seconds, per segment), tol (the progress window), slow (a segment is slow and the
    unit fits its span without a speed-up: the slow pairing / stretch rules)}."""
    hop = int(round(DUB_SR * PAUSE_FRAME))
    t0 = float(u[0]["start"])
    span = float(u[-1]["end"]) - t0

    # the TTS side, frame by frame
    frames, voiced, owner, bounds, pos = [], [], [], [], 0
    for i, c in enumerate(u):
        if i:
            n = _frames(max(float(c["start"] - u[i - 1]["end"]), 0.0))
            frames.append(np.zeros((n, hop), dtype=np.float32))
            voiced.append(np.zeros(n, dtype=bool))
            owner.append(np.full(n, -1))
            bounds.append((pos, pos + n))
            pos += n
        clip = clips[c["id"]]
        frames.append(clip["x"].reshape(-1, hop))
        voiced.append(clip["mask"])
        owner.append(np.full(len(clip["mask"]), i))
        pos += len(clip["mask"])
    frames, voiced, owner = np.concatenate(frames), np.concatenate(voiced), np.concatenate(owner)

    # the original speaker over the same span; nothing of theirs is spoken between their segments
    ov = _voiced(orig[int(t0 * orig_sr) : int((t0 + span) * orig_sr)], orig_sr)
    obounds = []
    for i in range(1, len(u)):
        a = min(_frames(float(u[i - 1]["end"]) - t0), len(ov))
        b = min(max(_frames(float(u[i]["start"]) - t0), a), len(ov))
        ov[a:b] = False
        obounds.append((a, b))
    lead = float(np.argmax(ov)) * PAUSE_FRAME if ov.any() else 0.0
    opauses = _cadence_pauses(ov, ORIG_PAUSE_MIN_S, obounds)
    slow = any(c.get("slow") for c in u) and len(voiced) * PAUSE_FRAME <= span
    tol = PROGRESS_TOLERANCE
    tpauses = _cadence_match(_cadence_pauses(voiced, TTS_PAUSE_CANDIDATE_S, bounds), opauses, tol, slow)

    compressed = [0.0] * len(u)
    if len(voiced) * PAUSE_FRAME > span:  # it needs a speed-up: shrink the pauses the original doesn't have
        keep = np.ones(len(voiced), dtype=bool)
        floor_n = _frames(UNMATCHED_TTS_PAUSE_FLOOR_S)
        for q in tpauses:
            if q["k"] is None and q["y"] is None and q["b"] - q["a"] > _frames(UNMATCHED_TTS_PAUSE_THRESH_S):
                cut = q["b"] - q["a"] - floor_n
                keep[q["a"] + floor_n // 2 : q["a"] + floor_n // 2 + cut] = False
                compressed[owner[q["a"]]] += cut * PAUSE_FRAME
        if not keep.all():
            idx = np.concatenate([[0], np.cumsum(keep)])
            bounds = [(int(idx[a]), int(idx[b])) for a, b in bounds]
            frames, voiced, owner = frames[keep], voiced[keep], owner[keep]
            tpauses = _cadence_match(_cadence_pauses(voiced, TTS_PAUSE_CANDIDATE_S, bounds), opauses, tol, slow)

    return {
        "block": frames.ravel(),
        "voiced": voiced,
        "owner": owner,
        "tpauses": tpauses,
        "opauses": opauses,
        "span": span,
        "lead": lead,
        "needed": _fit_speed(len(voiced) * PAUSE_FRAME, span),
        "compressed": compressed,
        "tol": tol,
        "slow": slow,
    }


def _cadence_fill(block, plan, speed, floor, ffmpeg_bin, max_speed=MAX_SPEED):
    """Cadence: the plan's block sped up by `speed`, then its pauses aligned.
    1. Onset: up to ONSET_MAX_S of the original's silence before its first word, if the unit has room for it.
    2. Anchors: a monotonic sequence of (TTS pause, original pause) pairs, chosen by dynamic programming to
       maximise sum(original pause length * (1 - |progress difference| / the plan's tol)) minus
       SPEED_PENALTY * |stretch speed - speed| per stretch between anchors. A stretch is only allowed if the speed
       it needs is at most max(max_speed, the unit's) and its speed (at least 1 and the floor, at most max_speed) is within LOCAL_SPEED_MIN/MAX_RATIO of `speed`. Each anchor's pause grows (never shrinks, at
       most to _matched_cap) so the speech after it starts where the original's does.
    3. Time still missing at the end goes to the pauses after the last anchor: boundary gaps back toward the
       original's, matched pauses toward theirs (at most STRETCH_MAX), then by k * w (k solved by bisection):
       w = max(x, y)^2 up to _matched_cap for matched pauses, UNMATCHED_STRETCH_WEIGHT * x^2 up to
       UNMATCHED_STRETCH_MAX_S for unmatched ones (SLOW_UNMATCHED_STRETCH_WEIGHT / _MAX_S in a slow plan). What none of them can take stays as silence at the end.
    Silence goes in the middle of each pause. Returns (audio, per segment {duration, matched}, {stretched,
    restored, end_gap, onset})."""
    sr, hop = DUB_SR, int(round(DUB_SR * PAUSE_FRAME))
    tp, op, span, tol = plan["tpauses"], plan["opauses"], plan["span"], plan["tol"]
    slow = plan.get("slow", False)
    budget = int(round(sum(o["dur"] for o in op) * sr)) if slow else None  # samples the pauses may grow by, in all
    un_w = SLOW_UNMATCHED_STRETCH_WEIGHT if slow else UNMATCHED_STRETCH_WEIGHT
    un_max = SLOW_UNMATCHED_STRETCH_MAX_S if slow else UNMATCHED_STRETCH_MAX_S
    total = len(plan["voiced"]) * PAUSE_FRAME
    n_block = len(block)
    y = _ffmpeg_pcm(ffmpeg_bin, block, sr, sr, _atempo_chain(speed)) if speed > 1.0 else block
    est = len(y) / sr
    scale = len(y) / n_block if n_block else 1.0

    def at(f):  # a frame of the block -> its sample in y
        return int(round(f * hop * scale))

    lead_n = int(round(min(plan["lead"], ONSET_MAX_S, max(span - est, 0.0)) * sr))
    lead = lead_n / sr

    def cost(t0, t1, o0, o1):  # TTS t0..t1 played in the original's o0..o1: speed penalty, None if not allowed
        if o1 - o0 <= 0:
            return None
        need = (t1 - t0) / (o1 - o0)
        v = min(max_speed, max(1.0, floor, need))
        if need > max(max_speed, plan["needed"]):
            return None
        if not LOCAL_SPEED_MIN_RATIO * speed <= v <= LOCAL_SPEED_MAX_RATIO * speed:
            return None
        return SPEED_PENALTY * abs(v - speed)

    def gain(q, o):
        if q["k"] is not None:
            return o["dur"]
        return o["dur"] * (1 - _progress_diff(q, o) / tol)

    # 2. anchors (candidates in order of TTS pause, then original pause, so predecessors come first)
    pf = PAUSE_FRAME
    cands = [
        (i, j)
        for i, q in enumerate(tp)
        for j, o in enumerate(op)
        if (q["k"] is not None and q["k"] == o["k"])
        or (q["k"] is None and o["k"] is None and _pairable(q, o, tol, slow))
    ]
    best = {}  # (i, j) -> (score of the best sequence ending there, previous anchor)
    for i, j in cands:
        q, o = tp[i], op[j]
        opts = []
        c = cost(0.0, q["a"] * pf, lead, o["a"] * pf)
        if c is not None:
            opts.append((gain(q, o) - c, None))
        for (pi, pj), (s, _) in best.items():
            if pi < i and pj < j:
                c = cost(tp[pi]["b"] * pf, q["a"] * pf, op[pj]["b"] * pf, o["a"] * pf)
                if c is not None:
                    opts.append((s + gain(q, o) - c, (pi, pj)))
        if opts:
            best[(i, j)] = max(opts, key=lambda t: t[0])
    score, last = 0.0, None  # no anchors: one stretch at the unit's speed
    for (i, j), (s, _) in best.items():
        c = cost(tp[i]["b"] * pf, total, op[j]["b"] * pf, span)
        if c is not None and s - c > score:
            score, last = s - c, (i, j)
    anchors = []
    while last is not None:
        anchors.append(last)
        last = best[last][1]
    anchors.reverse()

    # each anchor's pause grown so the speech after it starts where the original's does
    inserts, added, restored = [], 0, 0
    for i, j in anchors:
        ps, pe = at(tp[i]["a"]), at(tp[i]["b"])
        cur = (pe - ps) / sr
        cap = _matched_cap(cur, op[j]["dur"], tp[i]["k"] is not None, slow)
        want = min(max(op[j]["b"] * pf - (lead_n + ps + added) / sr, cur), max(cur, cap))
        n = int(round((want - cur) * sr))
        if budget is not None:
            n = min(n, budget - added)
        if n > 0:
            inserts.append(((ps + pe) // 2, n))
            added += n
            restored += n if tp[i]["k"] is not None else 0

    # 3. what's still missing goes to the pauses after the last anchor
    missing = span - (lead_n + len(y) + added) / sr
    if budget is not None:
        missing = min(missing, (budget - added) / sr)
    tail = list(range(anchors[-1][0] + 1 if anchors else 0, len(tp)))
    if missing > PAUSE_FRAME and tail:
        cur = {i: (at(tp[i]["b"]) - at(tp[i]["a"])) / sr for i in tail}
        add = dict.fromkeys(tail, 0.0)

        def give(want):  # up to want[i] seconds to each pause, scaled down to what's missing
            nonlocal missing
            need = sum(want.values())
            if need > 0 and missing > 0:
                s = min(1.0, missing / need)
                for i, w in want.items():
                    add[i] += w * s
                missing -= need * s

        matched = [i for i in tail if tp[i]["y"] is not None]
        cap = {i: _matched_cap(cur[i], tp[i]["y"], tp[i]["k"] is not None, slow) for i in matched}
        give({i: max(min(tp[i]["y"], cap[i]) - cur[i], 0.0) for i in matched if tp[i]["k"] is not None})
        give({i: max(min(tp[i]["y"], cap[i]) - cur[i], 0.0) for i in matched if tp[i]["k"] is None})
        w = {
            i: max(cur[i], tp[i]["y"]) ** 2 if tp[i]["y"] is not None else un_w * cur[i] ** 2
            for i in tail
        }
        room = {
            i: max(
                (cap[i] if tp[i]["y"] is not None else un_max)
                - cur[i] - add[i],
                0.0,
            )
            for i in tail
        }

        def grown(k):
            return sum(min(k * w[i], room[i]) for i in tail)

        if missing > PAUSE_FRAME and sum(room.values()) > 0:
            if grown(1e9) <= missing:  # every pause at its cap
                k = 1e9
            else:
                lo, hi = 0.0, 1.0
                while grown(hi) < missing:
                    hi *= 2
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if grown(mid) < missing else (lo, mid)
                k = hi
            for i in tail:
                add[i] += min(k * w[i], room[i])
        for i in tail:
            n = int(round(add[i] * sr))
            if n > 0:
                inserts.append(((at(tp[i]["a"]) + at(tp[i]["b"])) // 2, n))
                restored += n if tp[i]["k"] is not None else 0

    inserts.sort()
    out = np.concatenate([np.zeros(lead_n, dtype=np.float32), _insert_silence(y, inserts)])

    owner, segs = plan["owner"], []
    for s in range(int(owner.max()) + 1):
        f = np.flatnonzero(owner == s)
        s0, s1 = at(f[0]), at(f[-1] + 1)
        segs.append({
            "duration": (s1 - s0 + sum(n for p, n in inserts if s0 < p < s1)) / sr,
            "matched": sum(1 for i, _ in anchors if tp[i]["k"] is None and owner[tp[i]["a"]] == s),
        })
    info = {
        "slow": slow,
        "relaxed_pairs": sum(1 for i, j in anchors if tp[i]["k"] is None and _relaxed(tp[i], op[j], tol)),
        "max_pause_added": round(max((n for _, n in inserts), default=0) / sr, 3),
        "stretched": round(sum(n for _, n in inserts) / sr, 3),
        "restored": round(restored / sr, 3),
        "end_gap": round(max(span - len(out) / sr, 0.0), 3),
        "onset": round(lead, 3),
    }
    return out, segs, info


def _pairable(q, o, tol=PROGRESS_TOLERANCE, slow=False):
    """Cadence: whether TTS pause q (not a boundary) may pair with original pause o: within `tol` in speech
    progress (_progress_diff), and q at least MATCH_MIN_RATIO of o's length. If `slow`, also when o is at least
    SLOW_ORIG_PAUSE_S, within SLOW_PAIR_TOLERANCE and q at least MATCH_MIN_RATIO of SLOW_MATCH_REF_MAX_S (a short
    comma pause pairs with a speaker's multi-second pause)."""
    d = _progress_diff(q, o)
    if d <= tol and q["dur"] >= MATCH_MIN_RATIO * o["dur"]:
        return True
    return slow and o["dur"] >= SLOW_ORIG_PAUSE_S and d <= SLOW_PAIR_TOLERANCE and q["dur"] >= MATCH_MIN_RATIO * SLOW_MATCH_REF_MAX_S


def _relaxed(q, o, tol=PROGRESS_TOLERANCE):
    """Cadence: whether q and o pair only by the slow rule (_pairable)."""
    return _pairable(q, o, tol, True) and not _pairable(q, o, tol)


def _progress_diff(q, o):
    """Cadence: how far apart TTS pause q and original pause o are in speech progress."""
    return abs(q["p"] - o["p"])


def _matched_cap(x, y, boundary=True, slow=False):
    """Cadence: the longest a TTS pause of length x, matched to an original pause of length y, may grow to. A
    pause that isn't a boundary also grows to at most max(x + GROWTH_SLACK_S, GROWTH_RATIO * x), unless `slow` and
    y is at least SLOW_ORIG_PAUSE_S: then to the original's + slack, up to SLOW_STRETCH_MAX."""
    if slow and y >= SLOW_ORIG_PAUSE_S:
        return min(max(x, y) + MATCHED_PAUSE_SLACK_S, SLOW_STRETCH_MAX)
    cap = min(max(x, y) + MATCHED_PAUSE_SLACK_S, STRETCH_MAX)
    return cap if boundary else min(cap, max(x + GROWTH_SLACK_S, GROWTH_RATIO * x))


def _insert_silence(y, inserts):
    """y with zeros of the given lengths inserted at the given positions ((position, samples), sorted)."""
    parts, pos = [], 0
    for p, m in inserts:
        parts += [y[pos:p], np.zeros(m, dtype=np.float32)]
        pos = p
    parts.append(y[pos:])
    return np.concatenate(parts)


def _fit_speed(length, slot):
    """The speed-up a clip of `length` needs to fit `slot`: at least 1 (never slowed down), no cap."""
    return max(1.0, length / slot) if slot > 0 else 1.0


def _atempo_chain(tempo):
    """ffmpeg filter for `tempo`, chained so every atempo stage stays within 0.5-2.0."""
    stages = []
    while tempo > 2.0:
        stages.append(2.0)
        tempo /= 2.0
    while tempo < 0.5:
        stages.append(0.5)
        tempo /= 0.5
    stages.append(tempo)
    return ",".join(f"atempo={s:.6f}" for s in stages)


def _level(x, sr):
    """Speech level: 95th percentile of 20 ms frame RMS."""
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n == 0:
        return 0.0
    return float(np.percentile(np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)), 95))


def _runs(mask):
    """[start, end) indices of each consecutive True stretch."""
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _ffmpeg_pcm(ffmpeg_bin, x, sr_in, sr_out, filters=None):
    """Run mono float audio through ffmpeg (resample and/or filter) via pipes."""
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    cmd = [ffmpeg_bin, "-v", "error", "-f", "s16le", "-ar", str(sr_in), "-ac", "1", "-i", "pipe:0"]
    if filters:
        cmd += ["-filter:a", filters]
    cmd += ["-f", "s16le", "-ar", str(sr_out), "-ac", "1", "pipe:1"]
    out = subprocess.run(cmd, input=pcm, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32) / 32768


def _decode(ffmpeg_bin, path, sr, channels):
    """Decode any audio file to float (n, channels) at `sr`."""
    out = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-i", str(path), "-f", "s16le", "-ar", str(sr), "-ac", str(channels), "pipe:1"],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, dtype="<i2").astype(np.float32).reshape(-1, channels) / 32768


def _read_wav(path: Path):
    """16-bit PCM wav -> (mono float32 samples, sample rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def _write_wav(path: Path, x, sr):
    """Float audio, shape (n,) or (n, channels) -> 16-bit PCM wav."""
    x = np.asarray(x)
    ch = 1 if x.ndim == 1 else x.shape[1]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
