"""Faces at diarization: who is on screen and talking, and the diarization fixes that suggests.

Run with the LatentSync venv, through run.sh (RAM cap, free GPU):
    lipsync/run.sh faces <run dir> [video name, default input.mp4]
Reads the video, vocals_16k.wav and diarization.json (the audio-only turns) of a run being diarized and writes
<run dir>/faces.json: the pre-pass's shots, face tracks, people and Light-ASD scores (reused at lip-sync by
prepass.py), plus `fixes` and the `turns` they give. The app decides whether to apply them
(backend/diarization.py, ASD_APPLY).

Audio decides when someone speaks; faces decide who, where they are clear. Each turn is cut at shot cuts into
pieces; a piece with a clearly speaking face votes its seconds for (speaker, person). Every pair with enough
seconds makes that person a final speaker:
- split: an audio speaker with two or more such people (surely different ones) becomes one speaker per person;
- merge: audio speakers sharing such a person become one;
- relabel: a clear piece of such a person goes to them, whoever's turn it was.
A piece with no clear face stays with its audio speaker; if that speaker was split, it goes to the person of the
speaker's clear piece just before it (the one after only if that is much closer).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from prepass import FPS, GPU_FRACTION, Peaks, analyse, assign, probe, save_faces

FIX_MIN_S = 1.0  # s: shorter pieces never count as clear
FIX_MARGIN = 1.0  # a clear face's mean score leads the next eligible face in the piece (or 0, if alone) by this
FIX_POSITIVE = 0.6  # ... and at least this share of its frames in the piece score above 0
VOTE_MIN_S = 2.5  # s of clear pieces a (speaker, person) pair needs to make that person a speaker
DISTINCT_SIM = 0.25  # people never on screen together are surely different below this face similarity
AFTER_WEIGHT = 5.0  # an unclear piece goes to the clear piece after it only if that is this many times closer
MIN_SEG = 0.3  # s: a shorter piece takes its turn's neighbouring piece's speaker (diarization.MIN_SEG)


def speaking_face(p, tracks):
    """The face clearly speaking in a piece: {track, score, lead, positive}, or None."""
    ok = [c for c in p["candidates"] if c["eligible"]]  # sorted by score, best first
    if not ok or p["end"] - p["start"] < FIX_MIN_S or ok[0]["score"] <= 0:
        return None
    top = ok[0]
    lead = top["score"] - (ok[1]["score"] if len(ok) > 1 else 0.0)
    t = tracks[top["track"]]
    a, b = max(int(round(p["start"] * FPS)), t["f0"]), min(int(round(p["end"] * FPS)), t["f1"] + 1)
    if b <= a:
        return None
    positive = float(np.mean(t["smooth_scores"][a - t["f0"] : b - t["f0"]] > 0))
    if lead < FIX_MARGIN or positive < FIX_POSITIVE:
        return None
    return {"track": top["track"], "score": round(top["score"], 2), "lead": round(lead, 2), "positive": round(positive, 2)}


def distinct(a, b, tracks):
    """People a and b are surely different: on screen at the same time, or their faces are far apart."""
    ta = [t for t in tracks if t["person"] == a]
    tb = [t for t in tracks if t["person"] == b]
    if any(x["f0"] <= y["f1"] and y["f0"] <= x["f1"] for x in ta for y in tb):
        return True
    return float(np.mean([x["embedding"] @ y["embedding"] for x in ta for y in tb])) < DISTINCT_SIM


def fix_speakers(turns, shots, tracks):
    """Final speakers from the (speaker, person) seconds of clear pieces. Returns (turns, fixes, speaker_map):
    the turns renumbered 1..n in order of first appearance, touching same-speaker ones joined, and
    {audio-only speaker: [new speakers]}."""
    pieces, _ = assign(turns, shots, tracks)
    by_turn = {}
    for p in pieces:
        by_turn.setdefault(p["turn"], []).append(p)
    units = []  # the pieces in time order, edges snapped to the turn's
    for k, t in enumerate(turns):
        ps = sorted(by_turn.get(k, []), key=lambda p: p["start"])
        faces = [speaking_face(p, tracks) for p in ps] or [None]
        cuts = [t["start"]] + [p["start"] for p in ps[1:]] + [t["end"]]
        for i, f in enumerate(faces):
            units.append({"turn": k, "speaker": t["speaker"], "start": cuts[i], "end": cuts[i + 1], "face": f,
                          "person": tracks[f["track"]]["person"] if f else None})

    secs = {}
    for u in units:
        if u["person"] is not None:
            key = (u["speaker"], u["person"])
            secs[key] = secs.get(key, 0.0) + u["end"] - u["start"]
    people = {}  # speaker -> their people, most clear seconds first
    for (spk, person), s in sorted(secs.items(), key=lambda x: -x[1]):
        mine = people.setdefault(spk, [])
        if s >= VOTE_MIN_S and all(distinct(person, q, tracks) for q in mine):
            mine.append(person)
    counted = {p for ps in people.values() for p in ps}

    for u in units:
        mine = people.get(u["speaker"], [])
        if u["person"] in counted:
            u["key"] = ("person", u["person"])
        elif not mine:
            u["key"] = ("speaker", u["speaker"])
        elif len(mine) == 1:
            u["key"] = ("person", mine[0])
        else:
            anchors = [v for v in units if v["speaker"] == u["speaker"] and v["person"] in mine]
            before = max((v for v in anchors if v["end"] <= u["start"]), key=lambda v: v["end"], default=None)
            after = min((v for v in anchors if v["start"] >= u["end"]), key=lambda v: v["start"], default=None)
            if before is None or (after and AFTER_WEIGHT * (after["start"] - u["end"]) < u["start"] - before["end"]):
                before = after
            u["key"] = ("person", before["person"])
    for i, u in enumerate(units):
        if u["end"] - u["start"] < MIN_SEG:
            nb = [v for v in units[max(i - 1, 0) : i] + units[i + 1 : i + 2] if v["turn"] == u["turn"]]
            if nb:
                u["key"] = nb[0]["key"]

    joined = []
    for u in units:
        if joined and joined[-1][0] == u["key"] and u["start"] - joined[-1][2] < 1e-6:
            joined[-1][2] = u["end"]
        else:
            joined.append([u["key"], u["start"], u["end"]])
    ids = {}
    for key, _, _ in joined:
        ids.setdefault(key, len(ids) + 1)
    out = [{"speaker": ids[k], "start": round(a, 3), "end": round(b, 3)} for k, a, b in joined]
    speaker_map = {}
    for u in units:
        speaker_map.setdefault(str(u["speaker"]), set()).add(ids[u["key"]])

    fixes = []
    for spk, ps in people.items():
        if len(ps) > 1:
            fixes.append({"type": "split", "speaker": spk, "into": [ids[("person", p)] for p in ps], "people": ps,
                          "seconds": [round(secs[(spk, p)], 2) for p in ps]})
    for person in counted:
        spks = [s for s, ps in people.items() if person in ps]
        if len(spks) > 1:
            fixes.append({"type": "merge", "speakers": spks, "into": ids[("person", person)], "person": person,
                          "seconds": [round(secs[(s, person)], 2) for s in spks]})
    for u in units:
        if u["person"] in counted and u["person"] not in people.get(u["speaker"], []):
            fixes.append({"type": "relabel", "turn": u["turn"], "start": round(u["start"], 3),
                          "end": round(u["end"], 3), "from": u["speaker"], "to": ids[u["key"]],
                          "person": u["person"], **u["face"]})
    return out, fixes, {k: sorted(v) for k, v in speaker_map.items()}


def main():
    run = Path(sys.argv[1])
    video = run / (sys.argv[2] if len(sys.argv) > 2 else "input.mp4")
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    peaks, times, t0 = Peaks(), {}, time.perf_counter()

    w, h, _ = probe(video)
    turns = json.loads((run / "diarization.json").read_text())["turns"]
    n, shots, tracks, n_people = analyse(run, video, w, h, times)

    s = time.perf_counter()
    fixed, fixes, speaker_map = fix_speakers(turns, shots, tracks)
    times["fixes"] = time.perf_counter() - s
    times["total"] = time.perf_counter() - t0
    peaks.stop()

    save_faces(run / "faces.json", n, shots, tracks, n_people, {
        "fixes": fixes,
        "turns": fixed,
        "speaker_map": speaker_map,
        "stats": {"times_s": {k: round(v, 1) for k, v in times.items()},
                  "peak_vram_mb": peaks.vram_mb, "peak_anon_mb": peaks.anon_mb},
    })
    counts = {k: sum(f["type"] == k for f in fixes) for k in ("split", "merge", "relabel")}
    print("FACES " + json.dumps({"people": n_people, "tracks": len(tracks), **counts,
                                 "times_s": {k: round(v, 1) for k, v in times.items()}}), flush=True)


if __name__ == "__main__":
    main()
