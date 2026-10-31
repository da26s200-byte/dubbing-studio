"""SyncNet lip-sync confidence (LatentSync's eval/eval_sync_conf.py): the worker's keep-or-drop check, and a
standalone re-scorer for tests.

Standalone, with the LatentSync venv, through run.sh (RAM cap, free GPU):
    lipsync/run.sh syncscore <run dir>
re-scores each synced clip of lipsync/clips.json and the same crop unsynced with the same dub audio, and
writes lipsync/syncscore.json. Higher is better.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import fmean

from prepass import FFMPEG, LATENTSYNC

REPO = LATENTSYNC / "LatentSync"
MIN_TRACK = 25  # frames: shortest face track SyncNet scores (the stock 50 would leave ranges under 2 s unscored)


def load_scorer(work):
    """SyncNet and its S3FD face detector, working in `work` (they write ./detect_results and ./temp).
    Returns score(video) -> confidence, or None if no face track of MIN_TRACK frames is found."""
    for name in ("checkpoints", "eval", "latentsync"):
        if not (work / name).exists():
            (work / name).symlink_to(REPO / name)
    os.chdir(work)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from eval.syncnet import SyncNetEval
    from eval.syncnet_detect import SyncNetDetector

    syncnet = SyncNetEval(device="cuda")
    syncnet.loadParameters("checkpoints/auxiliary/syncnet_v2.model")
    detector = SyncNetDetector(device="cuda", detect_results_dir="detect_results")

    def score(video):
        shutil.rmtree("detect_results", ignore_errors=True)
        detector(video_path=str(video), min_track=MIN_TRACK)
        crops = sorted(os.listdir("detect_results/crop"))
        if not crops:
            return None
        return round(fmean(syncnet.evaluate(video_path=f"detect_results/crop/{c}", temp_dir="temp")[2] for c in crops), 2)

    return score


def unsynced(clip_dir, rid, out):
    """The range's crop as it was, with the dub audio: the reference the synced clip must beat."""
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(clip_dir / f"{rid}_in.mp4"), "-i", str(clip_dir / f"{rid}.wav"),
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], check=True)
    return out


def main():
    run = Path(sys.argv[1]).resolve()
    ls_dir = run / "lipsync"
    clips = json.loads((ls_dir / "clips.json").read_text())["clips"]
    work = ls_dir / "work_sync"
    work.mkdir(exist_ok=True)
    score = load_scorer(work)
    out = []
    for c in clips:
        if not (ls_dir / "clips" / f"{c['id']}.mp4").exists():
            continue
        ref = unsynced(ls_dir / "clips", c["id"], work / f"ref_{c['id']}.mp4")
        rec = {"id": c["id"], "seconds": round(c["end"] - c["start"], 2),
               "synced": score(ls_dir / "clips" / f"{c['id']}.mp4"), "unsynced": score(ref)}
        out.append(rec)
        print("SCORE " + json.dumps(rec), flush=True)
    (ls_dir / "syncscore.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
