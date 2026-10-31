"""Speaker portraits: a small square face crop per speaker, for the UI.

Made right after diarization from faces.json (lipsync/faces.py: shots, face tracks, people, Light-ASD scores) and
the run's final turns (diarization.json, after the face fixes, if any). A speaker gets the person whose face is
clearly speaking in their turns for the most seconds, the test faces.py uses for its fixes; no two speakers get
the same person. A speaker never clearly seen talking (a voice-over, a narrator) gets no portrait.
The crop is the sharpest of a few frames where that person is talking, read by ffmpeg.

Writes <run dir>/portraits/: <speaker>.jpg and index.json ({speaker: {person, seconds, frame}}, or {"error"}).

Old runs: python -m backend.portraits [run id ...] (default: every run in storage/). A run without faces.json
first gets one (lipsync/run.sh faces, needs a free GPU); its turns are left as they are.
"""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DIR = "portraits"
SIZE = 128  # px, square
PAD = 1.7  # crop side / face box side: room for hair and chin
LIFT = 0.06  # the crop's centre sits this share of its side above the face box's
CLEAR_MIN_S = 1.0  # faces.py FIX_MIN_S: shorter pieces never count as clear
CLEAR_LEAD = 1.0  # faces.py FIX_MARGIN: the face's mean score leads the next eligible face's (or 0) by this
CLEAR_POSITIVE = 0.6  # faces.py FIX_POSITIVE: share of its frames in the piece scoring above 0
VOTE_MIN_S = 2.0  # s of clear pieces a person needs to be a speaker's portrait
CANDIDATES = 6  # frames tried per portrait


def _smooth(scores):
    return np.convolve(np.pad(scores, 2, mode="edge"), np.ones(5) / 5, mode="valid")  # prepass.smooth


def _clear_pieces(turns, faces):
    """(speaker, person, track, a, b) for every piece (a turn cut at shot cuts, frames a..b-1) with a face
    clearly speaking in it."""
    fps, tracks = faces["fps"], faces["tracks"]
    out = []
    for turn in turns:
        a, b = int(round(turn["start"] * fps)), int(round(turn["end"] * fps))
        for s, (sa, sb) in enumerate(faces["shots"]):
            pa, pb = max(a, sa), min(b, sb)
            if (pb - pa) / fps < CLEAR_MIN_S:
                continue
            cands = []
            for t in tracks:
                oa, ob = max(pa, t["f0"]), min(pb, t["f1"] + 1)
                if t["shot"] != s or not t["eligible"] or ob - oa < 0.25 * (pb - pa):
                    continue
                sc = t["smooth"][oa - t["f0"] : ob - t["f0"]]
                cands.append((float(sc.mean()), float((sc > 0).mean()), t, oa, ob))
            cands.sort(key=lambda c: -c[0])
            if not cands:
                continue
            score, positive, t, oa, ob = cands[0]
            lead = score - (cands[1][0] if len(cands) > 1 else 0.0)
            if score > 0 and lead >= CLEAR_LEAD and positive >= CLEAR_POSITIVE:
                out.append((turn["speaker"], t["person"], t, oa, ob))
    return out


def match(turns, faces):
    """{speaker: (person, seconds, pieces)}: each speaker's person, most clear seconds first, one speaker each."""
    for t in faces["tracks"]:
        t["smooth"] = _smooth(np.array(t["scores"], float))
    pieces = _clear_pieces(turns, faces)
    secs = {}
    for spk, person, _, a, b in pieces:
        secs[(spk, person)] = secs.get((spk, person), 0.0) + (b - a) / faces["fps"]
    out, taken = {}, set()
    for (spk, person), s in sorted(secs.items(), key=lambda x: -x[1]):
        if s >= VOTE_MIN_S and spk not in out and person not in taken:
            out[spk] = (person, s, [p for p in pieces if p[0] == spk and p[1] == person])
            taken.add(person)
    return out


def _frames(pieces):
    """Candidate (track, frame) pairs: per piece its middle frame and its biggest face, biggest faces first."""
    cands = set()
    for _, _, t, a, b in pieces:
        heights = np.array([box[3] - box[1] for box in t["boxes"][a - t["f0"] : b - t["f0"]]])
        cands.add((t["id"], (a + b) // 2))
        cands.add((t["id"], a + int(heights.argmax())))
    tracks = {p[2]["id"]: p[2] for p in pieces}

    def height(c):
        box = tracks[c[0]]["boxes"][c[1] - tracks[c[0]]["f0"]]
        return box[3] - box[1]

    return [(tracks[tid], f) for tid, f in sorted(cands, key=height, reverse=True)[:CANDIDATES]]


def _grab(video, t_s, ffmpeg_bin):
    p = subprocess.run(
        [ffmpeg_bin, "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(video), "-frames:v", "1",
         "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
    )
    if p.returncode or not p.stdout:
        raise RuntimeError(f"ffmpeg couldn't read the frame at {t_s:.2f} s: {p.stderr.decode()[-300:]}")
    return Image.open(io.BytesIO(p.stdout)).convert("RGB")


def _crop(img, box):
    """A square around the face box, padded and lifted, kept inside the frame, at SIZE px."""
    x1, y1, x2, y2 = box
    side = min(PAD * max(x2 - x1, y2 - y1), img.width, img.height)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - LIFT * side
    left = min(max(cx - side / 2, 0), img.width - side)
    top = min(max(cy - side / 2, 0), img.height - side)
    return img.resize((SIZE, SIZE), Image.LANCZOS, box=(left, top, left + side, top + side))


def _sharpness(img):
    g = np.asarray(img.convert("L"), float)
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def build(jdir: Path, video: Path, ffmpeg_bin: str) -> dict:
    """Make <run dir>/portraits/ from faces.json and diarization.json's turns, replacing any earlier one.
    Returns the index. Never raises: a failure is recorded as the index's "error" (and no portraits)."""
    tmp = jdir / f"{DIR}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    try:
        faces = json.loads((jdir / "faces.json").read_text())
        turns = json.loads((jdir / "diarization.json").read_text())["turns"]
        index = {}
        for spk, (person, secs, pieces) in sorted(match(turns, faces).items()):
            best = None
            for t, f in _frames(pieces):
                crop = _crop(_grab(video, f / faces["fps"], ffmpeg_bin), t["boxes"][f - t["f0"]])
                sharp = _sharpness(crop)
                if best is None or sharp > best[0]:
                    best = (sharp, crop, f)
            best[1].save(tmp / f"{spk}.jpg", quality=88)
            index[str(spk)] = {"person": person, "seconds": round(secs, 2), "frame": best[2]}
    except Exception as exc:
        shutil.rmtree(tmp)
        tmp.mkdir()
        index = {"error": f"{type(exc).__name__}: {exc}"}
    (tmp / "index.json").write_text(json.dumps(index, indent=2))
    shutil.rmtree(jdir / DIR, ignore_errors=True)
    tmp.rename(jdir / DIR)
    return index


def clear(jdir: Path):
    shutil.rmtree(jdir / DIR, ignore_errors=True)


def load(jdir: Path) -> dict:
    """{speaker: version} of the speakers with a portrait; the version (the file's mtime) changes when it is
    remade, for the browser's cache. {} for runs without portraits."""
    out = {}
    for path in (jdir / DIR).glob("*.jpg"):
        if path.stem.isdigit():
            out[int(path.stem)] = int(path.stat().st_mtime)
    return out


def path(jdir: Path, speaker: int) -> Path | None:
    p = jdir / DIR / f"{speaker}.jpg"
    return p if p.is_file() else None


def main():
    root = Path(__file__).resolve().parent.parent
    storage = root / "storage"
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        sys.exit("needs imageio_ffmpeg (the app's venv)")
    ffmpeg_bin = get_ffmpeg_exe()
    runs = [storage / r for r in sys.argv[1:]] or sorted(p for p in storage.iterdir() if (p / "run.json").is_file())
    for jdir in runs:
        info = json.loads((jdir / "run.json").read_text())
        if not (jdir / "diarization.json").is_file() or not info.get("video"):
            print(f"{jdir.name}: not diarized, skipped")
            continue
        video = jdir / info["video"]
        if not (jdir / "faces.json").is_file():
            print(f"{jdir.name}: no faces.json, running lipsync/run.sh faces ...", flush=True)
            with open(jdir / "faces.log", "a") as log:
                code = subprocess.run([str(root / "lipsync/run.sh"), "faces", str(jdir), video.name],
                                      stdout=log, stderr=subprocess.STDOUT).returncode
            if code or not (jdir / "faces.json").is_file():
                print(f"{jdir.name}: faces failed (exit {code}, see faces.log), skipped")
                continue
        index = build(jdir, video, ffmpeg_bin)
        print(f"{jdir.name}: {index}")


if __name__ == "__main__":
    main()
