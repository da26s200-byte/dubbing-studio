"""Lip-sync pre-pass: who is on screen and who is talking (phase 2 of lipsync_plan.md).

Run with the LatentSync venv, through run.sh (RAM cap, free GPU):
    lipsync/run.sh prepass <run dir> [--debug]
Reads input.mp4, vocals_16k.wav (the ORIGINAL vocals), diarization.json and chunks.json of a dubbed run and
writes <run dir>/lipsync/plan.json, plus debug.mp4 and timeline.png with --debug. If the run has faces.json
(lipsync/faces.py, made at diarization), its shots, tracks, people and scores are reused. Frames are streamed from
ffmpeg, never held whole; only boxes, landmarks, embeddings and 112 px face patches are kept.
"""

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import python_speech_features
import torch
from scipy.io import wavfile
from scipy.ndimage import median_filter

LATENTSYNC = Path("/data1/pratyaymishra_data/latentsync")
FFMPEG = str(LATENTSYNC / "venv/bin/ffmpeg")
INSIGHTFACE_ROOT = str(LATENTSYNC / "LatentSync/checkpoints/auxiliary")  # models/buffalo_l, as LatentSync uses
LIGHT_ASD = LATENTSYNC / "Light-ASD"
ASD_WEIGHTS = LIGHT_ASD / "weight/finetuning_TalkSet.model"

FPS = 25  # LatentSync's and Light-ASD's frame rate; everything here is in frames at this rate
SHOT_THRESHOLD = 27.0  # PySceneDetect ContentDetector default
DET_SIZE = 640  # SCRFD input size
DET_EVERY = 2  # frames: faces are detected on every DET_EVERY-th frame (and each shot's first); tracks interpolate
DET_SCORE = 0.5  # detections below this are ignored
DET_MIN = 20  # px: smaller detections are ignored (noise)
MIN_FACE = 40  # px: a track whose median face height is below this is never synced (too small to look good)
TRACK_IOU = 0.5  # a detection continues a track if it overlaps the track's last box this much
TRACK_GAP = 10  # frames: missed detections bridged inside a track (Light-ASD's demo uses 10)
MIN_TRACK = 12  # frames: shorter tracks are dropped (0.5 s)
CROP_SCALE = 0.40  # Light-ASD's face crop padding (Columbia_test.py)
EMBED_EVERY = FPS  # frames between ArcFace embeddings along a track
SAME_PERSON = 0.35  # cosine similarity of mean ArcFace embeddings: two tracks are the same person
ASD_DURATIONS = (1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6)  # s: Light-ASD scoring windows, averaged (its demo)
ASD_MIN = 0.0  # mean speaking score a face needs to be picked for a turn (> 0 = speaking)
ASD_MARGIN = 0.5  # ... and its lead over the next face in the shot
STATIC_DIFF = 1.0  # mean abs change of the 112 px patch per frame (0-255) below this = a still image
RANGE_PAD = 0.2  # s added around each speaking interval
RANGE_JOIN = 1.0  # s: speaking intervals closer than this are joined
RANGE_MIN = 0.5  # s: shorter ranges are dropped
CHUNK_MAX = 30.0  # s: longer ranges are split (bounds LatentSync's RAM)
GPU_FRACTION = 0.6  # torch may use at most this share of the GPU
ORT_GPU_LIMIT = 2 << 30  # bytes: onnxruntime's GPU arena cap


# ---------- video ----------

def probe(path):
    """(width, height, fps) of the first video stream."""
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5}).*?([\d.]+) fps", err)
    if not m:
        raise RuntimeError(f"can't read the video stream of {path}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3))


def frames(path, w, h):
    """Yield (index, BGR frame) of the video resampled to FPS, decoded by ffmpeg in a pipe."""
    p = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", str(path), "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE,
    )
    size, i = w * h * 3, 0
    try:
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.stdout.close()
        p.kill()
        p.wait()


# ---------- models ----------

def load_insightface():
    from insightface.app import FaceAnalysis

    opts = {"device_id": 0, "gpu_mem_limit": ORT_GPU_LIMIT, "arena_extend_strategy": "kSameAsRequested"}
    app = FaceAnalysis(
        name="buffalo_l", root=INSIGHTFACE_ROOT, allowed_modules=["detection", "recognition"],
        providers=["CUDAExecutionProvider"], provider_options=[opts],
    )
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE), det_thresh=DET_SCORE)
    return app.det_model, app.models["recognition"]


class _ASD(torch.nn.Module):
    """Light-ASD's model and audio-visual head, loaded without its training wrapper (ASD.py needs pandas)."""

    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(LIGHT_ASD))
        from model.Model import ASD_Model
        from loss import lossAV

        self.model, self.lossAV = ASD_Model(), lossAV()
        own = self.state_dict()
        for name, value in torch.load(ASD_WEIGHTS, map_location="cpu").items():
            name = name if name in own else name.replace("module.", "")
            if name in own and own[name].shape == value.shape:  # skips lossV (visual-only head, unused)
                own[name].copy_(value)

    @torch.no_grad()
    def score(self, patches, audio):
        """Per-frame speaking score of a face track: patches (T, 112, 112) uint8 gray at FPS, audio int16
        16 kHz of the same span. Columbia_test.evaluate_network: MFCC at 100/s, windows of ASD_DURATIONS."""
        mfcc = python_speech_features.mfcc(audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
        length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100, len(patches) / FPS)
        mfcc, video = mfcc[: int(round(length * 100))], patches[: int(round(length * FPS))]
        runs = []
        for d in ASD_DURATIONS:
            s = []
            for i in range(int(np.ceil(length / d))):
                a = torch.FloatTensor(mfcc[i * d * 100 : (i + 1) * d * 100]).unsqueeze(0).cuda()
                v = torch.FloatTensor(video[i * d * FPS : (i + 1) * d * FPS]).unsqueeze(0).cuda()
                out = self.model.forward_audio_visual_backend(
                    self.model.forward_audio_frontend(a), self.model.forward_visual_frontend(v)
                )
                s.extend(self.lossAV(out, labels=None))
            runs.append(s)
        n = min(len(r) for r in runs)
        scores = np.mean([r[:n] for r in runs], axis=0)
        return np.pad(scores, (0, max(len(patches) - len(scores), 0)), mode="edge")[: len(patches)]


# ---------- helpers ----------

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def asd_patch(frame, cx, cy, s):
    """Light-ASD's face crop (Columbia_test.crop_video, cropScale 0.4) as the 112x112 gray patch its scorer
    reads: the centre of the 224 px resize."""
    bsi = int(s * (1 + 2 * CROP_SCALE))
    pad = cv2.copyMakeBorder(frame, bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT, value=(110, 110, 110))
    my, mx = cy + bsi, cx + bsi
    face = pad[int(my - s) : int(my + s * (1 + 2 * CROP_SCALE)), int(mx - s * (1 + CROP_SCALE)) : int(mx + s * (1 + CROP_SCALE))]
    gray = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def merge_intervals(iv, join=0.0):
    out = []
    for a, b in sorted(iv):
        if out and a - out[-1][1] < join:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class Peaks:
    """Peak VRAM of this process (nvidia-smi, so onnxruntime is included) and peak anonymous RAM (RssAnon:
    without memory-mapped files such as the checkpoint, which the kernel can drop), polled every 0.5 s."""

    def __init__(self):
        self.vram_mb, self.anon_mb, self._stop = 0, 0, threading.Event()
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        pid = str(os.getpid())
        while not self._stop.wait(0.5):
            with open("/proc/self/status") as f:
                anon = next(int(l.split()[1]) for l in f if l.startswith("RssAnon:"))
            self.anon_mb = max(self.anon_mb, anon // 1024)
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout
            for line in out.splitlines():
                p, _, m = line.partition(",")
                if p.strip() == pid:
                    self.vram_mb = max(self.vram_mb, int(m))

    def stop(self):
        self._stop.set()


# ---------- the pass ----------

def detect(video, w, h, det_model):
    """Pass 1: shot cuts (PySceneDetect) and SCRFD faces on every DET_EVERY-th frame and each shot's first.
    Frames not looked at get None."""
    from scenedetect.detectors import ContentDetector

    shots = ContentDetector(threshold=SHOT_THRESHOLD, min_scene_len=FPS // 2)
    cuts, dets, sw = [0], [], 320
    for i, frame in frames(video, w, h):
        small = cv2.resize(frame, (sw, int(h * sw / w)))
        new = [c for c in shots.process_frame(i, small) if c > 0]
        cuts += new
        if i % DET_EVERY and i not in new:
            dets.append(None)
            continue
        boxes, kpss = det_model.detect(frame, max_num=0, metric="default")
        dets.append([
            (b[:4].astype(float), float(b[4]), k)
            for b, k in zip(boxes, kpss)
            if b[4] >= DET_SCORE and b[3] - b[1] >= DET_MIN
        ])
    n = len(dets)
    cuts = sorted(set(c for c in cuts if c < n))
    return [(a, b) for a, b in zip(cuts, cuts[1:] + [n])], dets


def track(shots, dets):
    """Link detections into tracks inside each shot (IoU with the last box, gaps up to TRACK_GAP)."""
    shot_of = np.zeros(len(dets), int)
    for k, (a, b) in enumerate(shots):
        shot_of[a:b] = k
    tracks, active = [], []
    for i, faces in enumerate(dets):
        if faces is None:
            continue
        active = [t for t in active if i - t["det"][-1][0] <= TRACK_GAP and t["shot"] == shot_of[i]]
        pairs = sorted(
            ((iou(f[0], t["det"][-1][1]), fi, ti) for fi, f in enumerate(faces) for ti, t in enumerate(active)),
            key=lambda x: -x[0],
        )
        used_f, used_t = set(), set()
        for v, fi, ti in pairs:
            if v < TRACK_IOU:
                break
            if fi not in used_f and ti not in used_t:
                active[ti]["det"].append((i, faces[fi][0], faces[fi][2]))
                used_f.add(fi)
                used_t.add(ti)
        for fi, f in enumerate(faces):
            if fi not in used_f:
                t = {"shot": int(shot_of[i]), "det": [(i, f[0], f[2])]}
                tracks.append(t)
                active.append(t)

    out = []
    for t in tracks:
        f = np.array([d[0] for d in t["det"]])
        if f[-1] - f[0] + 1 < MIN_TRACK:
            continue
        b = np.array([d[1] for d in t["det"]])
        end = int(f[-1])
        while end + 1 < len(dets) and dets[end + 1] is None and shot_of[end + 1] == t["shot"]:
            end += 1  # frames after the last detection that weren't looked at (the box is held)
        span = np.arange(f[0], end + 1)
        looked = sum(dets[i] is not None for i in span)
        boxes = np.stack([np.interp(span, f, b[:, j]) for j in range(4)], axis=1)
        s = median_filter(np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / 2, 13, mode="nearest")
        cx = median_filter((boxes[:, 0] + boxes[:, 2]) / 2, 13, mode="nearest")
        cy = median_filter((boxes[:, 1] + boxes[:, 3]) / 2, 13, mode="nearest")
        out.append({
            "id": len(out), "shot": t["shot"], "f0": int(f[0]), "f1": end, "boxes": boxes,
            "smooth": np.stack([cx, cy, s], axis=1), "kps": {int(i): k for i, _, k in t["det"]},
            "detected": len(f) / looked, "height": float(np.median(boxes[:, 3] - boxes[:, 1])),
        })
    return out


def patches_and_embeddings(video, w, h, tracks, rec_model):
    """Pass 2: Light-ASD patches for every frame of every track, ArcFace embeddings every EMBED_EVERY."""
    from insightface.utils import face_align

    for t in tracks:
        t["patches"] = np.zeros((t["f1"] - t["f0"] + 1, 112, 112), np.uint8)
        t["emb"], t["last_emb"] = [], -EMBED_EVERY
    for i, frame in frames(video, w, h):
        for t in tracks:
            if not t["f0"] <= i <= t["f1"]:
                continue
            cx, cy, s = t["smooth"][i - t["f0"]]
            t["patches"][i - t["f0"]] = asd_patch(frame, cx, cy, s)
            if i in t["kps"] and i - t["last_emb"] >= EMBED_EVERY:
                e = rec_model.get_feat(face_align.norm_crop(frame, landmark=t["kps"][i]))[0]
                t["emb"].append(e / np.linalg.norm(e))
                t["last_emb"] = i
    for t in tracks:
        m = np.mean(t["emb"], axis=0)
        t["embedding"] = m / np.linalg.norm(m)
        p = t["patches"].astype(np.int16)
        t["motion"] = float(np.abs(np.diff(p, axis=0)).mean()) if len(p) > 1 else 0.0
        t["static"] = t["motion"] < STATIC_DIFF
        t["eligible"] = not t["static"] and t["height"] >= MIN_FACE


def identities(tracks):
    """Cluster tracks into people: average-linkage on mean embeddings, never joining tracks seen together."""
    groups = [[t["id"]] for t in tracks]

    def sim(g1, g2):
        return float(np.mean([tracks[a]["embedding"] @ tracks[b]["embedding"] for a in g1 for b in g2]))

    def together(g1, g2):
        return any(tracks[a]["f0"] <= tracks[b]["f1"] and tracks[b]["f0"] <= tracks[a]["f1"] for a in g1 for b in g2)

    while True:
        best = max(
            ((sim(g1, g2), i, j) for i, g1 in enumerate(groups) for j, g2 in enumerate(groups) if i < j and not together(g1, g2)),
            default=None,
        )
        if not best or best[0] < SAME_PERSON:
            break
        _, i, j = best
        groups[i] += groups.pop(j)
    for k, g in enumerate(groups):
        for tid in g:
            tracks[tid]["person"] = k
    return len(groups)


def assign(turns, shots, tracks):
    """Each diarization turn, split at shot cuts, gets the face that is speaking (Light-ASD), then speakers are
    matched to people by vote; unsure pieces fall back to the speaker's person if on screen."""
    pieces = []
    for k, turn in enumerate(turns):
        a, b = int(round(turn["start"] * FPS)), int(round(turn["end"] * FPS))
        for s, (sa, sb) in enumerate(shots):
            pa, pb = max(a, sa), min(b, sb)
            if pb - pa < 1:
                continue
            cands = []
            for t in tracks:
                oa, ob = max(pa, t["f0"]), min(pb, t["f1"] + 1)
                if t["shot"] != s or ob - oa < 0.25 * (pb - pa):
                    continue
                cands.append({
                    "track": t["id"], "score": float(np.mean(t["smooth_scores"][oa - t["f0"] : ob - t["f0"]])),
                    "eligible": t["eligible"],
                })
            cands.sort(key=lambda c: -c["score"])
            ok = [c for c in cands if c["eligible"]]
            piece = {"turn": k, "speaker": turn["speaker"], "shot": s, "start": pa / FPS, "end": pb / FPS,
                     "candidates": cands, "track": None, "method": "no_face"}
            if ok:
                lead = ok[0]["score"] - (ok[1]["score"] if len(ok) > 1 else -np.inf)
                if ok[0]["score"] >= ASD_MIN and lead >= ASD_MARGIN:
                    piece.update(track=ok[0]["track"], method="asd")
                else:
                    piece["method"] = "unsure"
            pieces.append(piece)

    votes = {}
    for p in pieces:
        if p["method"] == "asd":
            key = (p["speaker"], tracks[p["track"]]["person"])
            votes[key] = votes.get(key, 0.0) + p["end"] - p["start"]
    person_of = {}
    for (spk, person), v in sorted(votes.items(), key=lambda x: -x[1]):
        if spk not in person_of and person not in person_of.values():
            person_of[spk] = person

    for p in pieces:
        mine = person_of.get(p["speaker"])
        if p["method"] == "asd" and tracks[p["track"]]["person"] != mine:
            p.update(track=None, method="conflict")  # the face picked isn't this speaker's person
        elif p["method"] == "unsure":
            own = [c for c in p["candidates"] if c["eligible"] and tracks[c["track"]]["person"] == mine]
            if mine is not None and own:
                p.update(track=own[0]["track"], method="person")
            else:
                p["method"] = "skip"
    return pieces, person_of


def speaking_ranges(chunks, pieces, shots, tracks, person_of):
    """What gets lip-synced: per speaker, the union of their segments and where their dub plays (padded,
    joined, short ones dropped), cut to the shots and frames where their face is tracked, at most CHUNK_MAX."""
    by_id = {c["id"]: c for c in chunks}
    ranges = []
    for spk in sorted({c["speaker"] for c in chunks}):
        iv = []
        for c in chunks:
            if c["speaker"] != spk:
                continue
            iv.append((c["start"], c["end"]))
            d = c.get("dub")
            if d:  # where the dubbed clip actually plays inside its unit
                u0 = by_id[d["unit"]]["start"]
                ps = u0 + (c["start"] - u0) / d["speed"]
                iv.append((ps, ps + d["duration"]))
        iv = [(max(a - RANGE_PAD, 0.0), b + RANGE_PAD) for a, b in iv]
        for a, b in merge_intervals(iv, RANGE_JOIN):
            if b - a < RANGE_MIN:
                continue
            fa, fb = int(round(a * FPS)), int(round(b * FPS))
            for s, (sa, sb) in enumerate(shots):
                pa, pb = max(fa, sa), min(fb, sb)
                if pb - pa < 1:
                    continue
                picked = [p["track"] for p in pieces if p["speaker"] == spk and p["shot"] == s and p["track"] is not None]
                if picked:
                    tid = max(set(picked), key=picked.count)
                else:  # e.g. the dub runs into the next shot: the speaker's person, if tracked there
                    own = [t for t in tracks if t["shot"] == s and t["eligible"] and t["person"] == person_of.get(spk)]
                    if not own:
                        continue
                    tid = max(own, key=lambda t: min(pb, t["f1"] + 1) - max(pa, t["f0"]))["id"]
                t = tracks[tid]
                qa, qb = max(pa, t["f0"]), min(pb, t["f1"] + 1)
                if (qb - qa) / FPS < RANGE_MIN:
                    continue
                parts = int(np.ceil((qb - qa) / FPS / CHUNK_MAX))
                edges = np.linspace(qa, qb, parts + 1).round().astype(int)
                for x, y in zip(edges, edges[1:]):
                    ranges.append({"id": len(ranges), "speaker": spk, "track": tid, "person": t["person"],
                                   "f0": int(x), "f1": int(y) - 1, "start": round(x / FPS, 3), "end": round(y / FPS, 3)})
    return ranges


# ---------- debug outputs ----------

COLORS = [(66, 135, 245), (60, 180, 75), (230, 159, 0), (204, 121, 167), (0, 158, 115), (213, 94, 0)]


def debug_video(video, w, h, out, tracks, turns, ranges, shots):
    """Pass 3: the original video and audio with faces boxed by person, live score, diarization and SYNC."""
    enc = subprocess.Popen(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS),
         "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)],
        stdin=subprocess.PIPE,
    )
    font, sc = cv2.FONT_HERSHEY_SIMPLEX, max(h / 720, 0.5)
    for i, frame in frames(video, w, h):
        img, t_s = frame.copy(), i / FPS
        shot = next(k for k, (a, b) in enumerate(shots) if a <= i < b)
        spk = [tn["speaker"] for tn in turns if tn["start"] <= t_s < tn["end"]]
        sync = {r["track"]: r["speaker"] for r in ranges if r["f0"] <= i <= r["f1"]}
        for t in tracks:
            if not t["f0"] <= i <= t["f1"]:
                continue
            x1, y1, x2, y2 = t["boxes"][i - t["f0"]].astype(int)
            col = COLORS[t["person"] % len(COLORS)]
            v = t["smooth_scores"][i - t["f0"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 6 if t["id"] in sync else 2)
            label = f"T{t['id']} P{t['person']} {v:+.1f}" + ("" if t["eligible"] else (" static" if t["static"] else " small"))
            cv2.putText(img, label, (x1, max(y1 - 8, 20)), font, 0.6 * sc, col, 2)
            if t["id"] in sync:
                cv2.putText(img, f"SYNC S{sync[t['id']]}", (x1, y2 + int(24 * sc)), font, 0.7 * sc, (0, 0, 255), 2)
        head = f"{t_s:6.2f}s  shot {shot}  speaker {'S' + str(spk[0]) if spk else '-'}"
        cv2.rectangle(img, (0, 0), (int(430 * sc), int(34 * sc)), (0, 0, 0), -1)
        cv2.putText(img, head, (8, int(24 * sc)), font, 0.7 * sc, (255, 255, 255), 2)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()


def timeline(out, n, tracks, turns, ranges, shots):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(14, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for t in tracks:
        x = np.arange(t["f0"], t["f1"] + 1) / FPS
        col = np.array(COLORS[t["person"] % len(COLORS)][::-1]) / 255
        ax.plot(x, t["smooth_scores"], color=col, lw=1.2, ls="-" if t["eligible"] else ":",
                label=f"T{t['id']} P{t['person']}" + ("" if t["eligible"] else " (not eligible)"))
    ax.axhline(ASD_MIN, color="grey", lw=0.8)
    for a, _ in shots[1:]:
        ax.axvline(a / FPS, color="black", lw=0.6, ls="--")
        bx.axvline(a / FPS, color="black", lw=0.6, ls="--")
    ax.set_ylabel("Light-ASD score (smoothed)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    spks = sorted({tn["speaker"] for tn in turns})
    for tn in turns:
        y = spks.index(tn["speaker"])
        bx.barh(y + 0.2, tn["end"] - tn["start"], left=tn["start"], height=0.35, color="#999")
    for r in ranges:
        y = spks.index(r["speaker"]) if r["speaker"] in spks else 0
        bx.barh(y - 0.2, r["end"] - r["start"], left=r["start"], height=0.35, color="#d62728")
    bx.set_yticks(range(len(spks)), [f"S{s}" for s in spks])
    bx.set_xlabel("seconds  (grey: diarization turns, red: ranges to lip-sync, dashed: shot cuts)")
    bx.set_xlim(0, n / FPS)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


# ---------- faces: analysis shared with faces.py ----------

def analyse(run, video, w, h, times):
    """Shots, face tracks, people and Light-ASD scores of a run's video against its original vocals.
    Returns (frames, shots, tracks, people); times gets each stage's seconds."""
    sr, vocals = wavfile.read(run / "vocals_16k.wav")
    if sr != 16000 or vocals.dtype != np.int16:
        raise ValueError("vocals_16k.wav must be 16 kHz int16")

    s = time.perf_counter()
    det_model, rec_model = load_insightface()
    asd = _ASD().cuda().eval()
    times["load"] = time.perf_counter() - s

    s = time.perf_counter()
    shots, dets = detect(video, w, h, det_model)
    tracks = track(shots, dets)
    times["shots_faces"] = time.perf_counter() - s

    s = time.perf_counter()
    patches_and_embeddings(video, w, h, tracks, rec_model)
    n_people = identities(tracks)
    times["patches_embeddings"] = time.perf_counter() - s

    s = time.perf_counter()
    for t in tracks:
        a, b = t["f0"] * 16000 // FPS, (t["f1"] + 1) * 16000 // FPS
        t["scores"] = asd.score(t["patches"], vocals[a:b])
        t["smooth_scores"] = smooth(t["scores"])
        del t["patches"]
    times["asd"] = time.perf_counter() - s
    return len(dets), shots, tracks, n_people


def smooth(scores):
    return np.convolve(np.pad(scores, 2, mode="edge"), np.ones(5) / 5, mode="valid")


FACE_FIELDS = ("id", "shot", "person", "f0", "f1", "detected", "height", "motion", "static", "eligible")


def save_faces(path, n, shots, tracks, n_people, extra):
    """faces.json: what analyse() found, enough for the pre-pass to skip it."""
    path.write_text(json.dumps({
        "fps": FPS, "frames": n, "people": n_people, "shots": [[a, b] for a, b in shots],
        "tracks": [{
            **{k: t[k] for k in FACE_FIELDS},
            "boxes": np.round(t["boxes"], 1).tolist(), "scores": np.round(t["scores"], 3).tolist(),
        } for t in tracks],
        **extra,
    }))


def load_faces(path):
    f = json.loads(path.read_text())
    if f["fps"] != FPS:
        raise ValueError(f"{path.name} was made at {f['fps']} fps, not {FPS}")
    tracks = []
    for t in f["tracks"]:
        t = {**t, "boxes": np.array(t["boxes"], float), "scores": np.array(t["scores"], float)}
        t["smooth_scores"] = smooth(t["scores"])
        tracks.append(t)
    return f["frames"], [tuple(s) for s in f["shots"]], tracks, f["people"]


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--debug", action="store_true", help="also write debug.mp4 and timeline.png")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "lipsync"
    out.mkdir(exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    peaks, times, t0 = Peaks(), {}, time.perf_counter()

    video = run / "input.mp4"
    w, h, src_fps = probe(video)
    turns = json.loads((run / "diarization.json").read_text())["turns"]
    chunks = json.loads((run / "chunks.json").read_text())
    if (run / "faces.json").exists():  # made at diarization (lipsync/faces.py) from the same video and vocals
        s = time.perf_counter()
        n, shots, tracks, n_people = load_faces(run / "faces.json")
        times["faces_reused"] = time.perf_counter() - s
    else:
        n, shots, tracks, n_people = analyse(run, video, w, h, times)
    dets = range(n)  # only its length is used below

    pieces, person_of = assign(turns, shots, tracks)
    ranges = speaking_ranges(chunks, pieces, shots, tracks, person_of)

    if args.debug:
        s = time.perf_counter()
        debug_video(video, w, h, out / "debug.mp4", tracks, turns, ranges, shots)
        timeline(out / "timeline.png", len(dets), tracks, turns, ranges, shots)
        times["debug"] = time.perf_counter() - s
    times["total"] = time.perf_counter() - t0
    peaks.stop()

    plan = {
        "fps": FPS, "frames": len(dets), "source": {"width": w, "height": h, "fps": src_fps},
        "shots": [{"f0": a, "f1": b - 1, "start": round(a / FPS, 3), "end": round(b / FPS, 3)} for a, b in shots],
        "people": n_people,
        "speaker_person": {str(k): v for k, v in person_of.items()},
        "tracks": [{
            "id": t["id"], "shot": t["shot"], "person": t["person"], "f0": t["f0"], "f1": t["f1"],
            "detected": round(t["detected"], 3), "height": round(t["height"], 1), "motion": round(t["motion"], 2),
            "static": t["static"], "eligible": t["eligible"], "mean_score": round(float(np.mean(t["scores"])), 2),
            "boxes": np.round(t["boxes"]).astype(int).tolist(), "scores": np.round(t["scores"], 2).tolist(),
        } for t in tracks],
        "pieces": [{**p, "start": round(p["start"], 3), "end": round(p["end"], 3),
                    "candidates": [{**c, "score": round(c["score"], 2)} for c in p["candidates"]]} for p in pieces],
        "ranges": ranges,
        "stats": {
            "times_s": {k: round(v, 1) for k, v in times.items()},
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "peak_vram_mb": peaks.vram_mb,
            "peak_anon_mb": peaks.anon_mb,
            "peak_torch_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        },
    }
    (out / "plan.json").write_text(json.dumps(plan))
    print("PREPASS " + json.dumps({k: v for k, v in plan.items() if k not in ("tracks", "pieces")}), flush=True)


if __name__ == "__main__":
    main()
