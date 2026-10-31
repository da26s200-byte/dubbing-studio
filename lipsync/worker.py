"""Lip-sync worker (phase 3 of lipsync_plan.md): LatentSync 1.5 on each speaking range of plan.json.

Run with the LatentSync venv, through run.sh (RAM cap, free GPU):
    lipsync/run.sh worker <run dir> [--only 0,3]
    lipsync/run.sh worker <run dir> --update <t0>,<t1> --speakers 2   (after one segment was edited)
For each range of lipsync/plan.json: a fixed head crop (from its track's boxes) at 25 fps and that speaker's
dubbed voice, then LatentSync (10 steps, guidance 1.5, DeepCache). Writes lipsync/clips/<range id>.mp4 (the
synced crop, with its audio), lipsync/clips.json, lipsync/progress.json (polled by the app) and, last, the
run's lipsynced.mp4: the kept clips pasted back into dubbed.mp4. The model is loaded once.

Each synced clip is then checked with SyncNet against the same crop unsynced with the same audio: it is kept
only if it scores at least KEEP_MIN and at least KEEP_GAIN more. Otherwise (or if SyncNet finds no face to
score) the range is rejected and stays as in the original video.

--update (after an edited segment's dub changed, with plan.json made again): only the given speakers' ranges
overlapping t0..t1 are synced; the last run's other clips (clips.json) are kept, and the paste-back uses both.
"""

import argparse
import gc
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from math import gcd
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly

from prepass import FFMPEG, FPS, GPU_FRACTION, LATENTSYNC, ORT_GPU_LIMIT, Peaks, frames, probe
from syncscore import load_scorer, unsynced

REPO = LATENTSYNC / "LatentSync"
STEPS = 10  # benchmark: 10 steps cost nothing measurable against 20
GUIDANCE = 1.5  # 1.0 (no guidance) clearly hurt lip sync in the benchmark
SEED = 1247
FACE_TARGET = 220  # px: the face's height in the crop sent to LatentSync
CROP_MARGIN = 1.2  # crop side = the range's union face box + this many face sizes
CROP_MIN, CROP_MAX = 256, 768  # px: the crop's output size, clamped
UPSCALE_MAX = 4.0
SR = 16000  # LatentSync's audio rate
KEEP_MIN = 3.0  # SyncNet confidence a synced clip needs to be kept ...
KEEP_GAIN = 1.5  # ... and its lead over the same crop unsynced


def speaker_tracks(run, chunks, n_samples):
    """Each speaker's dubbed voice alone, at 16 kHz: their units (dub/<id>.wav) placed at the units' starts.
    A unit with several speakers (dubs made before units were split by speaker) goes to each of them."""
    by_id = {c["id"]: c for c in chunks}
    units = {}
    for c in chunks:
        if c.get("dub"):
            units.setdefault(c["dub"]["unit"], set()).add(c["speaker"])
    tracks, mixed = {}, []
    for uid, spks in sorted(units.items()):
        sr, x = wavfile.read(run / "dub" / f"{uid:04d}.wav")
        x = x.astype(np.float32) / (32768.0 if x.dtype == np.int16 else 1.0)
        if x.ndim > 1:
            x = x.mean(axis=1)
        g = gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        at = int(round(by_id[uid]["start"] * SR))
        n = max(min(len(x), n_samples - at), 0)
        for s in spks:
            tracks.setdefault(s, np.zeros(n_samples, np.float32))[at : at + n] += x[:n]
        if len(spks) > 1:
            mixed.append(uid)
    return tracks, mixed


def crop_box(r, track, w, h):
    """One fixed square box for the whole range, and the size it's scaled to."""
    b = np.array(track["boxes"])[r["f0"] - track["f0"] : r["f1"] - track["f0"] + 1]
    face = float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])))
    face_h = float(np.median(b[:, 3] - b[:, 1]))
    ux1, uy1, ux2, uy2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    side = int(min(max(ux2 - ux1, uy2 - uy1) + CROP_MARGIN * face, w, h)) // 2 * 2
    x = int(np.clip((ux1 + ux2) / 2 - side / 2, 0, w - side))
    y = int(np.clip((uy1 + uy2) / 2 - side / 2, 0, h - side))
    size = side * FACE_TARGET / face_h
    size = int(round(min(np.clip(size, CROP_MIN, CROP_MAX), side * UPSCALE_MAX) / 16) * 16)
    return {"x": x, "y": y, "side": side, "size": size, "face": round(face_h, 1)}


def write_crops(video, w, h, ranges, boxes, voices, clip_dir):
    """One decode of the video: each range's crop as <id>_in.mp4 (25 fps) and its audio as <id>.wav."""
    enc = {}
    for i, frame in frames(video, w, h):
        for r in ranges:
            if not r["f0"] <= i <= r["f1"]:
                continue
            b = boxes[r["id"]]
            if r["id"] not in enc:
                enc[r["id"]] = subprocess.Popen(
                    [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{b['size']}x{b['size']}",
                     "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                     str(clip_dir / f"{r['id']}_in.mp4")],
                    stdin=subprocess.PIPE,
                )
            crop = frame[b["y"] : b["y"] + b["side"], b["x"] : b["x"] + b["side"]]
            interp = cv2.INTER_LANCZOS4 if b["size"] > b["side"] else cv2.INTER_AREA
            enc[r["id"]].stdin.write(cv2.resize(crop, (b["size"], b["size"]), interpolation=interp).tobytes())
            if i == r["f1"]:
                p = enc.pop(r["id"])
                p.stdin.close()
                p.wait()
    for p in enc.values():  # ranges cut short by the end of the video
        p.stdin.close()
        p.wait()
    for r in ranges:
        a, n = r["f0"] * SR // FPS, (r["f1"] - r["f0"] + 1) * SR // FPS
        x = voices.get(r["speaker"], np.zeros(0, np.float32))[a : a + n]
        x = np.pad(x, (0, n - len(x)))
        wavfile.write(clip_dir / f"{r['id']}.wav", SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def load_latentsync(work):
    """LatentSync 1.5 as in scripts/inference.py, with: TF32, DeepCache, one shared face detector with a capped
    onnxruntime arena, and the UNet built on the GPU then filled from a memory-mapped checkpoint (the stock
    loader holds the fp32 model and the 5 GB checkpoint in RAM at once)."""
    for name in ("configs", "checkpoints", "latentsync"):
        if not (work / name).exists():
            (work / name).symlink_to(REPO / name)
    os.chdir(work)  # the pipeline writes ./temp and reads relative paths
    sys.path.insert(0, str(REPO))
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler
    from insightface.app import FaceAnalysis
    import latentsync.utils.image_processor as image_processor
    from latentsync.utils.face_detector import FaceDetector, INSIGHTFACE_DETECT_SIZE
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature
    from DeepCache import DeepCacheSDHelper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    detector = FaceDetector.__new__(FaceDetector)
    detector.app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"], root="checkpoints/auxiliary",
        providers=["CUDAExecutionProvider"],
        provider_options=[{"device_id": 0, "gpu_mem_limit": ORT_GPU_LIMIT, "arena_extend_strategy": "kSameAsRequested"}],
    )
    detector.app.prepare(ctx_id=0, det_size=(INSIGHTFACE_DETECT_SIZE, INSIGHTFACE_DETECT_SIZE))
    image_processor.FaceDetector = lambda device="cuda": detector  # ImageProcessor makes one per call otherwise

    config = OmegaConf.load("configs/unet/stage2.yaml")
    dtype = torch.float16
    audio_encoder = Audio2Feature(
        model_path="checkpoints/whisper/tiny.pt", device="cuda",
        num_frames=config.data.num_frames, audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    with torch.device("cuda"):
        unet = UNet3DConditionModel.from_config(OmegaConf.to_container(config.model))
    unet = unet.to(dtype=dtype)
    ckpt = torch.load("checkpoints/latentsync_unet.pt", map_location="cpu", mmap=True, weights_only=True)
    unet.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt
    gc.collect()
    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=DDIMScheduler.from_pretrained("configs")
    ).to("cuda")
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=3, cache_branch_id=0)
    helper.enable()
    return pipeline, config, dtype


class Progress:
    """lipsync/progress.json for the app to poll: the stage, the range being worked on, and seconds of video
    done out of the total (the current range counts in 16-frame batches). Written atomically, at most every
    second unless forced."""

    def __init__(self, path, ranges):
        self.path, self.last = path, 0.0
        self.total = sum((r["f1"] - r["f0"] + 1) / FPS for r in ranges)
        self.state = {"status": "cropping", "range": 0, "ranges": len(ranges), "seconds_done": 0.0,
                      "seconds_total": round(self.total, 2), "eta_s": None}
        self.base, self.t_sync, self.synced = 0.0, None, 0.0
        self.write(force=True)

    def write(self, force=False, **fields):
        self.state.update(fields)
        if force or time.perf_counter() - self.last >= 1.0:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state))
            tmp.replace(self.path)
            self.last = time.perf_counter()

    def batch(self, frac, seconds):
        """The current range is `frac` done; `seconds` is its length. ETA from the rate so far."""
        done = self.base + frac * seconds
        eta = None
        if self.t_sync is not None and done > 0:
            eta = round((time.perf_counter() - self.t_sync) / done * (self.total - done))
        self.write(seconds_done=round(done, 2), eta_s=eta)


FEATHER = 16  # px (at the frame's scale): soft edge of a pasted crop, on sides that aren't the frame's edge
FADE = 4 / FPS  # s: each pasted range fades in and out over 4 frames


def composite(run, ls_dir, kept, w, h):
    """lipsynced.mp4: dubbed.mp4's video with each kept clip scaled back to its box and pasted at its time
    (feathered edges, 4-frame fades), and dubbed.mp4's audio tracks (translated as the default, original)."""
    dubbed, out = run / "dubbed.mp4", run / "lipsynced.mp4"
    if not kept:
        shutil.copyfile(dubbed, out)
        return
    inputs, graph, prev = ["-i", str(dubbed)], [], "0:v"
    for j, c in enumerate(kept):
        b = c["box"]
        mask = np.full((b["side"], b["side"]), 255, np.uint8)
        ramp = np.linspace(0, 255, FEATHER + 2)[1:-1].astype(np.uint8)
        for edge, on in (("left", b["x"] > 0), ("right", b["x"] + b["side"] < w),
                         ("top", b["y"] > 0), ("bottom", b["y"] + b["side"] < h)):
            if not on:
                continue
            if edge == "left":
                mask[:, :FEATHER] = np.minimum(mask[:, :FEATHER], ramp[None, :])
            elif edge == "right":
                mask[:, -FEATHER:] = np.minimum(mask[:, -FEATHER:], ramp[::-1][None, :])
            elif edge == "top":
                mask[:FEATHER, :] = np.minimum(mask[:FEATHER, :], ramp[:, None])
            else:
                mask[-FEATHER:, :] = np.minimum(mask[-FEATHER:, :], ramp[::-1][:, None])
        cv2.imwrite(str(ls_dir / "work" / f"mask_{j}.png"), mask)
        dur = c["end"] - c["start"]
        inputs += ["-i", str(ls_dir / c.get("file", f"clips/{c['id']}.mp4")),
                   "-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(ls_dir / "work" / f"mask_{j}.png")]
        ci, mi = 1 + 2 * j, 2 + 2 * j
        graph += [
            f"[{ci}:v]scale={b['side']}:{b['side']}:flags=lanczos,format=yuva420p[c{j}]",
            f"[{mi}:v]format=gray[m{j}]",
            f"[c{j}][m{j}]alphamerge,fade=t=in:st=0:d={FADE}:alpha=1,"
            f"fade=t=out:st={max(dur - FADE, 0):.3f}:d={FADE}:alpha=1,setpts=PTS-STARTPTS+{c['start']}/TB[l{j}]",
            f"[{prev}][l{j}]overlay={b['x']}:{b['y']}:eof_action=pass[v{j}]",
        ]
        prev = f"v{j}"
    script = ls_dir / "work" / "composite.txt"
    script.write_text(";\n".join(graph))
    err = subprocess.run([FFMPEG, "-hide_banner", "-i", str(dubbed)], capture_output=True, text=True).stderr
    n_audio = err.count("Audio:")
    dispo = ["-metadata:s:a:0", "title=Translated", "-disposition:a:0", "default"]
    if n_audio > 1:
        dispo += ["-metadata:s:a:1", "title=Original", "-disposition:a:1", "0"]
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", *inputs, "-filter_complex_script", str(script), "-map", f"[{prev}]", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", *dispo,
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def update_ranges(ls_dir, clip_dir, ranges, span, speakers):
    """--update: the new plan's ranges to sync (the speakers' ranges overlapping the span, widened until no
    old clip of theirs is cut in two) and the last run's clips that stay (all others). The kept ones are
    renamed, since the new plan's range ids start at 0 again, and clips.json is rewritten at once, so a
    failed update can be run again."""
    lo, hi = map(float, span.split(","))
    spk = {int(s) for s in speakers.split(",")}
    prev = json.loads((ls_dir / "clips.json").read_text())
    hit = lambda x: x["speaker"] in spk and x["start"] < hi and x["end"] > lo
    while True:
        both = [r for r in ranges if hit(r)] + [c for c in prev["clips"] if hit(c)]
        a, b = min([lo, *(x["start"] for x in both)]), max([hi, *(x["end"] for x in both)])
        if (a, b) == (lo, hi):
            break
        lo, hi = a, b
    todo = [r for r in ranges if hit(r)]
    old = [c for c in prev["clips"] if not hit(c)]
    for c in prev["clips"]:
        if c["status"] != "kept":
            continue
        if hit(c):
            if "file" in c:  # renamed by an earlier update; the new plan's clips never reuse its name
                (ls_dir / c["file"]).unlink(missing_ok=True)
        elif "file" not in c:
            name = f"keep{int(time.time())}_{c['id']}.mp4"
            (clip_dir / f"{c['id']}.mp4").rename(clip_dir / name)
            c["file"] = f"clips/{name}"
    (ls_dir / "clips.json").write_text(json.dumps(prev, indent=1))
    print(f"UPDATE {lo:.2f}-{hi:.2f}s speakers {sorted(spk)}: {len(todo)} range(s) to sync, {len(old)} kept as they were",
          flush=True)
    return todo, old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--only", help="comma-separated range ids")
    ap.add_argument("--composite-only", action="store_true", help="redo only the paste-back from clips.json")
    ap.add_argument("--update", help="t0,t1 (s): redo only --speakers' ranges overlapping this span, keep the other clips")
    ap.add_argument("--speakers", help="comma-separated speakers, for --update")
    args = ap.parse_args()
    run = Path(args.run_dir).resolve()
    ls_dir = run / "lipsync"
    clip_dir, work = ls_dir / "clips", ls_dir / "work"
    clip_dir.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    video = run / "input.mp4"
    w, h, _ = probe(video)
    if args.composite_only:
        done = json.loads((ls_dir / "clips.json").read_text())
        s = time.perf_counter()
        composite(run, ls_dir, [c for c in done["clips"] if c["status"] == "kept"], w, h)
        print(f"COMPOSITE {time.perf_counter() - s:.1f}s", flush=True)
        return

    torch.cuda.set_per_process_memory_fraction(GPU_FRACTION)
    peaks, t0 = Peaks(), time.perf_counter()
    from accelerate.utils import set_seed

    plan = json.loads((ls_dir / "plan.json").read_text())
    chunks = json.loads((run / "chunks.json").read_text())
    ranges = plan["ranges"]
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        ranges = [r for r in ranges if r["id"] in keep]
    old = []  # --update: the last run's clips that stay
    if args.update:
        ranges, old = update_ranges(ls_dir, clip_dir, ranges, args.update, args.speakers)
    tracks = {t["id"]: t for t in plan["tracks"]}
    progress = Progress(ls_dir / "progress.json", ranges)

    s = time.perf_counter()
    voices, mixed = speaker_tracks(run, chunks, int(np.ceil(plan["frames"] / FPS * SR)) + SR)
    boxes = {r["id"]: crop_box(r, tracks[r["track"]], w, h) for r in ranges}
    write_crops(video, w, h, ranges, boxes, voices, clip_dir)
    del voices
    t_crops = time.perf_counter() - s

    progress.write(force=True, status="loading")
    s = time.perf_counter()
    # no ranges (no speaking face on screen): no model; lipsynced.mp4 is then the dub as it is
    pipeline, config, dtype = load_latentsync(work) if ranges else (None, None, None)
    score = load_scorer(work) if ranges else None
    t_load = time.perf_counter() - s

    # LatentSync decodes each 16-frame batch once: count them for progress inside a range
    decode, current = pipeline.decode_latents if pipeline is not None else None, {"done": 0, "batches": 1, "seconds": 0.0}

    def counted(*a, **k):
        out = decode(*a, **k)
        current["done"] += 1
        progress.batch(min(current["done"] / current["batches"], 1.0), current["seconds"])
        return out

    if pipeline is not None:
        pipeline.decode_latents = counted
    progress.t_sync = time.perf_counter()

    results = []
    for k, r in enumerate(ranges):
        n = r["f1"] - r["f0"] + 1
        current.update(done=0, batches=int(np.ceil(n / config.data.num_frames)), seconds=n / FPS)
        progress.write(force=True, status="lipsyncing", range=k + 1)
        rec = {**{x: r[x] for x in ("id", "speaker", "track", "start", "end")}, "box": boxes[r["id"]]}
        set_seed(SEED)
        torch.cuda.reset_peak_memory_stats()
        s = time.perf_counter()
        try:
            pipeline(
                video_path=str(clip_dir / f"{r['id']}_in.mp4"), audio_path=str(clip_dir / f"{r['id']}.wav"),
                video_out_path=str(clip_dir / f"{r['id']}.mp4"), num_frames=config.data.num_frames,
                num_inference_steps=STEPS, guidance_scale=GUIDANCE, weight_dtype=dtype,
                width=config.data.resolution, height=config.data.resolution,
                mask_image_path=config.data.mask_image_path,
            )
            rec["status"] = "synced"
        except RuntimeError as exc:
            if "Face not detected" not in str(exc):
                raise
            rec.update(status="skipped", error=str(exc))  # this range stays as in the original
        rec["seconds"] = round(time.perf_counter() - s, 1)
        if rec["status"] == "synced":
            progress.write(force=True, status="checking")
            torch.cuda.empty_cache()  # LatentSync's cached blocks would otherwise stack with SyncNet's
            s = time.perf_counter()
            synced = score(clip_dir / f"{r['id']}.mp4")
            before = score(unsynced(clip_dir, r["id"], work / f"ref_{r['id']}.mp4"))
            ok = synced is not None and synced >= KEEP_MIN and synced - (before or 0.0) >= KEEP_GAIN
            rec.update(status="kept" if ok else "rejected", sync_score=synced, unsynced_score=before,
                       check_seconds=round(time.perf_counter() - s, 1))
        rec.update(
            torch_peak_mb=round(torch.cuda.max_memory_allocated() / 2**20),
            rss_peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        )
        results.append(rec)
        print("RANGE " + json.dumps(rec), flush=True)
        progress.base += n / FPS
        progress.batch(0.0, 0.0)
        gc.collect()
        torch.cuda.empty_cache()

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    progress.write(force=True, status="compositing", eta_s=None)
    s = time.perf_counter()
    clips = old + results  # --update: with the last run's clips that stay
    composite(run, ls_dir, [x for x in clips if x["status"] == "kept"], w, h)
    t_composite = time.perf_counter() - s

    peaks.stop()
    summary = {
        "steps": STEPS, "guidance": GUIDANCE, "mixed_speaker_units": mixed,
        "keep_min": KEEP_MIN, "keep_gain": KEEP_GAIN,
        "times_s": {"crops": round(t_crops, 1), "load": round(t_load, 1),
                    "lipsync": round(sum(x["seconds"] for x in results), 1),
                    "check": round(sum(x.get("check_seconds", 0) for x in results), 1),
                    "composite": round(t_composite, 1), "total": round(time.perf_counter() - t0, 1)},
        "ranges": len(clips),
        "kept": [x["id"] for x in clips if x["status"] == "kept"],
        "rejected": [x["id"] for x in clips if x["status"] == "rejected"],
        "skipped": [x["id"] for x in clips if x["status"] == "skipped"],
        "range_seconds": round(sum(x["end"] - x["start"] for x in clips), 2),
        "kept_seconds": round(sum(x["end"] - x["start"] for x in clips if x["status"] == "kept"), 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "peak_anon_mb": peaks.anon_mb,
        "peak_vram_mb": peaks.vram_mb,
    }
    (ls_dir / "clips.json").write_text(json.dumps({"summary": summary, "clips": clips}, indent=1))
    progress.write(force=True, status="done", seconds_done=round(progress.total, 2), eta_s=0)
    print("WORKER " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
