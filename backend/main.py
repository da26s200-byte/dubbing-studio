import mimetypes
import re
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from mel_band_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config
from mel_band_roformer.attend import Attend, FlashAttentionConfig
from mel_band_roformer.inference import SafeLoaderWithTuple
from ml_collections import ConfigDict

try:
    from . import asr, samples, studio, translate, tts
except ImportError:
    import asr
    import samples
    import studio
    import translate
    import tts

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEPARATOR_MODEL = DEFAULT_MODEL  # melband-roformer-kim-vocals
# second pass on the vocals stem: kim files cheering, laughter and part of the clapping under vocals, so they'd
# be replaced by the dub; this model pulls that crowd sound out of the vocals and it goes back into bg_noise
CROWD_MODEL = "roformer-model-mel-roformer-crowd-aufr33-viperx"
CROWD_SEPARATION = True  # False: vocals.wav is kim's output as is (A/B tests)
CROWD_OVERLAP = 2  # windows averaged per sample; its config says 4, which doubles the pass for little gain (kim uses 2)
SEPARATOR_BATCH_SIZE = 4  # 8 s chunks processed per forward pass (keeps peak VRAM < 5 GB)

# TF32 for float32 matrix multiplies: faster on Ampere and newer GPUs, with a negligible effect on
# the output. Convolutions already use it by default.
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load Mel-Band RoFormer in the background, so the first upload doesn't pay for it and the server is up at once
    threading.Thread(target=_warm_up_separator, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_separators: dict[str, tuple[torch.nn.Module, ConfigDict]] = {}
_separator_lock = threading.Lock()  # a request arriving during the warm-up waits for it, not a second load


def get_separator(name: str = SEPARATOR_MODEL) -> tuple[torch.nn.Module, ConfigDict]:
    """(model, config) for a Mel-Band RoFormer checkpoint (SEPARATOR_MODEL or CROWD_MODEL), loaded once onto
    DEVICE with SDPA flash attention enabled on all modern GPUs."""
    with _separator_lock:
        if name not in _separators:
            ckpt_path, cfg_path = ensure_model_assets(name, models_dir=MODELS_DIR, download_missing=False)
            with open(cfg_path) as f:
                config = ConfigDict(yaml.load(f, Loader=SafeLoaderWithTuple))
            config.model.flash_attn = True
            if name == CROWD_MODEL:
                config.inference.num_overlap = CROWD_OVERLAP
            model = get_model_from_config("mel_band_roformer", config)
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            # mel_band_roformer.attend only enables flash_sdp on sm_80 (A100); enable all SDPA backends
            # so Ada/Blackwell/Ampere GPUs use hardware FlashAttention-2.
            if DEVICE == "cuda":
                for m in model.modules():
                    if isinstance(m, Attend) and m.flash:
                        m.cuda_config = FlashAttentionConfig(True, True, True)
            # under bf16 autocast the mask estimators return bf16, and the model's torch.view_as_complex
            # has no bf16 kernel; cast their output to float32 so the mask/iSTFT tail runs in full precision.
            for est in model.mask_estimators:
                est.register_forward_hook(lambda _m, _inp, out: out.float())
            model = model.to(DEVICE).eval()
            _separators[name] = (model, config)
        return _separators[name]


def _demix(model: torch.nn.Module, config: ConfigDict, mix: torch.Tensor) -> torch.Tensor:
    """Batched overlap-add inference for Mel-Band RoFormer on stereo float32 `mix` of shape (2, T).
    Returns the model's first stem (vocals for kim, crowd for the crowd model) of shape (2, T) on CPU."""
    C = getattr(config.inference, "chunk_size", None) or getattr(getattr(config, "audio", None), "chunk_size", 352800)
    N = config.inference.num_overlap
    step = C // N
    fade_size = C // 10
    border = C - step

    orig_len = mix.shape[1]
    padded_border = orig_len > 2 * border and border > 0
    if padded_border:
        mix = F.pad(mix, (border, border), mode="reflect")

    total_length = mix.shape[1]
    offsets = list(range(0, total_length, step))

    fadein = torch.linspace(0, 1, fade_size, device=DEVICE)
    fadeout = torch.linspace(1, 0, fade_size, device=DEVICE)
    base_win = torch.ones(C, device=DEVICE)
    base_win[:fade_size] *= fadein
    base_win[-fade_size:] *= fadeout

    use_amp = DEVICE == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.inference_mode(), torch.amp.autocast(DEVICE, dtype=amp_dtype, enabled=use_amp):
        mix_dev = mix.to(DEVICE)
        result = torch.zeros((2, total_length), dtype=torch.float32, device=DEVICE)
        counter = torch.zeros(total_length, dtype=torch.float32, device=DEVICE)

        for b_start in range(0, len(offsets), SEPARATOR_BATCH_SIZE):
            batch_offsets = offsets[b_start : b_start + SEPARATOR_BATCH_SIZE]
            parts, lengths = [], []
            for i in batch_offsets:
                part = mix_dev[:, i : i + C]
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = F.pad(part, (0, C - length), mode="reflect")
                    else:
                        part = F.pad(part, (0, C - length), mode="constant", value=0)
                parts.append(part)
                lengths.append(length)

            preds = model(torch.stack(parts, dim=0)).float()
            if preds.ndim == 4:  # multi-stem checkpoints return (B, S, C, T)
                preds = preds[:, 0]

            for idx, (i, length) in enumerate(zip(batch_offsets, lengths)):
                win = base_win
                if i == 0 or i + C >= total_length:
                    win = base_win.clone()
                    if i == 0:
                        win[:fade_size] = 1.0
                    if i + C >= total_length:
                        win[-fade_size:] = 1.0
                w = win[:length]
                result[:, i : i + length] += preds[idx, :, :length] * w
                counter[i : i + length] += w

        vocals = (result / counter.clamp(min=1e-8)).cpu()
        if padded_border:
            vocals = vocals[:, border : border + orig_len]
        return torch.nan_to_num(vocals, nan=0.0)


def _warm_up_separator():
    """Load Mel-Band RoFormer (and the crowd model) and run each once on one window of noise, so the model
    load, CUDA initialisation and first-use kernel setup are done before the first video."""
    start = time.perf_counter()
    try:
        for name in [SEPARATOR_MODEL] + ([CROWD_MODEL] if CROWD_SEPARATION else []):
            model, config = get_separator(name)
            sr = getattr(config.model, "sample_rate", 44100)
            _demix(model, config, torch.randn(2, sr * 8) * 0.05)
        print(f"Mel-Band RoFormer warm-up done in {time.perf_counter() - start:.1f}s", flush=True)
    except Exception as exc:  # not fatal: the first request then loads it as before
        print(f"Mel-Band RoFormer warm-up failed: {exc}", flush=True)


@app.get("/languages")
async def languages():
    return {
        "asr": [{"code": code, "name": name} for code, name in asr.LANGUAGES.items()],
        "translation": [{"code": code, "name": name} for code, name in translate.LANGUAGES.items()],
    }


@app.get("/tts-options")
async def tts_options():
    return tts.options()


@app.get("/tts-sample/{lang}/{voice}.wav")
def tts_sample(lang: str, voice: str):
    """`voice` reading the sample sentence in `lang` (samples.py; made the first time it's asked for)."""
    try:
        path = samples.clip(lang, voice)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(502, f"could not make the sample: {exc}")
    return FileResponse(path, media_type="audio/wav")


def _trim(src: Path, dst: Path, start: float, end: float | None):
    """Cut [start, end] out of the upload. Re-encoded, so the cut is exact rather than on a keyframe."""
    cmd = [FFMPEG_BIN, "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += [
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_audio(video_path: Path, jdir: Path) -> Path:
    """The video's audio as audio.wav, stereo 44.1 kHz."""
    audio_path = jdir / "audio.wav"
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "2",
            "-ar",
            "44100",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )
    return audio_path


def separate(audio_path: Path, jdir: Path) -> Path:
    """Mel-Band RoFormer: vocals.wav, and bg_noise.wav as (audio - vocals). With CROWD_SEPARATION, the crowd
    model then splits kim's vocals: its crowd stem (crowd.wav) is taken out of vocals.wav and so stays in
    bg_noise.wav. Returns the vocals path."""
    model, config = get_separator()
    mix_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    if mix_np.shape[1] == 1:
        mix_np = np.repeat(mix_np, 2, axis=1)
    elif mix_np.shape[1] > 2:
        mix_np = mix_np[:, :2]

    mix = torch.from_numpy(mix_np.T)
    vocals = _demix(model, config, mix)
    if CROWD_SEPARATION:
        crowd_model, crowd_config = get_separator(CROWD_MODEL)
        crowd = _demix(crowd_model, crowd_config, vocals)
        vocals = vocals - crowd
        sf.write(str(jdir / "crowd.wav"), crowd.numpy().T, sr, subtype="PCM_16")
    vocals = vocals.numpy().T
    bg = mix_np - vocals

    vocals_path = jdir / "vocals.wav"
    bg_path = jdir / "bg_noise.wav"
    sf.write(str(vocals_path), vocals, sr, subtype="PCM_16")
    sf.write(str(bg_path), bg, sr, subtype="PCM_16")
    return vocals_path


RUN_ID = re.compile(r"^[0-9a-f]{32}$")


def job_dir(job_id: str) -> Path:
    if not RUN_ID.match(job_id):
        raise HTTPException(status_code=404, detail="file not found")
    return STORAGE_DIR / job_id


@app.get("/download/{job_id}/{stem}")
async def download(job_id: str, stem: str):
    """A studio run's files: the video it works on, its stems and its dub."""
    if stem == "input":  # the video the pipeline works on (the trimmed portion if one was chosen)
        path = next(job_dir(job_id).glob("input.*"), None)
        if path is None:
            raise HTTPException(status_code=404, detail="file not found")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type, filename=path.name)
    if stem in ("vocals", "bg_noise", "dubbed_audio", "dubbed_vocals"):
        filename, media_type = f"{stem}.wav", "audio/wav"
    elif stem == "dubbed_video":
        filename, media_type = "dubbed.mp4", "video/mp4"
    elif stem == "lipsynced_video":
        filename, media_type = "lipsynced.mp4", "video/mp4"
    elif stem in ("diarization", "chunks", "pauses"):
        filename, media_type = f"{stem}.json", "application/json"
    else:
        raise HTTPException(status_code=400, detail="invalid stem")
    path = job_dir(job_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type=media_type, filename=filename)


# --- Studio flow (/v2): upload → diarize, then ASR on request, with past runs ---

studio.configure(STORAGE_DIR, FFMPEG_BIN, trim=_trim, extract_audio=extract_audio, separate=separate)
app.include_router(studio.router)


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "ui", html=True), name="ui")
