#!/bin/bash
# Sets up everything Dubbing Studio needs on a fresh Linux + NVIDIA machine, from a clone of the repo:
#   1. backend venv (venv/) from requirements.txt
#   2. backend model weights into models/ (RoFormer vocals + crowd, wav2vec2 age-gender) and the Nemotron
#      diarization weights into the Hugging Face cache
#   3. lip-sync / face-analysis environment in $LATENTSYNC_DIR (default third_party/latentsync): LatentSync 1.5 and
#      Light-ASD repos at the commits the app was built against, their venv from lipsync/requirements.txt, and
#      their checkpoints (LatentSync UNet, Whisper-tiny, SyncNet, S3FD, InsightFace buffalo_l, SD VAE)
# Every step skips what is already there, so it is safe to re-run.
#
#   ./setup.sh                  everything
#   ./setup.sh --no-lipsync     backend only (faces step and lip-sync will then be unavailable)
#   ./setup.sh --no-models      environments only (e.g. models come from a mounted volume)
#   ./setup.sh --models-only    weights only, into existing environments
#
# Env: PYTHON (default python3.10), LATENTSYNC_DIR, HF_TOKEN (or the "hf - <token>" line of keys.txt), HF_HOME.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)

LIPSYNC=1
MODELS=1
ENVS=1
for arg in "$@"; do
  case $arg in
    --no-lipsync) LIPSYNC=0 ;;
    --no-models) MODELS=0 ;;
    --models-only) ENVS=0 ;;
    -h|--help) sed -n 2,17p "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

PYTHON=${PYTHON:-python3.10}
LATENTSYNC_DIR=${LATENTSYNC_DIR:-$ROOT/third_party/latentsync}
LATENTSYNC_REPO=https://github.com/bytedance/LatentSync.git
LATENTSYNC_COMMIT=7b380d6adde7aa23cba4a53bb15cd5cfb3c168e7
LIGHT_ASD_REPO=https://github.com/Junhua-Liao/Light-ASD.git
LIGHT_ASD_COMMIT=ed38c232de5efe0261dbd68627c0ade7cdfe14eb
export HF_TOKEN=${HF_TOKEN:-$(sed -n 's/^hf - //p' keys.txt 2>/dev/null)}

step() { echo; echo "==> $*"; }
warn() { echo "WARNING: $*" >&2; }

# --- 0. system checks ---------------------------------------------------------------------------------------------
step "checking system"
command -v "$PYTHON" >/dev/null || { echo "$PYTHON not found (set PYTHON=...)" >&2; exit 1; }
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), sys.version' \
  || { echo "$PYTHON must be Python 3.10" >&2; exit 1; }
"$PYTHON" -m venv --help >/dev/null 2>&1 || { echo "python venv module missing (apt install python3.10-venv)" >&2; exit 1; }
for f in bodhan.txt keys.txt; do
  [ -f $f ] || warn "$f missing: copy $f.example to $f and fill it in (or use the env vars it lists)"
done
command -v git >/dev/null || { echo "git not found" >&2; exit 1; }
command -v ffmpeg >/dev/null || warn "ffmpeg not on PATH"
command -v nvidia-smi >/dev/null || warn "nvidia-smi not found: no GPU visible (the app needs NVIDIA GPUs)"
[ -n "$HF_TOKEN" ] || warn "no HF_TOKEN (env or keys.txt): Nemotron diarization weights can't be fetched"

# --- 1. backend venv ----------------------------------------------------------------------------------------------
step "backend venv: $ROOT/venv"
if [ "$ENVS" = 1 ]; then
  [ -x venv/bin/python ] || "$PYTHON" -m venv venv
  venv/bin/pip install -q --upgrade pip wheel
  venv/bin/pip install -r requirements.txt
fi

# --- 2. backend models ------------------------------------------------------------------------------------------
if [ "$MODELS" = 1 ]; then
  step "backend models: $ROOT/models"
  mkdir -p models
  venv/bin/python - <<'EOF'
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from mel_band_roformer.download import ensure_model_assets

models = Path("models").resolve()
for name in ("melband-roformer-kim-vocals", "roformer-model-mel-roformer-crowd-aufr33-viperx"):
    ckpt, _ = ensure_model_assets(name, models_dir=models, download_missing=True)
    print("ok", ckpt)

gender = models / "wav2vec2-age-gender"
if not (gender / "config.json").exists():
    snapshot_download("audeering/wav2vec2-large-robust-24-ft-age-gender", local_dir=gender)
print("ok", gender)

# diarization loads it by name from the HF cache (HF_HOME), so it is cached, not put in models/
try:
    print("ok", snapshot_download("nvidia/Nemotron-3-Diarization", token=os.environ.get("HF_TOKEN") or None))
except Exception as e:
    print(f"WARNING: Nemotron-3-Diarization not downloaded ({e}); it will be fetched on first use with HF_TOKEN")
EOF
fi

# --- 3. lip-sync environment --------------------------------------------------------------------------------------
if [ "$LIPSYNC" = 1 ]; then
  step "lip-sync environment: $LATENTSYNC_DIR"
  mkdir -p "$LATENTSYNC_DIR"
  clone() {  # clone <url> <dir> <commit>
    [ -d "$2/.git" ] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
  }
  clone "$LATENTSYNC_REPO" "$LATENTSYNC_DIR/LatentSync" "$LATENTSYNC_COMMIT"
  clone "$LIGHT_ASD_REPO" "$LATENTSYNC_DIR/Light-ASD" "$LIGHT_ASD_COMMIT"  # weights are in the repo (weight/)

  VENV=$LATENTSYNC_DIR/venv
  if [ "$ENVS" = 1 ]; then
    [ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip wheel
    "$VENV/bin/pip" install -r lipsync/requirements.txt
  fi

  if [ "$MODELS" = 1 ]; then
    step "lip-sync checkpoints: $LATENTSYNC_DIR/LatentSync/checkpoints"
    "$VENV/bin/python" - "$LATENTSYNC_DIR/LatentSync/checkpoints" <<'EOF'
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from insightface.utils.storage import ensure_available

ckpt = Path(sys.argv[1])
for f in ("latentsync_unet.pt", "whisper/tiny.pt", "auxiliary/syncnet_v2.model",
          "auxiliary/s3fd-619a316812.pth", "auxiliary/sfd_face.pth"):
    if not (ckpt / f).exists():
        hf_hub_download("ByteDance/LatentSync-1.5", f, local_dir=ckpt)
    print("ok", ckpt / f)

# InsightFace detector / landmarks / ArcFace (prepass.py and worker.py load it from checkpoints/auxiliary)
print("ok", ensure_available("models", "buffalo_l", root=str(ckpt / "auxiliary")))
# LatentSync's VAE; lip-sync runs with HF_HUB_OFFLINE=1, so it must be in the HF cache beforehand
print("ok", snapshot_download("stabilityai/sd-vae-ft-mse"))
EOF
  fi
fi

step "done"
echo "Copy bodhan.txt.example -> bodhan.txt and keys.txt.example -> keys.txt and fill in your keys, then: ./run_app.sh"
