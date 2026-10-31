cd "$(dirname "$0")"
source venv/bin/activate
# Backend models (separation, ASR, diarization, gender) go on the GPU with the most free memory at startup
# (re-picked on every r reload). Set CUDA_VISIBLE_DEVICES yourself to override.
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -nr | head -1 | cut -d, -f1
}
# token from the environment, else the "hf - " line of keys.txt (gitignored)
export HF_TOKEN=${HF_TOKEN:-$(sed -n 's/^hf - //p' keys.txt 2>/dev/null)}

# r restarts the server (picks up code changes; a run in progress is interrupted), Ctrl+C stops it
GPU_PINNED=${CUDA_VISIBLE_DEVICES:-}
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null; exit' INT TERM
while true; do
  if [ -z "${GPU_PINNED:-}" ]; then
    GPU=$(pick_gpu)
    [ -n "$GPU" ] && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU && echo "[backend on GPU $GPU]"
  fi
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
  pid=$!
  echo "[press r to reload]"
  if [ -t 0 ]; then
    while read -rsn1 key; do [ "$key" = r ] && break; done
  else
    wait $pid; exit
  fi
  echo "[reloading]"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
done
