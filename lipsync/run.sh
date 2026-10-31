#!/bin/bash
# Run one lip-sync step for one run, with a hard RAM cap and only a free GPU.
#   lipsync/run.sh faces <run dir> [video name]    (at diarization, see faces.py)
#   lipsync/run.sh prepass <run dir> [--debug]
#   lipsync/run.sh worker <run dir> [--only 0,3]
#   lipsync/run.sh syncscore <run dir>
# RAM: a systemd scope with MemoryMax (the kernel kills the job, not the machine, if it's exceeded).
# GPU: the first one with no compute processes on it (other than the app's backend) and at least 8 GB free; refuses to start otherwise.
set -euo pipefail
VENV=/data1/pratyaymishra_data/latentsync/venv
HERE=$(cd "$(dirname "$0")" && pwd)
MEM_MAX=${LIPSYNC_MEM_MAX:-12G}
STEP=$1
shift
[ -f "$HERE/$STEP.py" ] || { echo "unknown step: $STEP" >&2; exit 2; }

# started from the app, the environment may lack the user's systemd bus
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

# the app's own backend (our parent process) doesn't count as busy: lip-sync may share its GPU
BUSY=$(nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', ' -v me="$PPID" '$1 != me {print $2}' | sort -u)
GPU=""
while IFS=', ' read -r idx bus free; do
  if ! grep -qiF "$bus" <<<"$BUSY" && [ "$free" -ge 8000 ]; then GPU=$idx; break; fi
done < <(nvidia-smi --query-gpu=index,pci.bus_id,memory.free --format=csv,noheader,nounits)
[ -n "$GPU" ] || { echo "no free GPU (every GPU has other processes or < 8 GB free)" >&2; exit 1; }
echo "$STEP: GPU $GPU, RAM cap $MEM_MAX" >&2

exec systemd-run --user --scope --quiet -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 \
  env PATH="$VENV/bin:$PATH" CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 \
  /usr/bin/time -v "$VENV/bin/python" "$HERE/$STEP.py" "$@"
