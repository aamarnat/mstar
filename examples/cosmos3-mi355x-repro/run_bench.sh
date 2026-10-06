#!/usr/bin/env bash
# Unprofiled Cosmos3 benchmark on one MI355X: M*, then vLLM-Omni.
# HF_TOKEN must already be set. HF_HOME defaults to $HOME/hf-cache.
# Writes obs.png, mstar-unprofiled.json, and omni-unprofiled.json here.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

PNG_SHA=$(docker run --rm --entrypoint python -v "$PWD":/work \
  cosmos3-mi355x:repro /work/cosmos3_contract.py --out /work/obs.png)
test -n "$PNG_SHA"
echo "$PNG_SHA"

docker rm -f cosmos3-mstar >/dev/null 2>&1 || true
docker run -d --name cosmos3-mstar \
  --privileged --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --network=host \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -e GPU_ARCHS=gfx950 \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN -e HF_HOME=/cache/huggingface \
  -v "$HF_HOME":/cache/huggingface \
  -v "$PWD/cosmos3-droid-aiter-fmha.yaml":/config/droid.yaml:ro \
  --entrypoint mstar cosmos3-mi355x:repro \
  serve cosmos3_droid --config /config/droid.yaml --gpus 0 \
  --cache-dir /cache/huggingface --tensor-comm-protocol SHM \
  --host 127.0.0.1 --port 18080

ready=0
for _ in $(seq 1 180); do
  if curl -sf http://127.0.0.1:18080/health >/dev/null; then
    ready=1
    break
  fi
  sleep 10
done
test "$ready" = 1
docker logs cosmos3-mstar 2>&1 | grep -F "AiterFmhaAttentionManager"
docker logs cosmos3-mstar 2>&1 | grep -F "paged KV writes disabled"

docker run --rm --network=host \
  --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -v "$PWD":/work --entrypoint python cosmos3-mi355x:repro \
  /work/cosmos3_mstar.py \
  --url http://127.0.0.1:18080/generate \
  --image /work/obs.png --image-sha "$PNG_SHA" \
  --out /work/mstar-unprofiled.json \
  --warmups 1 --repeats 5 --timeout 3600

docker rm -f cosmos3-mstar

docker run --rm \
  --privileged --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
  -e GPU_ARCHS=gfx950 \
  -e HF_TOKEN -e HUGGING_FACE_HUB_TOKEN -e HF_HOME=/cache/huggingface \
  -v "$HF_HOME":/cache/huggingface -v "$PWD":/work \
  --entrypoint python cosmos3-mi355x:repro \
  /work/cosmos3_omni.py \
  --image /work/obs.png --image-sha "$PNG_SHA" \
  --out /work/omni-unprofiled.json \
  --warmups 1 --repeats 5
