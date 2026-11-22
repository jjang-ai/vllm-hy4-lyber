#!/usr/bin/env bash
# Example launch script for vllm-hy4-lyber.
#
# Edit MODEL and (optionally) GPUS / PORT for your setup.  The image
# already contains the two compressed-tensors MXFP4 MoE patches; no
# DeepFly / DeepFlash flag is needed because DeepFly auto-enables from
# the checkpoint's config.json.
set -euo pipefail

MODEL="${MODEL:-/model}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
IMAGE="${IMAGE:-ghcr.io/jjang-ai/vllm-hy4-lyber:v6}"

docker run --rm --gpus all --ipc=host --shm-size=16g \
    -p "${PORT}:8000" \
    -v "${MODEL}:/model:ro" \
    "${IMAGE}" \
    --model /model \
    --served-model-name hy4-flash \
    --tensor-parallel-size "${TP}" \
    --quantization compressed-tensors \
    --max-model-len "${MAX_MODEL_LEN}" \
    --trust-remote-code \
    --disable-log-requests
