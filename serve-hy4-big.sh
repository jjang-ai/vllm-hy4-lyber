#!/usr/bin/env bash
# Launch one Hy4 (big) replica on all 8 GPUs (TP8), e.g. lyber-ai/Hy4-UNCENSORED-NVFP4 (NVFP4 routed experts) or an
# FP8 / MXFP4 Hy4 checkpoint. This is the configuration validated in production on 8x B300 (README "Hy4 (big)").
#
#   KV: FP8 (fp8_ds_mla, 64-token blocks) GPU prefix cache + CPU LRU tier (OFFLOAD GiB in /dev/shm). Blocks evicted from
#       the GPU stay in CPU RAM and are swapped back on a prefix hit; the CPU tier evicts least-recently-used blocks.
#   DFly2 speculative decoding, image input (1 per request, the model's limit), tools, reasoning parser hy_v4.
#   Sampling defaults come from the checkpoint's generation_config.json (Tencent: temperature 0.9, top_p 1.0).
#   1M context; a request without max_tokens may generate up to the remaining context.
#
#   MODEL=/path/to/Hy4-UNCENSORED-NVFP4 PORT=8031 NAME=hy4-big ./serve-hy4-big.sh
#   API key: put VLLM_API_KEY=... in a 0600 file and pass KEYFILE=/path/to/file (recommended for anything reachable).
set -euo pipefail
MODEL="${MODEL:?set MODEL=/path/to/checkpoint}"
NAME="${NAME:-hy4-big}"
PORT="${PORT:-8031}"
BIND="${BIND:-127.0.0.1}"            # publish on loopback; front it with a tunnel / reverse proxy
SERVED="${SERVED:-hy4-big}"           # space-separated served model names
IMAGE="${IMAGE:-ghcr.io/jjang-ai/vllm-hy4-lyber:opt4}"
NSPEC="${NSPEC:-5}"                   # DFly draft tokens (Tencent's setting); 0 disables DFly
OFFLOAD="${OFFLOAD:-480}"             # CPU tier GiB (pinned). Keep it above the GPU KV pool (~180 GiB/rank FP8) or it adds
                                      # nothing, and well below free RAM: 960 GiB on a 2 TB host caused reclaim stalls. 0 = GPU only
KVDTYPE="${KVDTYPE:-fp8_ds_mla}"      # auto = BF16 KV (about 1.7x fewer GPU KV tokens)
MAXLEN="${MAXLEN:-1048576}"
MAXSEQS="${MAXSEQS:-64}"
NVFP4_PER_TOKEN="${NVFP4_PER_TOKEN:-1}"   # NVFP4 checkpoints: W4A4 with per-token activation scales (no calibration)
CACHE="${CACHE:-$HOME/.cache/vllm-hy4/$NAME}"
KEYFILE="${KEYFILE:-}"
mkdir -p "$CACHE"
# Pinning the CPU tier must not fight a huge page cache (e.g. the checkpoint just copied/verified): drop it first.
[ "$OFFLOAD" = 0 ] || { sync; echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true; }

KV_FLAGS=(--kv-cache-dtype "$KVDTYPE")
[ "$KVDTYPE" = auto ] || KV_FLAGS+=(--block-size 64)   # FP8 sparse FlashMLA reads 64-token blocks
[ "$OFFLOAD" = 0 ] || KV_FLAGS+=(--kv-offloading-size "$OFFLOAD" --kv-offloading-backend native)
SPEC_FLAGS=()
[ "$NSPEC" = 0 ] || SPEC_FLAGS=(--speculative-config "{\"method\":\"dfly\",\"num_speculative_tokens\":${NSPEC}}")
ENV_FLAGS=(-e VLLM_USE_RUST_FRONTEND=1 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e HY4_NVFP4_PER_TOKEN="$NVFP4_PER_TOKEN")
[ -z "$KEYFILE" ] || ENV_FLAGS+=(--env-file "$KEYFILE")

# shellcheck disable=SC2086
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --restart unless-stopped --init \
  --gpus all --privileged --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p "${BIND}:${PORT}:${PORT}" "${ENV_FLAGS[@]}" \
  -v "${MODEL}:/model:ro" -v "${CACHE}:/root/.cache" \
  "$IMAGE" /model \
    --served-model-name $SERVED \
    --dtype bfloat16 --tensor-parallel-size 8 \
    --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
    "${KV_FLAGS[@]}" "${SPEC_FLAGS[@]}" \
    --attention-backend FLASHMLA_SPARSE \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --enable-prompt-tokens-details \
    --generation-config auto \
    --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice \
    --host 0.0.0.0 --port "$PORT" >/dev/null
echo "started $NAME on ${BIND}:${PORT} (KV=$KVDTYPE, CPU tier=${OFFLOAD} GiB, DFly=$NSPEC, image=$IMAGE)"
