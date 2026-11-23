#!/usr/bin/env bash
# Launch one Hy4-flash replica (default TP2). On an 8-GPU node run four replicas:
#   GPUS=0,1 PORT=8016 NAME=hy4-0 ./serve.sh   (… GPUS=6,7 PORT=8019 NAME=hy4-3)
#
# KV memory modes (KV=...):
#   offload:<GiB>   GPU-resident KV + CPU LRU prefix tier (default, offload:400)
#   hisparse:<GiB>  HiSparse: GPU hot window + host-resident KV (very long contexts)
#   none            GPU only
set -euo pipefail
TP="${TP:-2}"

MODEL="${MODEL:?set MODEL=/path/to/checkpoint}"
QUANT="${QUANT---quantization compressed-tensors}"   # QUANT= for native FP8 checkpoints
NAME="${NAME:-hy4}"
GPUS="${GPUS:-0,1}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-ghcr.io/jjang-ai/vllm-hy4-lyber:opt3}"
KV="${KV:-offload:400}"
NSPEC="${NSPEC:-3}"            # DFly draft tokens; 0 disables DFly
KVDTYPE="${KVDTYPE:-fp8_ds_mla}"  # KV cache dtype; auto = BF16 (about 2.6x less KV capacity)
MAXLEN="${MAXLEN:-1048576}"    # full native context
MAXSEQS="${MAXSEQS:-64}"      # validated per TP2 replica on B300
HSBUF="${HSBUF:-11008}"        # HiSparse hot rows per request; >= (NSPEC+1)*2176, 256-aligned
CACHE="${CACHE:-$HOME/.cache/vllm-hy4/$NAME}"   # keeps compiled kernels across restarts
mkdir -p "$CACHE"

KV_FLAGS=()
case "$KV" in
  none) ;;
  offload:*)  KV_FLAGS=(--kv-offloading-size "${KV#offload:}" --kv-offloading-backend native) ;;
  hisparse:*) KV_FLAGS=(
      --kv-transfer-config "{\"kv_connector\":\"HiSparseConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"host_pool_gib\":${KV#hisparse:}}}"
      --attention-config "{\"hisparse_config\":{\"device_buffer_size\":${HSBUF}}}") ;;
  *) echo "unknown KV mode: $KV" >&2; exit 1 ;;
esac

KVD_FLAGS=(--kv-cache-dtype "$KVDTYPE")
# FP8 sparse FlashMLA kernels read 64-token blocks; the packed BLHNC layout cannot split larger ones.
[ "$KVDTYPE" = auto ] || KVD_FLAGS+=(--block-size 64)

SPEC_FLAGS=()
if [ "$NSPEC" != "0" ]; then
  SPEC_FLAGS=(--speculative-config "{\"method\":\"dfly\",\"num_speculative_tokens\":${NSPEC},\"draft_sample_method\":\"probabilistic\"}")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "\"device=${GPUS}\"" --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${MODEL}:/model:ro" -v "${CACHE}:/root/.cache" \
  "$IMAGE" \
  --model /model --served-model-name hy4-flash \
  --dtype bfloat16 $QUANT \
  --tensor-parallel-size "$TP" --gpu-memory-utilization 0.92 \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  "${KVD_FLAGS[@]}" "${SPEC_FLAGS[@]}" "${KV_FLAGS[@]}" \
  --enable-prompt-tokens-details \
  --limit-mm-per-prompt '{"image":8,"video":0}' \
  --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice \
  --engram-config '{"cpu_offload":false}' \
  --host 0.0.0.0 --port "$PORT"
echo "started $NAME on GPUs $GPUS port $PORT (KV=$KV, KV dtype=$KVDTYPE, DFly=$NSPEC)"
