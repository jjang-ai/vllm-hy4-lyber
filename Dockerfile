# vllm-hy4-lyber: pinned vLLM runtime for Hy4-flash FP8 / FP8 + MXFP4 checkpoints.
#
# Base: upstream vLLM nightly f8b29fab (2026-11-21). Adds:
#  - DFly speculative decoding together with local KV connectors (CPU offload, HiSparse)
#  - bit-exact fused decode kernels for Hy4-flash (RMSNorm, M-RoPE, indexer, Engram)
#  - FP8 (fp8_ds_mla) KV cache on all layers, including the sliding-window layers
#  - CPU KV offload tier that actually retains evicted prefixes for Hy4-flash
# See README.md and patches/*.diff.

ARG BASE=vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b
FROM ${BASE}

ARG VLLM_PY=/usr/local/lib/python3.14/dist-packages/vllm

COPY patches/models_hy_v4_flash_nvidia_model.py              ${VLLM_PY}/models/hy_v4_flash/nvidia/model.py
COPY patches/models_hy_v4_flash_nvidia_fused_ops.py          ${VLLM_PY}/models/hy_v4_flash/nvidia/fused_ops.py
COPY patches/models_hy_v4_flash_nvidia_attention.py          ${VLLM_PY}/models/hy_v4_flash/nvidia/attention.py
COPY patches/models_hy_v4_flash_nvidia_swa.py                ${VLLM_PY}/models/hy_v4_flash/nvidia/swa.py
COPY patches/models_hy_v4_flash_nvidia_compressed_indexer.py ${VLLM_PY}/models/hy_v4_flash/nvidia/compressed_indexer.py
COPY patches/models_hy_v4_flash_nvidia_engram.py             ${VLLM_PY}/models/hy_v4_flash/nvidia/engram.py
COPY patches/models_hy_v4_nvidia_moe.py                      ${VLLM_PY}/models/hy_v4/nvidia/moe.py
COPY patches/models_hy_v4_nvidia_flashmla_sparse.py          ${VLLM_PY}/models/hy_v4/nvidia/flashmla_sparse.py
COPY patches/models_hy_v4_nvidia_dfly_speculator.py          ${VLLM_PY}/models/hy_v4/nvidia/dfly_speculator.py
COPY patches/models_hy_v4_nvidia_dfly_target_reuse.py        ${VLLM_PY}/models/hy_v4/nvidia/dfly_target_reuse.py
COPY patches/v1_hisparse_binding.py                          ${VLLM_PY}/v1/hisparse/binding.py
COPY patches/model_executor_layers_attention_mla_attention.py ${VLLM_PY}/model_executor/layers/attention/mla_attention.py
COPY patches/distributed_kv_transfer_kv_connector_v1_offloading_state.py ${VLLM_PY}/distributed/kv_transfer/kv_connector/v1/offloading/state.py

RUN python3 -c "import vllm.models.hy_v4_flash.nvidia.model, vllm.models.hy_v4_flash.nvidia.fused_ops, vllm.models.hy_v4.nvidia.dfly_speculator, vllm.models.hy_v4.nvidia.dfly_target_reuse, vllm.v1.hisparse.binding"

LABEL org.opencontainers.image.source="https://github.com/jjang-ai/vllm-hy4-lyber"
LABEL org.opencontainers.image.description="vLLM nightly f8b29fab + Hy4-flash decode fusions + DFly/KV-offload compatibility."
LABEL org.opencontainers.image.licenses="Apache-2.0"
