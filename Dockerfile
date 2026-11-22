# vllm-hy4-lyber: pinned vLLM runtime for Hy4-flash FP8 + MXFP4 checkpoints,
# with DFly speculative decoding and CPU/GPU KV offloading enabled together.
#
# Base: upstream vLLM nightly f8b29fab (2026-11-21). Upstream already serves
# compressed-tensors FP8 + weight-only MXFP4 Hy4-flash checkpoints; this image
# adds four small patches (see README.md and patches/*.diff).

ARG BASE=vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b
FROM ${BASE}

ARG VLLM_PY=/usr/local/lib/python3.14/dist-packages/vllm

COPY patches/models_hy_v4_flash_nvidia_model.py        ${VLLM_PY}/models/hy_v4_flash/nvidia/model.py
COPY patches/models_hy_v4_nvidia_dfly_speculator.py    ${VLLM_PY}/models/hy_v4/nvidia/dfly_speculator.py
COPY patches/models_hy_v4_nvidia_dfly_target_reuse.py  ${VLLM_PY}/models/hy_v4/nvidia/dfly_target_reuse.py
COPY patches/v1_hisparse_binding.py                    ${VLLM_PY}/v1/hisparse/binding.py

RUN python3 -c "import vllm.models.hy_v4_flash.nvidia.model, vllm.models.hy_v4.nvidia.dfly_speculator, vllm.models.hy_v4.nvidia.dfly_target_reuse, vllm.v1.hisparse.binding"

LABEL org.opencontainers.image.source="https://github.com/jjang-ai/vllm-hy4-lyber"
LABEL org.opencontainers.image.description="vLLM nightly f8b29fab + DFly/KV-offload compatibility patches for Hy4-flash."
LABEL org.opencontainers.image.licenses="Apache-2.0"
