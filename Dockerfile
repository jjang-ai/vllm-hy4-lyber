# vllm-hy4-lyber: vLLM runtime image for Hy4-flash compressed-tensors MXFP4.
#
# Base: vllm/vllm-openai:hy4  (the public Hy4 preview image).
# Adds two small patches so the compressed-tensors MXFP4 MoE path
# dispatches correctly and side-steps a known CUTLASS swizzle bug on
# routed-expert shapes where w2_K == w2_N.  See README.md and patches/*.diff
# for the exact changes and the upstream PR status.

FROM vllm/vllm-openai:hy4

ARG VLLM_PY=/usr/local/lib/python3.12/dist-packages/vllm
ARG CT_MOE_DIR=${VLLM_PY}/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe

# Patch 1: ensure weight-only MXFP4 configs reach W4A4Mxfp4MoEMethod
#          (so Marlin can serve them).  The clean upstream form of the
#          dispatch is `if quant_config._is_mxfp4(weight_quant):`.
# Patch 2: force Marlin fallback inside W4A4Mxfp4MoEMethod to work around
#          a CUTLASS MXFP4 swizzle+reshape mismatch on Hy4-flash routed
#          experts (see patches/compressed_tensors_moe_w4a4_mxfp4.diff).
COPY patches/compressed_tensors_moe.py                  ${CT_MOE_DIR}/compressed_tensors_moe.py
COPY patches/compressed_tensors_moe_w4a4_mxfp4.py       ${CT_MOE_DIR}/compressed_tensors_moe_w4a4_mxfp4.py

# Quick sanity check: the module still imports.
RUN python -c "from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import CompressedTensorsW4A4Mxfp4MoEMethod" \
 && python -c "from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import CompressedTensorsMoEMethod"

LABEL org.opencontainers.image.source="https://github.com/jjang-ai/vllm-hy4-lyber"
LABEL org.opencontainers.image.description="vLLM runtime image for Hy4-flash compressed-tensors MXFP4 (two patches on top of vllm/vllm-openai:hy4)."
LABEL org.opencontainers.image.licenses="Apache-2.0"
