# vllm-hy4-lyber

A small overlay image on top of `vllm/vllm-openai:hy4` that applies two
patches so vLLM can serve **Hy4-flash** checkpoints quantized with
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
MXFP4.

**Status:** ships internally; upstream PRs pending.

## What it is

- **Base image:** `vllm/vllm-openai:hy4`
- **Overlay tag:** `ghcr.io/jjang-ai/vllm-hy4-lyber:v6`
- **Target model family:** Hy4-flash class MoE models exported in the
  compressed-tensors MXFP4 format (W4, group size 32, no activation
  quantization — i.e. weight-only MXFP4 on the MoE experts).

The two patches both live under
`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/`.
They are kept as small, targeted changes against the upstream files so
they can be reviewed and (eventually) merged upstream.

## Patches

### 1. `compressed_tensors_moe.py` — MoE method dispatch

File: `compressed_tensors_moe/compressed_tensors_moe.py`, in
`CompressedTensorsMoEMethod.get_moe_method`.

The MXFP4 dispatch branch must accept weight-only configs (i.e. the
checkpoint has no `input_activations` section). The clean upstream form
is:

```python
if quant_config._is_mxfp4(weight_quant):
    ...
    return CompressedTensorsW4A4Mxfp4MoEMethod(layer.moe_config)
```

Some downstream forks have added `and input_quant is not None`, which
sends weight-only MXFP4 configs to the WNA16 path instead and refuses
to load the checkpoint. This overlay pins the branch to the upstream
form, so weight-only MXFP4 configs reach `W4A4Mxfp4MoEMethod` and the
Marlin fallback can serve them.

See `patches/compressed_tensors_moe.diff`.

### 2. `compressed_tensors_moe_w4a4_mxfp4.py` — force Marlin fallback

File: `compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py`,
in `CompressedTensorsW4A4Mxfp4MoEMethod.__init__`, line ~67.

On Hy4-flash routed experts the CUTLASS MXFP4 path raises inside
`process_weights_after_loading`. The root cause:

- `swizzle_mxfp4_scales(s2, w2_M, w2_N)` produces a tensor whose flat
  size is `w2_M * swizzle_scale_cols(w2_N)`.
- That tensor is then reshaped to `(w2_M, w2_scale_N)` where
  `w2_scale_N = w2_K / group_size`.

When a FusedMoE layer receives the un-TP-sliced down-proj scale and
the expert shape has `w2_K == w2_N` (the "square" Hy4-flash case),
those two sizes disagree — e.g. the swizzle produces `1536 * 8 = 12288`
elements but the reshape expects `1536 * 6 = 9216`. The call raises
and the server fails to come up.

This overlay force-disables the CUTLASS path for this method
(`self.use_cutlass_mxfp4 = False`), so the Marlin backend is used
instead. Marlin repacks weights and scales itself and is unaffected.

A proper fix (either TP-slicing the scale before swizzle, or handling
the `w2_K == w2_N` case correctly) is in flight upstream; see
`patches/compressed_tensors_moe_w4a4_mxfp4.diff` for the commentary.

## Usage

### Build

```bash
docker build -t ghcr.io/jjang-ai/vllm-hy4-lyber:v6 .
```

The `Dockerfile` just `COPY`s the two patched files into the base
image's site-packages and runs a quick import check.

### Serve

See `serve.sh` for the launch command. In short:

```bash
docker run --rm --gpus all --ipc=host \
    -p 8000:8000 \
    -v /path/to/model:/model \
    ghcr.io/jjang-ai/vllm-hy4-lyber:v6 \
    --model /model \
    --tensor-parallel-size 8 \
    --quantization compressed-tensors \
    --max-model-len 65536
```

No DeepFly / DeepFlash override is needed — DeepFly auto-enables from
the model's `config.json`.

## Hardware

- **Blackwell B300** (SM_100A): primary target, tested end-to-end.
- **Hopper H200 / H100**: Marlin MXFP4 path works but you should
  benchmark; memory footprint for a full Hy4-flash deployment fits
  tight on 8x H100 80GB.
- Older architectures (Ampere, Ada) are not supported by the MXFP4
  Marlin kernels in this image.

## Upstream

Two PRs are in draft against `vllm-project/vllm`:

1. `[compressed-tensors][MoE] accept weight-only MXFP4 configs on Marlin fallback`
2. `[compressed-tensors][MoE] handle untiled scale shape in CUTLASS MXFP4 swizzle`

When they land, this overlay will be retired.

## License

Apache-2.0 (matches vLLM upstream). See `LICENSE`.
