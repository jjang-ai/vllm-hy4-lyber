# vllm-hy4-lyber

A pinned vLLM runtime image for **Hy4-flash** checkpoints in compressed-tensors
**FP8 + MXFP4** format. It lets DFly speculative decoding run together with
CPU/GPU KV-cache offloading, plus image input and tool calling.

Upstream vLLM nightly already loads these checkpoints natively. This image adds
two things on top:

- small patches so that DFly and local KV connectors can be used at the same time,
  a combination upstream currently rejects;
- bit-exact fused decode kernels, which make decode about 1.9× faster.

**Status:** used in production; upstream PRs pending. Once the patches land
upstream, a plain upstream image replaces this one.

## Image

| | |
|---|---|
| Base | `vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b` (2026-11-21) |
| Tag | `ghcr.io/jjang-ai/vllm-hy4-lyber:opt2` |
| MoE kernel | CUTLASS MXFP4 (`CutlassExpertsMxfp4`), selected automatically |
| Hardware | NVIDIA Blackwell (B200/B300); validated on 8×B300 as 4 × TP2 and 2 × TP4 replicas |

```bash
docker build -t ghcr.io/jjang-ai/vllm-hy4-lyber:opt2 .
```

## Patches

All patches are in `patches/`. Each modified file is included in full, and patches 1–4
are also provided as a `.diff` against the base.

1. **`hy_v4_flash/nvidia/model.py`**: `validate_runtime` accepted only
   `HiSparseConnector`. This patch also allows the other single-engine KV
   connectors: `OffloadingConnector`, `SimpleCPUOffloadConnector` and
   `MultiConnector`.
2. **`hy_v4/nvidia/dfly_speculator.py`**: DFly rejected any
   `kv_transfer_config` with "DFly does not support PD disaggregation". This
   patch allows the same local connectors, since single-engine offload is not
   P/D disaggregation.
3. **`hy_v4/nvidia/dfly_target_reuse.py`**: DFly draft layers share the target
   layers' KV storage through a temporary `kv_sharing_target_layer_name` alias.
   That alias is cleared after storage validation. This patch records a
   persistent `dfly_kv_target_layer_name`, so connectors that bind caches later
   can still resolve the target.
4. **`v1/hisparse/binding.py`**:
   - `get_kv_sharing_readers` now honors the persistent DFly alias.
   - Each reader gets its own `draft_layer` flag; previously it inherited the
     target layer's flag.
   - Without patches 3 and 4, HiSparse with DFly fails in warmup with
     `AssertionError` (`self.view is not None`) in `hisparse/runtime.py`.

### 5. Bit-exact decode fusions (`patches/decode_exact_fusion.diff`)

vLLM's HY4-Flash model does not support `torch.compile`. With compilation enabled,
vLLM's CustomOp dispatch still picks the PyTorch-native RMSNorm and M-RoPE, expecting
the compiler to fuse them. Nothing fuses, so each decode step launches about 10,600
small kernels per token. This patch replaces the hot paths with Triton kernels that
reproduce PyTorch's rounding exactly:

- RMSNorm, including PyTorch's summation order for the mean
- M-RoPE
- the IndexPool rotate and FWHT quantize step
- index select/expand and pool writes
- the SWA window indexer
- Engram
- cached padded attention sinks

Greedy output, both tokens and logprobs, is bit-identical to the unpatched image.
Kernels per token drop from about 10,600 to about 4,200.

Each fusion can be switched off with `VLLM_HY4_FUSE_SKIP=<name>`.
`VLLM_HY4_EXACT_FUSION=0` disables them all.

| per replica, DFly 3 | TP4 before | TP4 after | TP2 after |
|---|---:|---:|---:|
| single stream tok/s | 79 | 151–185 | 149 |
| aggregate tok/s at 64 concurrent | 2,143 | 3,550–3,723 | 3,572 |

### 6. FP8 KV cache on every layer (`patches/model_executor_layers_attention_mla_attention.py`)

Hy4-flash keeps a full-length cache in only 12 of its 42 layers; the other 30 hold a
512-token sliding window. With BF16 KV (`--kv-cache-dtype auto`) a TP2 replica on B300
holds about 5.7M tokens. With `--kv-cache-dtype fp8_ds_mla` it holds about 8.6M (Engram
on GPU) or 14.6M (Engram on CPU).

Upstream rejects quantized KV on sliding-window MLA layers. Hy4-flash runs those layers
through the same sparse FlashMLA kernel as the full layers, using window indices
(`FlashSWABackend`), so they read the same `fp8_ds_mla` pages. The patch:

- allows `fp8_ds_mla`/`nvfp4_ds_mla` on sliding-window layers when the backend is
  `FLASHMLA_SPARSE` and supports sliding windows;
- applies `kv_cache_dtype_skip_layers` only to sliding-window layers, so DFly draft
  layers, which share target KV, keep the target's format.

FP8 sparse FlashMLA reads 64-token kernel blocks, so `serve.sh` adds `--block-size 64`.
Mixing BF16 sliding-window layers with FP8 full layers through the skip list is not
used: page padding forces 512-token blocks, which the packed BLHNC layout cannot split.

Measured on one TP2 replica (B300, Engram on GPU, DFly 3):

| | BF16 KV | FP8 KV |
|---|---:|---:|
| KV tokens | 5.71M | 8.65M |
| 1M-token requests at once | 5.4 | 8.3 |
| single stream tok/s (6 fixed prompts) | 193 | 184 |
| DFly acceptance, same prompts | 43.0% | 42.5% |

Needle retrieval at 500K and 980K, image input, and 1M-token multi-turn recall matched BF16.
Aggregate throughput at 96 concurrent 8K-context users was about 12% lower with FP8 KV
(measured with Engram offloaded to CPU).

## Serving

Use **four TP2 replicas per 8-GPU node**. Decode here is limited by per-step overhead
rather than compute or bandwidth, so a TP2 replica is about as fast as a TP4 one, and
four replicas give about twice the node capacity of two. Both FP8 and FP8 + MXFP4
checkpoints fit at TP2.

```bash
for i in 0 1 2 3; do
  MODEL=/path/to/checkpoint TP=2 GPUS=$((2*i)),$((2*i+1)) PORT=$((8016+i)) NAME=hy4-$i ./serve.sh
done
```

`serve.sh` settings:

| var | default | meaning |
|---|---|---|
| `KV` | `offload:400` | `offload:<GiB>` = GPU KV + CPU LRU prefix tier; `hisparse:<GiB>` = HiSparse; `none` |
| `NSPEC` | `3` | DFly draft tokens (`0` = off) |
| `MAXLEN` | `131072` | context length (model maximum 1,048,576) |
| `MAXSEQS` | `256` | max running sequences per replica |
| `HSBUF` | `11008` | HiSparse hot rows per request (see below) |

DFly must be enabled with `--speculative-config`. The `speculative_config` entry
in the checkpoint's `config.json` is not applied automatically.

### KV memory modes

- **`offload` (default):** running requests keep their KV on GPU.
  - Cached prefix blocks are evicted least-recently-used first.
  - Every block is also stored in a CPU-RAM pool (`/dev/shm`, also LRU).
  - When a returning conversation's prefix is no longer on GPU, the matching
    blocks are copied back from CPU instead of being recomputed.
  - Best for many users with multi-turn, chat-sized contexts.
- **`hisparse`:** each request's full KV also lives in a pinned host pool.
  - While GPU blocks are plentiful, requests stay fully resident.
  - When free GPU blocks fall below a watermark, requests switch to reading
    from host. Clean pages are released from GPU, and each decode step pulls only
    the sparse-attention top-k rows into a per-request GPU LRU hot buffer.
  - This roughly doubles KV capacity per replica (about 14.5M tokens with
    400 GiB host on 4×B300, against 7.35M GPU-only).
  - It is slower per token when everything already fits on GPU. Use it for
    many concurrent very long contexts.
- **HiSparse with DFly needs `device_buffer_size` pinned.**
  - Target and draft layers resolve different default sizes (top-k 2048 against
    2176). They share one LRU table, which fails with
    `region_stride must match the LRU size`.
  - Use a value ≥ `(NSPEC + 1) × 2176`, rounded up to a multiple of 256: 11008
    for `NSPEC=3`.
- **Host-pool sizing:** both offload pools live in `/dev/shm`, which defaults to
  half of RAM and is shared by all replicas on the node. Size the pools to fit.

### Notes

- **First boot takes about 10 minutes:** FlashInfer compiles its FP4 GEMM
  tactics serially. Keep the `/root/.cache` mount so later restarts reuse the
  compiled kernels.
- **Harmless startup messages:** `flashinfer/comm/mnnvl.py` logs
  `CUDA_ERROR_NOT_PERMITTED`. This is a fabric-memory probe and can be ignored.
- **DFly acceptance on general chat** is about 1.9 tokens per step with 3 draft
  tokens; positions 1/2/3 accept at about 0.54 / 0.23 / 0.09.

## License

Apache-2.0, same as upstream vLLM. See `LICENSE`.
