# vllm-hy4-lyber

A pinned vLLM runtime image for **Hy4-flash** checkpoints in compressed-tensors
**FP8 + MXFP4** format. It lets DFly speculative decoding run together with
CPU/GPU KV-cache offloading, plus image input and tool calling.

Upstream vLLM nightly already loads these checkpoints natively. This image adds
four small patches so that DFly and local KV connectors can be used at the same
time. Upstream currently rejects that combination.

**Status:** used in production; upstream PRs pending. Once the patches land
upstream, a plain upstream image replaces this one.

## Image

| | |
|---|---|
| Base | `vllm/vllm-openai:nightly-f8b29fab414884206460fe63b983ca34fc6e3e9b` (2026-11-21) |
| Tag | `ghcr.io/jjang-ai/vllm-hy4-lyber:n1` |
| MoE kernel | CUTLASS MXFP4 (`CutlassExpertsMxfp4`), selected automatically |
| Hardware | NVIDIA Blackwell (B200/B300); validated on 8×B300 as 2 × TP4 replicas |

```bash
docker build -t ghcr.io/jjang-ai/vllm-hy4-lyber:n1 .
```

## Patches

All four are in `patches/`, as a `.diff` against the base plus the resulting file.

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

## Serving

Use **two TP4 replicas per 8-GPU node** rather than one TP8 replica. At TP4 the
CUTLASS MXFP4 scale layout lines up, and two replicas give higher total
concurrency.

```bash
MODEL=/path/to/checkpoint GPUS=0,1,2,3 PORT=8016 NAME=hy4-a ./serve.sh
MODEL=/path/to/checkpoint GPUS=4,5,6,7 PORT=8017 NAME=hy4-b ./serve.sh
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
