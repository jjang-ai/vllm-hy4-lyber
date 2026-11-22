# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact Triton fusions of HY4-Flash decode glue.

Every kernel here reproduces the exact rounding sequence of the eager PyTorch
code it replaces (same fp32/bf16 casts, IEEE division, libdevice transcendental
functions, no FMA contraction), so greedy outputs are unchanged. Reductions
whose summation order PyTorch owns (``mean``, ``softmax``, ``topk``) stay in
PyTorch. Set ``VLLM_HY4_EXACT_FUSION=0`` to fall back to the original code.
"""

import os

import numpy as np
import torch

from vllm.triton_utils import tl, tldevice, triton

ENABLED = os.environ.get("VLLM_HY4_EXACT_FUSION", "1") != "0"
_SKIP = set(filter(None, os.environ.get("VLLM_HY4_FUSE_SKIP", "").split(",")))


# torch clamp_min(1e-6) on a bf16 tensor compares against the scalar cast to
# bf16 (verified bitwise in the unit test).
ENGRAM_LO = 1e-6


def on(name: str) -> bool:
    """Whether fusion `name` (norm, mrope, silu, sigmoid, swa, query, pool,
    select) is enabled; VLLM_HY4_FUSE_SKIP lists the disabled ones."""
    return ENABLED and name not in _SKIP

_NOFUSE = {"enable_fp_fusion": False}


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


# ---------------------------------------------------------------------------
# RMSNorm (vllm.ir native semantics):
#   x = x.float(); var = x.pow(2).mean(-1, keepdim); x = x * rsqrt(var + eps)
#   out = (x.to(bf16) * weight).to(bf16)
# mean stays in PyTorch (its reduction order); the rest is two kernels.
# ---------------------------------------------------------------------------
@triton.jit
def _sq_kernel(X, X_S0, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(X + r * X_S0 + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + c, x * x, mask=m)


@triton.jit
def _rms_epilogue(X, X_S0, VAR, W, OUT, EPS, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(X + r * X_S0 + c, mask=m, other=0.0).to(tl.float32)
    var = tl.load(VAR + r)
    y = _bf16(x * tldevice.rsqrt(var + EPS))
    w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + c, (y * w).to(tl.bfloat16), mask=m)


def exact_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    d = shape[-1]
    x2 = x.reshape(-1, d)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    rows = x2.shape[0]
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    block = triton.next_power_of_2(d)
    nw = 4 if block <= 1024 else 8
    bw = torch_mean_block_width(rows, d) if on("norm1") else None
    if bw is not None:
        _rms_fused_kernel[(rows,)](
            x2,
            x2.stride(0),
            weight,
            out,
            float(np.float32(eps)),
            float(np.float32(1.0) / np.float32(d)),
            d,
            bw,
            -(-d // (4 * bw)),
            block,
            num_warps=nw,
            **_NOFUSE,
        )
        return out
    sq = torch.empty((rows, d), dtype=torch.float32, device=x.device)
    _sq_kernel[(rows,)](x2, x2.stride(0), sq, d, block, num_warps=nw, **_NOFUSE)
    var = sq.mean(dim=-1, keepdim=True)
    eps32 = float(np.float32(eps))
    _rms_epilogue[(rows,)](
        x2, x2.stride(0), var, weight, out, eps32, d, block, num_warps=nw, **_NOFUSE
    )
    return out


@triton.jit
def _halve(v, W: tl.constexpr):
    """value[t] = value[t] + value[t + W/2] (torch block_x_reduce step)."""
    v2 = tl.reshape(v, (2, W // 2))
    row = tl.arange(0, 2)[:, None]
    a = tl.sum(tl.where(row == 0, v2, -0.0), axis=0)
    b = tl.sum(tl.where(row == 1, v2, -0.0), axis=0)
    return a + b


@triton.jit
def _rms_fused_kernel(
    X,
    X_S0,
    W,
    OUT,
    EPS,
    FACTOR,
    D: tl.constexpr,
    BW: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One-kernel RMSNorm whose mean(x^2) replays torch's reduce_kernel order:
    BW threads, 4 vector-lane accumulators, lanes combined left to right,
    then the halving block/warp tree; mean = sum * (1/D)."""
    r = tl.program_id(0).to(tl.int64)
    t = tl.arange(0, BW)[:, None]
    lane = tl.arange(0, 4)[None, :]
    acc = tl.zeros((BW, 4), tl.float32)
    for k in tl.static_range(K):
        idx = 4 * (t + k * BW) + lane
        v = tl.load(X + r * X_S0 + idx, mask=idx < D, other=0.0).to(tl.float32)
        acc = acc + v * v
    a0 = tl.sum(tl.where(lane == 0, acc, -0.0), axis=1)
    a1 = tl.sum(tl.where(lane == 1, acc, -0.0), axis=1)
    a2 = tl.sum(tl.where(lane == 2, acc, -0.0), axis=1)
    a3 = tl.sum(tl.where(lane == 3, acc, -0.0), axis=1)
    v = ((a0 + a1) + a2) + a3
    if BW >= 512:
        v = _halve(v, 512)
    if BW >= 256:
        v = _halve(v, 256)
    if BW >= 128:
        v = _halve(v, 128)
    if BW >= 64:
        v = _halve(v, 64)
    if BW >= 32:
        v = _halve(v, 32)
    if BW >= 16:
        v = _halve(v, 16)
    if BW >= 8:
        v = _halve(v, 8)
    if BW >= 4:
        v = _halve(v, 4)
    if BW >= 2:
        v = _halve(v, 2)
    var = tl.sum(v, axis=0) * FACTOR
    inv = tldevice.rsqrt(var + EPS)
    c = tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(X + r * X_S0 + c, mask=m, other=0.0).to(tl.float32)
    y = _bf16(x * inv)
    w = tl.load(W + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + c, (y * w).to(tl.bfloat16), mask=m)


def _last_pow2(n: int) -> int:
    return 1 << (int(n).bit_length() - 1)


def torch_mean_block_width(rows: int, d: int) -> int | None:
    """block.x of torch's reduce_kernel for mean(-1) of a contiguous fp32
    [rows, d] tensor (setReduceConfig, vectorized input, no warp/CTA split)."""
    if d < 128 or d % 4 or rows < 1 or rows >= 512:
        return None
    dim0, dim1, mnt = d // 4, rows, 512
    p0 = _last_pow2(dim0) if dim0 < mnt else mnt
    p1 = _last_pow2(dim1) if dim1 < mnt else mnt
    bw = min(p0, 32)
    bh = min(p1, mnt // bw)
    bw = min(p0, mnt // bh)
    vpt = -(-d // bw)
    if vpt >= min(bh * 16, 256):  # split across warps -> different order
        return None
    return bw


def exact_rms_supported(x, weight, variance_size_override) -> bool:
    return (
        on("norm")
        and variance_size_override is None
        and weight is not None
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )


# ---------------------------------------------------------------------------
# M-RoPE (MRotaryEmbedding.forward_native, GPT-J style, all-bf16 arithmetic)
#   o1 = x1*cos - x2*sin ; o2 = x2*cos + x1*sin, every op rounded to bf16.
# Rotates q[..., ROT_OFF:ROT_OFF+64] in place and writes k_pe to KOUT.
# ---------------------------------------------------------------------------
@triton.jit
def _mrope_native_kernel(
    POS,
    POS_S0,
    SEL,
    CACHE,
    CACHE_S0,
    Q,
    Q_S0,
    Q_S1,
    K,
    K_S0,
    KOUT,
    NQH: tl.constexpr,
    HALF: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    i = tl.arange(0, HALF)
    axis = tl.load(SEL + i)
    pos = tl.load(POS + axis * POS_S0 + t)
    cos = tl.load(CACHE + pos * CACHE_S0 + i).to(tl.float32)
    sin = tl.load(CACHE + pos * CACHE_S0 + HALF + i).to(tl.float32)
    if h < NQH:
        base = Q + t * Q_S0 + h * Q_S1
        dst = base
    else:
        base = K + t * K_S0
        dst = KOUT + t * (2 * HALF)
    x1 = tl.load(base + 2 * i).to(tl.float32)
    x2 = tl.load(base + 2 * i + 1).to(tl.float32)
    o1 = _bf16(_bf16(x1 * cos) - _bf16(x2 * sin))
    o2 = _bf16(_bf16(x2 * cos) + _bf16(x1 * sin))
    tl.store(dst + 2 * i, o1.to(tl.bfloat16))
    tl.store(dst + 2 * i + 1, o2.to(tl.bfloat16))


def mrope_sel(rotary_emb, device) -> torch.Tensor:
    sel = getattr(rotary_emb, "_hy4_exact_sel", None)
    if sel is None or sel.device != device:
        half = rotary_emb.rotary_dim // 2
        sec = rotary_emb.mrope_section
        ch = torch.arange(half)
        if rotary_emb.mrope_interleaved:
            sel = torch.zeros(half, dtype=torch.int64)
            sel[(ch % 3 == 1) & (ch < sec[1] * 3)] = 1
            sel[(ch % 3 == 2) & (ch < sec[2] * 3)] = 2
        else:
            sel = torch.cat(
                [torch.full((w,), a, dtype=torch.int64) for a, w in enumerate(sec)]
            )
        sel = sel.to(device)
        rotary_emb._hy4_exact_sel = sel
    return sel


def exact_mrope_supported(rotary_emb, positions, q, k_pe) -> bool:
    return (
        on("mrope")
        and positions.ndim == 2
        and positions.shape[0] == 3
        and getattr(rotary_emb, "mrope_section", None)
        and not rotary_emb.is_neox_style
        and rotary_emb.rotary_dim == rotary_emb.head_size == 64
        and q.dtype == torch.bfloat16
        and k_pe.dtype == torch.bfloat16
        and positions.stride(-1) == 1
        and q.stride(-1) == 1
        and k_pe.stride(-1) == 1
    )


def exact_mrope_(rotary_emb, positions, q, rot_off, k_pe) -> torch.Tensor:
    """Rotate q[:, :, rot_off:rot_off+64] in place; return rotated k_pe [T,1,64]."""
    tokens, heads = q.shape[0], q.shape[1]
    cache = rotary_emb._match_cos_sin_cache_dtype(q)
    kout = torch.empty((tokens, 1, 64), dtype=k_pe.dtype, device=k_pe.device)
    if tokens:
        qv = q[..., rot_off:]
        _mrope_native_kernel[(tokens, heads + 1)](
            positions,
            positions.stride(0),
            mrope_sel(rotary_emb, q.device),
            cache,
            cache.stride(0),
            qv,
            qv.stride(0),
            qv.stride(1),
            k_pe,
            k_pe.stride(0),
            kout,
            heads,
            32,
            num_warps=1,
            **_NOFUSE,
        )
    return kout


# ---------------------------------------------------------------------------
# F.silu(x[..., :d]) * x[..., d:]   and   out * torch.sigmoid(gate)
# ---------------------------------------------------------------------------
@triton.jit
def _silu_mul_kernel(X, X_S0, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    a = tl.load(X + r * X_S0 + c, mask=m, other=0.0).to(tl.float32)
    b = tl.load(X + r * X_S0 + D + c, mask=m, other=0.0).to(tl.float32)
    s = _bf16(tldevice.div_rn(a, 1.0 + tldevice.exp(-a)))
    tl.store(OUT + r * D + c, (s * b).to(tl.bfloat16), mask=m)


def exact_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x2 = x.reshape(-1, 2 * d)
    out = torch.empty((*x.shape[:-1], d), dtype=x.dtype, device=x.device)
    if x2.shape[0]:
        block = min(1024, triton.next_power_of_2(d))
        _silu_mul_kernel[(x2.shape[0], triton.cdiv(d, block))](
            x2, x2.stride(0), out, d, block, num_warps=4, **_NOFUSE
        )
    return out


@triton.jit
def _sigmoid_mul_kernel(A, G, G_S0, OUT, D: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = c < D
    a = tl.load(A + r * D + c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + r * G_S0 + c, mask=m, other=0.0).to(tl.float32)
    s = _bf16(tldevice.div_rn(1.0, 1.0 + tldevice.exp(-g)))
    tl.store(OUT + r * D + c, (a * s).to(tl.bfloat16), mask=m)


def exact_sigmoid_mul(a: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    rows, d = a.shape
    out = torch.empty_like(a)
    if rows:
        block = min(1024, triton.next_power_of_2(d))
        _sigmoid_mul_kernel[(rows, triton.cdiv(d, block))](
            a, gate, gate.stride(0), out, d, block, num_warps=4, **_NOFUSE
        )
    return out


# ---------------------------------------------------------------------------
# Flash SWA indexer: rows -> request -> logical position -> causal window.
# ---------------------------------------------------------------------------
@triton.jit
def _swa_window_kernel(
    STARTS,
    SEQ_LENS,
    OUT,
    OUT_S0,
    NUM_ACTUAL,
    BATCH,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    BB: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, BB)
    ends = tl.load(STARTS + 1 + b, mask=b < BATCH, other=0).to(tl.int64)
    cnt = tl.sum(((ends <= row) & (b < BATCH)).to(tl.int64), axis=0)
    req = tl.minimum(cnt, BATCH - 1)
    s0 = tl.load(STARTS + req).to(tl.int64)
    s1 = tl.load(STARTS + req + 1).to(tl.int64)
    last = tl.load(STARTS + BATCH).to(tl.int64)
    sl = tl.load(SEQ_LENS + req).to(tl.int64)
    logical = sl - (s1 - s0) + row - s0
    live = (row < NUM_ACTUAL) & (row < last)
    pos = tl.where(live, logical, -1)
    col = tl.arange(0, WIDTH).to(tl.int64)
    idx = pos - WINDOW + 1 + col
    ok = (pos >= 0) & (col < WINDOW) & (idx >= 0)
    tl.store(OUT + row * OUT_S0 + col, tl.where(ok, idx, -1).to(tl.int32))


def swa_window_indices(output, starts, seq_lens, num_actual, window):
    rows, width = output.shape
    batch = starts.numel() - 1
    if rows == 0:
        return output
    _swa_window_kernel[(rows,)](
        starts,
        seq_lens,
        output,
        output.stride(0),
        num_actual,
        batch,
        window,
        width,
        triton.next_power_of_2(max(batch, 1)),
        num_warps=4,
    )
    return output


# ---------------------------------------------------------------------------
# IndexPool: rotate_indexer (interleaved M-RoPE on the last 64 dims, de-
# interleaved output) + fwht_quantize (Hadamard butterflies in fp32, power-of-2
# scale, FP8 E4M3), bit-identical to indexpool_ops.
# ---------------------------------------------------------------------------
_INV_448 = float(np.float32(1.0) / np.float32(448.0))
_INV_SQRT128 = 128**-0.5


@triton.jit
def _rope_fwht_row(X_ROW, POS, POS_S0, NPOS: tl.constexpr, SEL, INV, t, INV448):
    """Return the 128 fp32 FWHT outputs and the fp32 scale of one row."""
    d = tl.arange(0, 128)
    x = tl.load(X_ROW + d).to(tl.float32)  # bf16 -> f32
    # --- rotate (rope dims 64..127) ---
    j = tl.where(d >= 96, d - 96, d - 64)  # frequency index for rope lanes
    j = tl.where(d >= 64, j, 0)
    if NPOS == 3:
        axis = tl.load(SEL + j)
        p = tl.load(POS + axis * POS_S0 + t).to(tl.float32)
    else:
        p = tl.load(POS + t).to(tl.float32)
    f = p * tl.load(INV + j)
    cf = tldevice.cos(f)
    sf = tldevice.sin(f)
    # pe'[j] = pe[2j] (j<32), pe'[32+j] = pe[2j+1]; rot[j] = pe'[j]cos - pe'[32+j]sin
    # rot[32+j] = pe'[32+j]cos + pe'[j]sin
    lo = tl.where(d >= 96, 64 + 2 * j + 1, 64 + 2 * j)  # own pe' element
    hi = tl.where(d >= 96, 64 + 2 * j, 64 + 2 * j + 1)  # partner pe' element
    own = tl.load(X_ROW + tl.where(d >= 64, lo, d)).to(tl.float32)
    other = tl.load(X_ROW + tl.where(d >= 64, hi, d)).to(tl.float32)
    sgn_other = tl.where(d >= 96, other, -other)
    rot = own * cf + sgn_other * sf
    x = tl.where(d >= 64, _bf16(rot), x)
    # --- FWHT butterflies (same pairing/order as the eager stack loop) ---
    x = _fwht_stage(x, 64, 1)
    x = _fwht_stage(x, 32, 2)
    x = _fwht_stage(x, 16, 4)
    x = _fwht_stage(x, 8, 8)
    x = _fwht_stage(x, 4, 16)
    x = _fwht_stage(x, 2, 32)
    x = _fwht_stage(x, 1, 64)
    x = x * 0.08838834764831845
    amax = tl.max(tl.abs(x), axis=0)
    r = tl.maximum(amax, 1e-4) * INV448
    scale = tldevice.exp2(tldevice.ceil(tldevice.log2(r)))
    return x, scale


@triton.jit
def _fwht_stage(x, A: tl.constexpr, B: tl.constexpr):
    """One butterfly with stride B on a 128-vector viewed as [A, 2, B]."""
    v = tl.reshape(x, (A, 2, B))
    sel = tl.arange(0, 2)[None, :, None]
    a = tl.sum(tl.where(sel == 0, v, -0.0), axis=1)
    b = tl.sum(tl.where(sel == 1, v, -0.0), axis=1)
    lo = (a + b)[:, None, :]
    hi = (a - b)[:, None, :]
    return tl.reshape(tl.where(sel == 0, lo, hi), (128,))


@triton.jit
def _quantize(x, scale):
    q = tldevice.div_rn(x, scale)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0)
    return q.to(tl.float8e4nv)


@triton.jit
def _mx_scale(scale):
    return (scale.to(tl.int32, bitcast=True) >> 23) * 0x01010101


@triton.jit
def _indexer_query_kernel(
    X,
    X_S0,
    POS,
    POS_S0,
    NPOS: tl.constexpr,
    SEL,
    INV,
    HW,
    HW_S0,
    QOUT,
    WOUT,
    WCONST,
    INV448,
    H: tl.constexpr,
    HP: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, 128)
    if h < H:
        x, scale = _rope_fwht_row(X + t * X_S0 + h * 128, POS, POS_S0, NPOS, SEL, INV, t, INV448)
        q = _quantize(x, scale)
        tl.store(QOUT + (t * HP + h) * 128 + d, q)
        w = tl.load(HW + t * HW_S0 + h).to(tl.float32)
        tl.store(WOUT + t * HP + h, (w * WCONST) * scale)
    else:
        tl.store(QOUT + (t * HP + h) * 128 + d, tl.zeros((128,), tl.float8e4nv))
        tl.store(WOUT + t * HP + h, 0.0)


def indexer_query(query2d, positions, sel, inv, hw, heads, heads_pad, wconst):
    """query2d [T, H*128] bf16 -> (quant_q [T,HP,128] fp8, weights [T,HP] f32)."""
    tokens = query2d.shape[0]
    qout = torch.empty(
        (tokens, heads_pad, 128), dtype=torch.float8_e4m3fn, device=query2d.device
    )
    wout = torch.empty((tokens, heads_pad), dtype=torch.float32, device=query2d.device)
    if tokens:
        _indexer_query_kernel[(tokens, heads_pad)](
            query2d,
            query2d.stride(0),
            positions,
            positions.stride(0) if positions.ndim == 2 else 0,
            positions.shape[0] if positions.ndim == 2 else 1,
            sel,
            inv,
            hw,
            hw.stride(0),
            qout,
            wout,
            float(np.float32(wconst)),
            _INV_448,
            heads,
            heads_pad,
            num_warps=4,
            **_NOFUSE,
        )
    return qout, wout


@triton.jit
def _indexer_pool_kernel(
    X,
    POS,
    POS_S0,
    NPOS: tl.constexpr,
    SEL,
    INV,
    CACHE,
    CACHE_S0,
    TABLE,
    TABLE_S0,
    TABLE_COLS,
    OWNERS,
    GROUPS,
    ACTIVE,
    INV448,
    PAGE: tl.constexpr,
    MX: tl.constexpr,
):
    s = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    x, scale = _rope_fwht_row(X + s * 128, POS, POS_S0, NPOS, SEL, INV, s, INV448)
    q = _quantize(x, scale)
    owner = tl.load(OWNERS + s).to(tl.int64)
    group = tl.load(GROUPS + s).to(tl.int64)
    active = tl.load(ACTIVE + s)
    col = tl.minimum(tl.maximum(group // PAGE, 0), TABLE_COLS - 1)
    block = tl.load(TABLE + owner * TABLE_S0 + col).to(tl.int64)
    base = block * CACHE_S0
    row = group % PAGE
    if active:
        tl.store(CACHE + base + row * 128 + d, q.to(tl.uint8, bitcast=True))
        if MX:
            sv = _mx_scale(scale)
        else:
            sv = scale.to(tl.int32, bitcast=True)
        byte = tl.arange(0, 4)
        tl.store(
            CACHE + base + PAGE * 128 + row * 4 + byte,
            ((sv >> (8 * byte)) & 0xFF).to(tl.uint8),
        )


def indexer_pool_write(pooled, positions, sel, inv, cache, table, owners, groups, active, mx):
    slots = pooled.shape[0]
    if slots == 0:
        return
    assert pooled.is_contiguous() and pooled.dtype == torch.bfloat16
    _indexer_pool_kernel[(slots,)](
        pooled,
        positions,
        positions.stride(0) if positions.ndim == 2 else 0,
        positions.shape[0] if positions.ndim == 2 else 1,
        sel,
        inv,
        cache,
        cache.stride(0),
        table,
        table.stride(0),
        table.shape[1],
        owners,
        groups,
        active,
        _INV_448,
        cache.shape[1],
        mx,
        num_warps=4,
        **_NOFUSE,
    )


def indexer_rope_tables(rope_dim, parameters, device):
    """`inverse` exactly as rotate_indexer computes it, plus the M-RoPE axis map."""
    theta = parameters.get("rope_theta", 10000.0)
    inv = theta ** (
        -torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim
    )
    half = rope_dim // 2
    section = parameters.get("mrope_section")
    ch = torch.arange(half)
    sel = torch.zeros(half, dtype=torch.int64)
    if section is not None:
        if parameters.get("mrope_interleaved", True):
            sel[(ch % 3 == 1) & (ch < section[1] * 3)] = 1
            sel[(ch % 3 == 2) & (ch < section[2] * 3)] = 2
        else:
            sel = torch.cat(
                [torch.full((w,), a, dtype=torch.int64) for a, w in enumerate(section)]
            )
    return sel.to(device), inv


# ---------------------------------------------------------------------------
# select_groups (post-topk) + expand_groups, writing the final index row.
# ---------------------------------------------------------------------------
@triton.jit
def _mask_scores_kernel(L, L_S0, UPPER, W, BLOCK: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    up = tl.load(UPPER + r).to(tl.int64)
    m = (c < W) & (c >= up)
    tl.store(L + r * L_S0 + c, tl.full((BLOCK,), float("-inf"), tl.float32), mask=m)


def mask_scores_(logits, upper):
    rows, w = logits.shape
    if rows:
        block = 1024
        _mask_scores_kernel[(rows, triton.cdiv(w, block))](
            logits, logits.stride(0), upper, w, block, num_warps=4
        )
    return logits


@triton.jit
def _select_expand_kernel(
    VALS,
    IDX,
    COUNT,
    LOGICAL,
    LIVE,
    OUT,
    OUT_S0,
    GB: tl.constexpr,  # budget + 1 (power of two): sorted ids + one tail slot
    R: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g = tl.arange(0, GB)
    m = g < COUNT
    v = tl.load(VALS + row * COUNT + g, mask=m, other=float("-inf"))
    i = tl.load(IDX + row * COUNT + g, mask=m, other=0).to(tl.int32)
    sentinel = 2147483647
    i = tl.where(v != float("-inf"), i, sentinel)
    i = tl.where(m, i, sentinel)
    i = tl.sort(i)
    i = tl.where(i == sentinel, -1, i)
    # slot GB-1 is the padded budget slot -> replaced by the tail
    pos = tl.load(LOGICAL + row).to(tl.int64)
    live = (pos >= 0) & (tl.load(LIVE + row) != 0)
    upper = pos // R
    off = tl.arange(0, R).to(tl.int64)
    gid = i.to(tl.int64)
    valid = (gid >= 0) & (gid < upper) & live
    hist = tl.where(valid[:, None], gid[:, None] * R + off[None, :], -1)
    tail = upper * R + off
    tail = tl.where(live & (tail <= pos), tail, -1)
    is_tail = (g == GB - 1)[:, None]
    cand = tl.where(is_tail, tail[None, :], hist)
    cand = tl.reshape(cand, (GB * R,)).to(tl.int32)
    ok = cand >= 0
    okn = ok.to(tl.int32)
    inc = tl.cumsum(okn, axis=0)
    total = tl.sum(okn, axis=0)
    bad = tl.cumsum(1 - okn, axis=0)
    dest = tl.where(ok, inc - 1, total + bad - 1)
    tl.store(OUT + row * OUT_S0 + dest, cand)


def select_expand(values, indices, logical, live, ratio, out):
    rows, count = values.shape
    width = out.shape[1]
    gb = width // ratio
    if rows:
        _select_expand_kernel[(rows,)](
            values.contiguous(),
            indices.contiguous(),
            count,
            logical,
            live,
            out,
            out.stride(0),
            gb,
            ratio,
            num_warps=8,
        )
    return out


# ---------------------------------------------------------------------------
# Flash Engram glue (4 layers): row dequant/mask/norm, the gate chain of the
# four hc channels, and the value-norm/gate/scale/mask/residual combine.
# ---------------------------------------------------------------------------
@triton.jit
def _engram_rows(ROWS, SCALES, IDX, GAMMA, VAR, OUT, EPS, H, D: tl.constexpr,
                 BLOCK: tl.constexpr, FP8: tl.constexpr, PHASE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)  # (token, head) row
    c = tl.arange(0, BLOCK)
    m = c < D
    x = tl.load(ROWS + r * D + c, mask=m, other=0.0).to(tl.float32)
    if FP8:
        x = x * tl.load(SCALES + r).to(tl.float32)
    dead = tl.load(IDX + r) < 0
    x = tl.where(dead, 0.0, x)
    if PHASE == 0:
        tl.store(OUT + r * D + c, x * x, mask=m)
    else:
        inv = tldevice.rsqrt(tl.load(VAR + r) + EPS)
        g = tl.load(GAMMA + (r % H) * D + c, mask=m, other=0.0)
        tl.store(OUT + r * D + c, ((x * inv) * g).to(tl.bfloat16), mask=m)


def engram_rows_norm(rows, scales, indices, gamma, eps):
    """rows [T,H,D] (fp8|bf16), scales [T,H] bf16|None, indices [T,H] -> bf16."""
    t, h, d = rows.shape
    n = t * h
    block = triton.next_power_of_2(d)
    sq = torch.empty((t, h, d), dtype=torch.float32, device=rows.device)
    fp8 = scales is not None
    sc = scales if fp8 else rows
    eps32 = float(np.float32(eps))
    _engram_rows[(n,)](rows, sc, indices, gamma, sq, sq, eps32, h, d, block, fp8, 0,
                       num_warps=1, **_NOFUSE)
    var = sq.mean(-1, keepdim=True)
    out = torch.empty((t, h, d), dtype=torch.bfloat16, device=rows.device)
    _engram_rows[(n,)](rows, sc, indices, gamma, var, out, eps32, h, d, block, fp8, 1,
                       num_warps=1, **_NOFUSE)
    return out


@triton.jit
def _engram_gate(S0, S1, S2, S3, OUT, T, CONST, LO, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = t < T
    for ch in tl.static_range(4):
        if ch == 0:
            g = tl.load(S0 + t, mask=m, other=0.0)
        elif ch == 1:
            g = tl.load(S1 + t, mask=m, other=0.0)
        elif ch == 2:
            g = tl.load(S2 + t, mask=m, other=0.0)
        else:
            g = tl.load(S3 + t, mask=m, other=0.0)
        g = _bf16(g.to(tl.float32) * CONST)
        a = tl.abs(g)
        a = tl.where(a != a, a, tl.maximum(a, LO))
        s = _bf16(tldevice.sqrt_rn(a))
        sg = tl.where(g > 0, 1.0, tl.where(g < 0, -1.0, 0.0))
        v = _bf16(s * sg)
        sig = tldevice.div_rn(1.0, 1.0 + tldevice.exp(-v))
        tl.store(OUT + t * 4 + ch, sig.to(tl.bfloat16), mask=m)


def engram_gate(sums, const, lo):
    t = sums[0].shape[0]
    out = torch.empty((t, 4), dtype=torch.bfloat16, device=sums[0].device)
    if t:
        block = 128
        _engram_gate[(triton.cdiv(t, block),)](
            *sums, out, t, float(np.float32(const)), float(lo), block, num_warps=4,
            **_NOFUSE)
    return out


@triton.jit
def _torch_mean_sq(X_ROW, D: tl.constexpr, BW: tl.constexpr, K: tl.constexpr, FACTOR):
    t = tl.arange(0, BW)[:, None]
    lane = tl.arange(0, 4)[None, :]
    acc = tl.zeros((BW, 4), tl.float32)
    for k in tl.static_range(K):
        idx = 4 * (t + k * BW) + lane
        v = tl.load(X_ROW + idx, mask=idx < D, other=0.0).to(tl.float32)
        acc = acc + v * v
    a0 = tl.sum(tl.where(lane == 0, acc, -0.0), axis=1)
    a1 = tl.sum(tl.where(lane == 1, acc, -0.0), axis=1)
    a2 = tl.sum(tl.where(lane == 2, acc, -0.0), axis=1)
    a3 = tl.sum(tl.where(lane == 3, acc, -0.0), axis=1)
    v = ((a0 + a1) + a2) + a3
    if BW >= 512:
        v = _halve(v, 512)
    if BW >= 256:
        v = _halve(v, 256)
    if BW >= 128:
        v = _halve(v, 128)
    if BW >= 64:
        v = _halve(v, 64)
    if BW >= 32:
        v = _halve(v, 32)
    if BW >= 16:
        v = _halve(v, 16)
    if BW >= 8:
        v = _halve(v, 8)
    if BW >= 4:
        v = _halve(v, 4)
    if BW >= 2:
        v = _halve(v, 2)
    return tl.sum(v, axis=0) * FACTOR


@triton.jit
def _engram_combine(HID, HID_S0, HID_S1, VAL, VAL_S0, GATE, PRE, SCALES, HASH, HASH_S0,
                    NH, OUT, EPS, FACTOR, D: tl.constexpr, BW: tl.constexpr,
                    K: tl.constexpr, BLOCK: tl.constexpr, HB: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    var = _torch_mean_sq(VAL + t * VAL_S0, D, BW, K, FACTOR)
    inv = tldevice.rsqrt(var + EPS)
    c = tl.arange(0, BLOCK)
    m = c < D
    v = tl.load(VAL + t * VAL_S0 + c, mask=m, other=0.0).to(tl.float32)
    pre = tl.load(PRE + c, mask=m, other=0.0).to(tl.float32)
    signal = (pre * v) * inv
    g = tl.load(GATE + t * 4 + ch).to(tl.float32)
    sc = tl.load(SCALES + ch * D + c, mask=m, other=0.0).to(tl.float32)
    res = _bf16((g + sc) * signal)
    hb = tl.arange(0, HB)
    hv = tl.load(HASH + t * HASH_S0 + hb, mask=hb < NH, other=-1)
    dead = tl.sum((hv >= 0).to(tl.int32), axis=0) == 0
    res = tl.where(dead, 0.0, res)
    h = tl.load(HID + t * HID_S0 + ch * HID_S1 + c, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + (t * 4 + ch) * D + c, (h + res).to(tl.bfloat16), mask=m)


def engram_combine(hidden, value, gate, pre_gamma, scales, hashes, eps):
    """hidden [T,4,D] bf16, value [T,D] bf16 (post all-reduce) -> [T,4,D] bf16."""
    t, hc, d = hidden.shape
    out = torch.empty((t, hc, d), dtype=hidden.dtype, device=hidden.device)
    if t == 0:
        return out
    bw = torch_mean_block_width(t, d)
    assert bw is not None
    _engram_combine[(t, hc)](
        hidden, hidden.stride(0), hidden.stride(1), value, value.stride(0), gate,
        pre_gamma, scales, hashes, hashes.stride(0), hashes.shape[1], out,
        float(np.float32(eps)), float(np.float32(1.0) / np.float32(d)), d, bw,
        -(-d // (4 * bw)), triton.next_power_of_2(d),
        triton.next_power_of_2(hashes.shape[1]), num_warps=8, **_NOFUSE)
    return out
