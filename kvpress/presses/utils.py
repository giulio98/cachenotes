# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Giulio on 2026-01-08

import torch
from torch import nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention
from transformers.models.phi3.modeling_phi3 import Phi3Attention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


def get_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Extracts the query states from a given attention module and hidden states tensor.

    This function supports multiple attention module types: Phi3Attention, Qwen3Attention, Gemma3Attention,
    and Llama-like modules. It handles the appropriate projection and reshaping to obtain the query states
    in the expected format.

    Parameters
    ----------
    module : nn.Module
        The attention module from which to extract query states. Must be one of
        Phi3Attention, Qwen3Attention, Gemma3Attention, or a Llama-like attention module
        with a 'q_proj' attribute.
    hidden_states : torch.Tensor
        The input hidden states of shape (batch_size, seq_len, hidden_dim).

    Returns
    -------
    query_states : torch.Tensor
        The extracted query states of shape (batch_size, num_heads, seq_len, head_dim).
    """
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim

    if isinstance(module, Phi3Attention):
        qkv = module.qkv_proj(hidden_states)
        query_states = qkv[..., : num_heads * head_dim]
    elif hasattr(module, "q_proj"):
        # Assume Llama-like attention layer
        query_states = module.q_proj(hidden_states)
    else:
        raise NotImplementedError(f"Press not yet implemented for {module.__class__}.")

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

    # Support for Qwen3 and Gemma3 QK norm
    if isinstance(module, (Qwen3Attention, Gemma3Attention)):
        query_states = module.q_norm(query_states)

    return query_states

from transformers.models.llama.modeling_llama import repeat_kv, rotate_half


import math
import torch
import triton
import triton.language as tl


@triton.jit
def _window_scores_kernel(
    Q_ptr,          # [B, H, W, D]   (queries after RoPE)
    K_ptr,          # [B, Kv, Q, D]  (keys; NOT repeated)
    OUT_ptr,        # [B, Kv, C]     (fp32 accumulation)
    B: tl.int32,
    H: tl.int32,
    Kv: tl.int32,
    W: tl.int32,            # window size
    Q_all: tl.int32,        # total sequence length
    C: tl.int32,            # prefix length (Q_all - W)
    G: tl.int32,            # H = Kv * G
    inv_sqrt_hd: tl.float32,
    normalize: tl.int32,    # 0/1
    # strides (elements)
    stride_q_b: tl.int64, stride_q_h: tl.int64, stride_q_w: tl.int64, stride_q_d: tl.int64,
    stride_k_b: tl.int64, stride_k_kv: tl.int64, stride_k_q: tl.int64, stride_k_d: tl.int64,
    stride_o_b: tl.int64, stride_o_kv: tl.int64, stride_o_c: tl.int64,
    # compile-time
    BLOCK_M: tl.constexpr,   # rows in W
    BLOCK_N: tl.constexpr,   # cols in C
    HEAD_DIM: tl.constexpr,  # head dim (for tl.arange)
):
    # program ids
    pid_m = tl.program_id(0)         # tile over window rows
    pid_bh = tl.program_id(1)        # over B*H

    b = pid_bh // H
    h = pid_bh % H
    kv = h // G  # map head -> kv-group index

    # Base pointers
    q_base   = Q_ptr   + b * stride_q_b + h  * stride_q_h
    k_base   = K_ptr   + b * stride_k_b + kv * stride_k_kv
    out_base = OUT_ptr + b * stride_o_b + kv * stride_o_kv

    # Tile row offsets (global window rows)
    m0 = pid_m * BLOCK_M
    offs_m = m0 + tl.arange(0, BLOCK_M)          # [BM]
    row_mask = offs_m < W

    d_offs = tl.arange(0, HEAD_DIM)              # [D]

    # Load queries: [BM, D] in src dtype; matmul result will be promoted when multiplied by fp32
    q_ptrs = q_base + (offs_m[:, None] * stride_q_w) + (d_offs[None, :] * stride_q_d)
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    # Streaming softmax stats in fp32, base-2 with LOG2E factor (to match base-e)
    LOG2E = 1.4426950408889634  # 1 / ln(2)
    scale2 = inv_sqrt_hd * LOG2E

    minus_inf = -float("inf")
    m_i = tl.full((BLOCK_M,), minus_inf, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # --- PASS 1A: keys 0..C-1 (always valid) ---
    for n0 in tl.range(0, C, BLOCK_N, warp_specialize=False):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_mask = offs_n < C

        k_ptrs = k_base + (offs_n[:, None] * stride_k_q) + (d_offs[None, :] * stride_k_d)
        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)

        # logits in fp32, scaled, converted to base-2 domain
        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale2
        max_j = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, max_j)
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    # --- PASS 1B: keys C..Q_all-1 (valid iff j - C <= i) ---
    for n0 in tl.range(C, Q_all, BLOCK_N, warp_specialize=False):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_mask = offs_n < Q_all

        k_ptrs = k_base + (offs_n[:, None] * stride_k_q) + (d_offs[None, :] * stride_k_d)
        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale2

        # causal tail: valid if offs_m[:,None] >= (offs_n[None,:] - C)
        j_delta = offs_n[None, :] - C
        mask_tail = offs_m[:, None] >= j_delta
        mask = row_mask[:, None] & col_mask[None, :] & mask_tail
        qk = tl.where(mask, qk, minus_inf)

        max_j = tl.max(qk, axis=1)
        m_ij = tl.maximum(m_i, max_j)
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    # --- PASS 2: accumulate normalized probs into OUT[:, :, 0..C-1] ---
    # counts_i (global): arange(C, C+W) if normalize else 1
    counts = tl.where(
        normalize != 0,
        (C + offs_m).to(tl.float32),
        tl.full((BLOCK_M,), 1.0, tl.float32),
    )
    counts = tl.where(row_mask, counts, 0.0)

    WGf = (W * G) + 0.0  # cast to fp32 via +0.0
    counts = counts / WGf

    for n0 in tl.range(0, C, BLOCK_N, warp_specialize=False):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        col_mask = offs_n < C

        k_ptrs = k_base + (offs_n[:, None] * stride_k_q) + (d_offs[None, :] * stride_k_d)
        k = tl.load(k_ptrs, mask=col_mask[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * scale2
        # normalized probabilities (base-2 consistent with pass1), fp32 softmax
        p = tl.math.exp2(qk - m_i[:, None])
        row_scale = counts / l_i                        # [BM]
        contrib = p * row_scale[:, None]                # [BM, BN]
        s_col = tl.sum(contrib, axis=0)                 # [BN]

        out_ptrs = out_base + offs_n * stride_o_c
        tl.atomic_add(out_ptrs, s_col, mask=col_mask)


def compute_window_attention_triton(
    module,
    hidden_states: torch.Tensor,             # [B, Q, hidden]
    keys: torch.Tensor,                      # [B, Kv, Q, D]
    window_size: int,
    position_embeddings,                     # (cos, sin)
    normalize_scores: bool = False,
):
    device = hidden_states.device
    dtype_q = hidden_states.dtype

    B, Q_all, _ = hidden_states.shape
    H  = module.config.num_attention_heads
    D  = module.head_dim
    Kv = module.config.num_key_value_heads
    G  = H // Kv
    W  = int(window_size)
    C  = Q_all - W
    assert C > 0, "window_size must be < sequence length"

    # queries (last W) + RoPE
    q = get_query_states(module, hidden_states[:, -W:])  # [B, H, W, D]
    cos, sin = position_embeddings
    cos, sin = cos[:, -W:], sin[:, -W:]
    q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
    q = q.contiguous()

    # keys: [B, Kv, Q, D], not repeated
    k = keys.contiguous()

    # output [B, Kv, C] fp32
    out = torch.zeros((B, Kv, C), device=device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    grid = (triton.cdiv(W, BLOCK_M), B * H)

    # Strides (elements)
    stride_q_b, stride_q_h, stride_q_w, stride_q_d = q.stride()
    stride_k_b, stride_k_kv, stride_k_q, stride_k_d = k.stride()
    stride_o_b, stride_o_kv, stride_o_c = out.stride()

    inv_sqrt_hd = 1.0 / math.sqrt(D)

    _window_scores_kernel[grid](
        q, k, out,
        B, H, Kv, W, Q_all, C, G, inv_sqrt_hd,
        int(normalize_scores),
        stride_q_b, stride_q_h, stride_q_w, stride_q_d,
        stride_k_b, stride_k_kv, stride_k_q, stride_k_d,
        stride_o_b, stride_o_kv, stride_o_c,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=D,            # constexpr
        num_warps=4,
        num_stages=2,
    )

    # out is already averaged over W and G, like your final PyTorch `scores`
    return out.to(dtype_q)






