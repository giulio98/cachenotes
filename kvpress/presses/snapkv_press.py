 


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.utils import get_query_states


@dataclass
class SnapKVPress(ScorerPress):
    """
    SnapKV: Attention-based KV cache compression using recent token patterns.

    Uses attention patterns of the most recent tokens to estimate importance
    of previous key-value pairs.

    Based on SnapKV (https://arxiv.org/abs/2404.14469).

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=64
        Number of recent tokens to use for computing attention-based importance scores.
    kernel_size : int, default=5
        Size of the pooling kernel applied to attention weights for smoothing.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5
    
    @staticmethod
    def compute_chunked_window_attention(
        module, hidden_states, keys, window_size, position_embeddings, chunk_size, normalize_scores=False
    ):
        """
        Returns reduced 'scores' with shape [B, Kv, C], where:
        - C = Q - window_size
        - Kv = num_key_value_heads
        Semantics match:
            attn = softmax([context, window_causal], dim=-1, fp32).to(q.dtype)
            attn_ctx = attn[..., :C]
            scores = mean_over_groups( mean_over_queries(attn_ctx * counts) )
        """
        B, Q = hidden_states.size(0), hidden_states.size(1)
        H  = module.config.num_attention_heads
        D  = module.head_dim
        Kv = module.config.num_key_value_heads
        G  = H // Kv
        W  = window_size
        C  = Q - W
        assert C > 0
        device = hidden_states.device
        chunk_size = max(1, min(int(chunk_size), int(C)))
        inv_sqrt_hd = 1.0 / math.sqrt(D)

        # Queries (last W) + RoPE
        q = get_query_states(module, hidden_states[:, -W:])
        q_dtype = q.dtype
        cos, sin = (t[:, -W:] for t in position_embeddings)
        q = (q * cos.unsqueeze(1)) + (rotate_half(q) * sin.unsqueeze(1))
        m = torch.full((B, H, W), -float("inf"), device=device, dtype=torch.float32)
        Z = torch.zeros((B, H, W), device=device, dtype=torch.float32)

        for s in range(0, C, chunk_size):
            e = min(s + chunk_size, C)
            kv = repeat_kv(keys[:, :, s:e, :], G)                                 # [B,H,cs,D], keys.dtype
            logits = torch.matmul(q, kv.transpose(2, 3)) * inv_sqrt_hd            # [B,H,W,cs], q_dtype
            logits = logits.to(torch.float32)
            m_old = m
            m = torch.maximum(m, logits.amax(-1))                               # [B,H,W]
            Z = Z * torch.exp(m_old - m) + torch.exp(logits - m.unsqueeze(-1)).sum(-1)

        kv = repeat_kv(keys[:, :, C:C+W, :], G)                                 # [B,H,W,D]
        logits = torch.matmul(q, kv.transpose(2, 3)) * inv_sqrt_hd                       # [B,H,W,W], q_dtype
        logits = logits.to(torch.float32)
        causal_mask = torch.triu(torch.ones(W, W, device=device, dtype=torch.bool), 1)
        logits.masked_fill_(causal_mask, -float("inf"))

        m_old = m
        m = torch.maximum(m, logits.amax(-1))
        Z = Z * torch.exp(m_old - m) + torch.exp(logits - m.unsqueeze(-1)).sum(-1)
        if normalize_scores:
            counts = torch.arange(Q - W, Q, device=device, dtype=q_dtype).view(1, 1, W, 1)
        else:
            counts = torch.ones((1, 1, W, 1), device=device, dtype=q_dtype)
        scores = torch.zeros((B, H, C), device=device, dtype=q_dtype)

        for s in range(0, C, chunk_size):
            e = min(s + chunk_size, C)
            kv = repeat_kv(keys[:, :, s:e, :], G)                                  # [B,H,cs,D]
            logits = (torch.matmul(q, kv.transpose(2, 3)) * inv_sqrt_hd).to(torch.float32)
            probs = (torch.exp(logits - m.unsqueeze(-1)) / Z.unsqueeze(-1)).to(q_dtype)  # fp32

            scores[:, :, s:e] = (probs * counts).mean(dim=2)
        scores = scores.view(B, Kv, G, C).mean(dim=2)

        return scores

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        query_states = get_query_states(module, hidden_states[:, -window_size:])

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        scores = attn_weights.mean(dim=-2)
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len - self.window_size)
        scores = scores.mean(2)

        # Add back the observation window. Use min score to make sure the window is pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.min().item())

        return scores
