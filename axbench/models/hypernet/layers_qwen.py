"""Qwen2/Qwen3-based cross-attention + decoder layer for HyperSteer.

Targets transformers >= 4.51 where Qwen3 lives in `transformers.models.qwen3`
and the per-layer API is:
  * the model precomputes `position_embeddings = rotary_emb(h, position_ids)`
    once and threads them down through every layer / attention call;
  * decoder layers take `position_embeddings` as a positional kwarg;
  * Qwen3Attention has q_norm / k_norm on the per-head dim;
  * `config.layer_types[layer_idx]` selects full vs sliding attention.

The cross-attention sub-block is written from scratch (no
`Qwen3Attention` inheritance) so its forward stays stable across
transformers releases. It still uses Qwen3's q/k norm + rotary on Q
(K comes from the policy model's residual at the steer layer and is
not rotated, per the standard cross-attention convention).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.cache_utils import Cache
from transformers.utils import logging

# Prefer Qwen3 (transformers >= 4.51); fall back to Qwen2 only for the
# class types we actually inherit from.
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Config as _QwenConfig,
        Qwen3RMSNorm as _QwenRMSNorm,
        Qwen3DecoderLayer as _QwenDecoderLayer,
        Qwen3RotaryEmbedding as _QwenRotaryEmbedding,
        apply_rotary_pos_emb as _apply_rotary_pos_emb,
        repeat_kv,
    )
    _QWEN_FAMILY = "qwen3"
except ImportError:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Config as _QwenConfig,
        Qwen2RMSNorm as _QwenRMSNorm,
        Qwen2DecoderLayer as _QwenDecoderLayer,
        Qwen2RotaryEmbedding as _QwenRotaryEmbedding,
        apply_rotary_pos_emb as _apply_rotary_pos_emb,
        repeat_kv,
    )
    _QWEN_FAMILY = "qwen2"


logger = logging.get_logger(__name__)


class HypernetCrossAttentionQwen(nn.Module):
    """Self-contained cross-attention block for the Qwen-based hypernet.

    Q comes from `hidden_states` (the hypernet's residual at this layer);
    K and V come from `encoder_hidden_states` (the policy model's
    residual at the steer layer). Rotary embeddings are applied only to
    Q — K positions live in a separate (encoder) coordinate system.
    Includes the Qwen3 q_norm / k_norm on the per-head dim.

    Does NOT inherit from `Qwen3Attention` so its forward stays stable
    across transformers releases.
    """

    def __init__(self, config: _QwenConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads,
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        attn_bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=attn_bias)

        # Qwen3-specific: per-head Q/K RMSNorm. We include them so the
        # Qwen3 hypernet keeps the same normalization shape as the
        # policy model's attention.
        if _QWEN_FAMILY == "qwen3":
            self.q_norm = _QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = _QwenRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(encoder_hidden_states).view(bsz, kv_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(encoder_hidden_states).view(bsz, kv_len, self.num_key_value_heads, self.head_dim)

        # Qwen3 q_norm/k_norm act on the per-head dim (last axis), pre-transpose
        query_states = self.q_norm(query_states).transpose(1, 2)  # (B, nh, q_len, head_dim)
        key_states = self.k_norm(key_states).transpose(1, 2)      # (B, nkv, kv_len, head_dim)
        value_states = value_states.transpose(1, 2)               # (B, nkv, kv_len, head_dim)

        # Apply rotary to Q only. K positions are encoder positions and
        # should not be rotated against Q's coordinate system in cross-attn.
        if query_position_embeddings is not None:
            cos, sin = query_position_embeddings
            # apply_rotary_pos_emb signature: (q, k, cos, sin) -> (q_rot, k_rot)
            # We feed Q in both slots and discard the K-rotated copy so we
            # only get Q rotated.
            query_states, _ = _apply_rotary_pos_emb(query_states, query_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        # NaN-guard kept from the pre-4.51 layout — long encoder sequences
        # in bf16 can occasionally produce NaN in the QK^T product.
        if torch.isnan(attn_weights).any():
            n_nan = torch.isnan(attn_weights).sum().item()
            n_total = attn_weights.numel()
            logger.warning(
                f"[HypernetCrossAttentionQwen NaN-guard] layer_idx={self.layer_idx} "
                f"NaN in attn_weights: {n_nan}/{n_total} ({100 * n_nan / n_total:.2f}%) "
                f"q_len={q_len} kv_len={kv_len} dtype={query_states.dtype}"
            )
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Encoder mask (padding on the K side).
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dim() == 2:
                # (B, kv_len) 0/1 -> additive mask broadcastable to (B, nh, q_len, kv_len)
                emask = encoder_attention_mask[:, None, None, :].to(attn_weights.dtype)
                emask = (1.0 - emask) * torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights + emask
            else:
                # already-additive mask (already includes -inf at padded positions)
                attn_weights = attn_weights + encoder_attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class HypernetDecoderLayerQwen(_QwenDecoderLayer):
    """Qwen-based hypernet decoder layer with an extra cross-attention sub-block.

    Order: self-attn -> cross-attn into policy residual -> MLP (matches
    the Gemma2 variant). Self-attn's `position_embeddings` come from the
    parent model (computed once per forward, not per layer). The cross-
    attention block reuses the same Q-side `position_embeddings`.
    """

    def __init__(self, config: _QwenConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.cross_attention = HypernetCrossAttentionQwen(config=config, layer_idx=layer_idx)
        self.pre_cross_attention_layernorm = _QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_cross_attention_layernorm = _QwenRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.Tensor] = None,
        base_encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Self attention. transformers >= 4.51 self_attn returns (h, attn_weights);
        # past_key_value is mutated in-place via cache, not returned.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self_attn_out = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        # Tolerate both 2-tuple (new) and 3-tuple (older) returns.
        if isinstance(self_attn_out, tuple) and len(self_attn_out) == 3:
            hidden_states, self_attn_weights, _present = self_attn_out
        else:
            hidden_states, self_attn_weights = self_attn_out
        hidden_states = residual + hidden_states

        # Cross-attention into the policy model's residual at the steer layer.
        residual = hidden_states
        hidden_states = self.pre_cross_attention_layernorm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attention(
            hidden_states=hidden_states,
            encoder_hidden_states=base_encoder_hidden_states,
            encoder_attention_mask=base_encoder_attention_mask,
            query_position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (past_key_value,)
        return outputs
