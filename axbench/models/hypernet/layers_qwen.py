"""Qwen2/Qwen3-based cross-attention + decoder layer for HyperSteer.

Mirrors `layers.py` (Gemma2 variant) with three substantive removals:
  * No `attn_logit_softcapping` (Qwen has no such field).
  * No sliding-window branching in HypernetDecoderLayer.forward
    (Qwen2's sliding window is handled differently; Qwen3 has none).
  * No GEMMA2_ATTENTION_CLASSES dict (it was dead code in the
    Gemma2 file too, and the Qwen attention backend is selected
    via `_attn_implementation` on the base Qwen{2,3}Attention class).

Imports prefer Qwen3 (recent transformers) and fall back to Qwen2
(older transformers, or Qwen3-route-via-Qwen2 in some configs).
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.utils import logging

# Prefer Qwen3 if available, otherwise fall back to Qwen2.
# Qwen3-8B in newer transformers (>=4.51) exposes its own classes; older
# transformers route Qwen3 through Qwen2 architecture.
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3Attention as _QwenAttention,
        Qwen3Config as _QwenConfig,
        Qwen3RMSNorm as _QwenRMSNorm,
        Qwen3DecoderLayer as _QwenDecoderLayer,
        Qwen3RotaryEmbedding as _QwenRotaryEmbedding,
        repeat_kv,
    )
    _QWEN_FAMILY = "qwen3"
except ImportError:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2Attention as _QwenAttention,
        Qwen2Config as _QwenConfig,
        Qwen2RMSNorm as _QwenRMSNorm,
        Qwen2DecoderLayer as _QwenDecoderLayer,
        Qwen2RotaryEmbedding as _QwenRotaryEmbedding,
        repeat_kv,
    )
    _QWEN_FAMILY = "qwen2"

from .configuration_hypernet_qwen import HypernetQwenConfig
from .utils import apply_rotary_pos_emb


logger = logging.get_logger(__name__)


class HypernetCrossAttentionQwen(_QwenAttention):
    """Cross-attention block for the Qwen-based hypernet.

    Q comes from `hidden_states` (the hypernet's residual at this layer);
    K and V come from `encoder_hidden_states` (the policy model's
    residual at the steer layer). Same shape contract as the Gemma2
    version, minus logits softcapping.
    """

    def __init__(self, config: _QwenConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        _, kv_len, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states = apply_rotary_pos_emb(query_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            # query mask along the q axis (Q's own padding)
            query_attention_mask = attention_mask[:, :, -1, :]
            # key mask from the encoder (policy-model padding on the encoder side)
            key_attention_mask = encoder_attention_mask.unsqueeze(1)

            query_attention_mask = query_attention_mask.unsqueeze(-1)
            key_attention_mask = key_attention_mask.unsqueeze(-2)

            causal_mask = torch.min(query_attention_mask, key_attention_mask)
            attn_weights = attn_weights + causal_mask

        # upcast softmax to fp32, cast back
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class HypernetDecoderLayerQwen(_QwenDecoderLayer):
    """Qwen-based hypernet decoder layer with an extra cross-attention sub-block.

    Identical structure to the Gemma2 variant but without the sliding-window
    mask manipulation (Qwen2's sliding window is per-layer config and the base
    `_QwenDecoderLayer` handles it; Qwen3 has none). The pre/post cross-
    attention norms are RMS norms, matching the rest of the Qwen stack.
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
        base_encoder_hidden_states: Optional[torch.Tensor] = None,
        base_encoder_attention_mask: Optional[torch.Tensor] = None,
        base_encoder_position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Cross-attention into the policy model's residual at the steer layer
        residual = hidden_states
        hidden_states = self.pre_cross_attention_layernorm(hidden_states)
        hidden_states, cross_attn_weights, _ = self.cross_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=base_encoder_hidden_states,
            encoder_attention_mask=base_encoder_attention_mask,
            encoder_position_ids=base_encoder_position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
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
            outputs += (present_key_value,)
        return outputs
