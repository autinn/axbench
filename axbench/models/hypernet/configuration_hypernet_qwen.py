"""Hypernet config for Qwen-based policy models (Qwen2 / Qwen3).

Mirrors `configuration_hypernet.py` (Gemma2 variant) but with defaults
shaped for Qwen3-8B and without Gemma2-specific fields
(`final_logit_softcapping`, `attn_logit_softcapping`, `sliding_window`,
`cache_implementation="hybrid"`, `query_pre_attn_scalar`).

Defaults are mostly placeholders — `from_pretrained` overrides them
from the checkpoint's config.json. The field set matters more than
the values.
"""

from transformers import PretrainedConfig


class HypernetQwenConfig(PretrainedConfig):

    model_type = "hypernet_qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        use_target_model_embedding: bool = True,
        # Qwen3-8B defaults (will be overridden by from_pretrained from the
        # actual checkpoint's config.json):
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=12288,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=151643,
        eos_token_id=151645,
        bos_token_id=None,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=None,
        use_sliding_window=False,
        max_window_layers=None,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.use_target_model_embedding = use_target_model_embedding
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.use_sliding_window = use_sliding_window
        self.max_window_layers = max_window_layers
        self.rope_scaling = rope_scaling
