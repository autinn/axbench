"""Register Qwen3 in pyvene's intervenable_modelcard.

pyvene 0.1.8 (the version installed on Box B) does not include Qwen3 in its
`type_to_module_mapping` / `type_to_dimension_mapping` registries. Without an
entry, pyvene's `getattr_for_torch_module` returns an empty path string and
fails with:

    AttributeError: 'Qwen3ForCausalLM' object has no attribute ''

axbench's LoReFT class hardcodes `component: "block_output"` in
`pyreft.ReftConfig`, so it relies on pyvene to map that string to a module
path.

Qwen3 has the same layer structure as Qwen2 (`model.layers[i]` with
`self_attn.{q,k,v,o}_proj` and `mlp.{up,down,gate}_proj` + `mlp.act_fn`), so we
copy Qwen2's mappings verbatim.

Importing this module (top of program, before any pyreft / pyvene patching of
the registries) is sufficient to register Qwen3.
"""
from __future__ import annotations


def _patch() -> None:
    import transformers.models as hf_models
    from pyvene.models import intervenable_modelcard as imc
    from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
        qwen2_type_to_module_mapping,
        qwen2_type_to_dimension_mapping,
        qwen2_lm_type_to_module_mapping,
        qwen2_lm_type_to_dimension_mapping,
        qwen2_classifier_type_to_module_mapping,
        qwen2_classifier_type_to_dimension_mapping,
    )

    try:
        qwen3_mod = hf_models.qwen3.modeling_qwen3
    except AttributeError as exc:
        raise RuntimeError(
            "transformers does not expose qwen3.modeling_qwen3; "
            "upgrade transformers."
        ) from exc

    qwen3_classes_module = (
        qwen3_mod.Qwen3Model,
        qwen3_mod.Qwen3ForCausalLM,
    )
    # Qwen3ForSequenceClassification may not exist in every transformers
    # release; only register it if present.
    qwen3_seq_cls = getattr(qwen3_mod, "Qwen3ForSequenceClassification", None)

    module_entries = [
        (qwen3_mod.Qwen3Model, qwen2_type_to_module_mapping),
        (qwen3_mod.Qwen3ForCausalLM, qwen2_lm_type_to_module_mapping),
    ]
    dim_entries = [
        (qwen3_mod.Qwen3Model, qwen2_type_to_dimension_mapping),
        (qwen3_mod.Qwen3ForCausalLM, qwen2_lm_type_to_dimension_mapping),
    ]
    if qwen3_seq_cls is not None:
        module_entries.append(
            (qwen3_seq_cls, qwen2_classifier_type_to_module_mapping)
        )
        dim_entries.append(
            (qwen3_seq_cls, qwen2_classifier_type_to_dimension_mapping)
        )

    added = []
    for cls, mapping in module_entries:
        if cls not in imc.type_to_module_mapping:
            imc.type_to_module_mapping[cls] = mapping
            added.append(cls.__name__)
    for cls, mapping in dim_entries:
        if cls not in imc.type_to_dimension_mapping:
            imc.type_to_dimension_mapping[cls] = mapping

    print(
        f"[patch_pyvene_qwen3] registered Qwen3 classes in pyvene "
        f"type_to_module_mapping: {added or 'already present'}"
    )


_patch()
