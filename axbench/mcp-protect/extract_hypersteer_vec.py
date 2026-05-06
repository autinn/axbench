#!/usr/bin/env python3
"""Extract a static steering vector from a trained HyperSteer dump.

The hypernet (``HyperSteer.concept_embedding``) maps ``(concept_text,
base_encoder_hidden_states) -> v[hidden_dim]``. ``v`` therefore depends on the
user input. For multi-vector composition (D12/D7) we want a *static* vector
per (dump, concept_id) pair, so we marginalize the input dependence by feeding
a fixed neutral prompt. The resulting tensor is the same shape (``[hidden_dim]``)
that ``SimpleAdditionIntervention`` adds to ``model.layers[L].output`` at every
position — see ``axbench/models/interventions.py::SimpleAdditionIntervention``.

The concept->v function is ``HyperSteer.concept_embedding(...).last_hidden_state``
(a Hypernet{Qwen,}Model forward pass — see ``axbench/models/hypersteer.py:494``
in ``predict_steer`` for the canonical call site).

Usage::

    python axbench/mcp-protect/extract_hypersteer_vec.py \
        --dump-dir axbench/outputs/mcp_hsteer_qwen3_8b_v17_terse \
        --concept-id 0 \
        --out /root/v17_vec.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import atexit
import datetime
import socket
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

# Same package-root insertion as serve_mcp_hypersteer.py.
_SCRIPT = Path(__file__).resolve()
_AX_PKG_PARENT = _SCRIPT.parents[2]  # .../axbench (contains axbench/ package)
if str(_AX_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_AX_PKG_PARENT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import axbench  # noqa: E402,F401  (registers HyperSteer in package namespace)
from axbench.utils.constants import CHAT_MODELS  # noqa: E402
from axbench.utils.model_utils import get_prefix_length  # noqa: E402
from axbench.scripts.args.training_args import ModelParams  # noqa: E402


# Neutral prompt fed as base input so the hypernet's cross-attention has
# something to attend to. Empirically the v vector is dominated by concept_text
# (cross-attn from concept tokens to base hidden states is the conditioning),
# so any reasonable prompt yields a usable static direction. We use a short
# benign instruction; if you suspect input-conditioning matters, regenerate
# vectors per-prompt instead of using this static extraction path.
_NEUTRAL_PROMPT = "Hello, please respond briefly."


def _load_metadata_rows(generate_dir: Path) -> list[dict[str, Any]]:
    path = generate_dir / "metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hypersteer_model_params(
    train_cfg: dict[str, Any], model_name: str = "HyperSteer"
) -> ModelParams:
    raw = (train_cfg.get("models") or {})[model_name]
    data = {k: v for k, v in raw.items() if k in {f.name for f in fields(ModelParams)}}
    return ModelParams(**data)


def _ensure_dist() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=1,
        rank=0,
        timeout=datetime.timedelta(seconds=120),
    )

    def _destroy() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    atexit.register(_destroy)


def _axbench_dump_root(dump_dir: Path) -> Path:
    p = dump_dir.resolve()
    if (p / "train").is_dir() and (p / "generate").is_dir():
        return p
    if (p.parent / "train").is_dir() and (p.parent / "generate").is_dir():
        return p.parent
    raise FileNotFoundError(
        f"Expected --dump-dir to be a run dir with train/ and generate/, got: {dump_dir}"
    )


def _build_hsteer(train_cfg, train_dir, generate_dir, device):
    model_name = train_cfg["model_name"]
    layer = int(train_cfg["layer"])
    use_bf16 = bool(train_cfg.get("use_bf16", True))
    hparams = _hypersteer_model_params(train_cfg)
    meta_rows = _load_metadata_rows(generate_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, model_max_length=8192
    )
    tokenizer.padding_side = "right"
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dtype = torch.bfloat16 if use_bf16 else None
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()

    h = axbench.HyperSteer(
        model_instance,
        tokenizer,
        layer=layer,
        low_rank_dimension=len(meta_rows),
        device=device,
        training_args=hparams,
        lm_model_name=model_name,
    )
    h.load(
        dump_dir=str(train_dir),
        low_rank_dimension=1,
        mode="steering",
        hypernet_initialize_from_pretrained=hparams.hypernet_initialize_from_pretrained,
        hypernet_name_or_path=hparams.hypernet_name_or_path,
        num_hidden_layers=hparams.num_hidden_layers,
    )
    h.concept_embedding.eval()
    return h, model_name, layer


def _resolve_concept_text(generate_dir: Path, dump_dir: Path, concept_id: int) -> str:
    """Look up concept text by id. Prefer generate/metadata.jsonl, fall back
    to merged_concepts_mcp.json at the dump root."""
    rows = _load_metadata_rows(generate_dir)
    by_id = {int(r["concept_id"]): r["concept"] for r in rows if "concept_id" in r and "concept" in r}
    if concept_id in by_id:
        return by_id[concept_id]

    merged = dump_dir / "merged_concepts_mcp.json"
    if merged.is_file():
        with merged.open("r", encoding="utf-8") as f:
            for entry in json.load(f):
                if int(entry.get("concept_id", -1)) == concept_id:
                    desc = entry.get("description") or entry.get("concept")
                    if desc:
                        return desc
    raise SystemExit(
        f"concept_id {concept_id} not found in {generate_dir/'metadata.jsonl'} "
        f"or {merged} (available: {sorted(by_id.keys())[:20]})"
    )


@torch.no_grad()
def extract_vec(h, layer: int, concept_text: str, device: str) -> torch.Tensor:
    """Run the canonical concept_embedding(...) forward (matches predict_steer
    in axbench/models/hypersteer.py) on a fixed neutral prompt."""
    base_input = h.tokenizer(
        _NEUTRAL_PROMPT, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    concept_input = h.hypernet_tokenizer(
        concept_text, return_tensors="pt", add_special_tokens=True,
        padding=True, truncation=True,
    ).to(device)

    base_hidden = h.model(
        input_ids=base_input["input_ids"],
        attention_mask=base_input["attention_mask"],
        output_hidden_states=True,
    ).hidden_states[layer]

    v = h.concept_embedding(
        input_ids=concept_input["input_ids"],
        inputs_embeds=None,
        attention_mask=concept_input["attention_mask"],
        base_encoder_hidden_states=base_hidden,
        base_encoder_attention_mask=base_input["attention_mask"],
        output_hidden_states=False,
    ).last_hidden_state  # [batch=1, hidden_dim]

    return v.squeeze(0).detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract a static HyperSteer steering vector")
    ap.add_argument("--dump-dir", type=Path, required=True,
                    help="Run dir containing train/ and generate/ (same as serve_mcp_hypersteer)")
    ap.add_argument("--concept-id", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="Output .pt path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--config", type=Path, default=None,
                    help="Optional explicit YAML; default <dump-dir>/mcp_hypersteer_config.yaml")
    args = ap.parse_args()

    _ensure_dist()
    run_root = _axbench_dump_root(args.dump_dir)
    cfg_path = args.config or (run_root / "mcp_hypersteer_config.yaml")
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg.get("train") or {}

    h, model_name, layer = _build_hsteer(
        train_cfg, run_root / "train", run_root / "generate", args.device
    )
    concept_text = _resolve_concept_text(
        run_root / "generate", run_root, args.concept_id
    )
    print(f"[extract] dump={run_root.name} cid={args.concept_id} text={concept_text!r}")

    vec = extract_vec(h, layer, concept_text, args.device)
    payload = {
        "vec": vec,
        "concept_text": concept_text,
        "source_dir": str(run_root),
        "concept_id": int(args.concept_id),
        "hidden_dim": int(vec.shape[-1]),
        "model_name": model_name,
        "layer": int(layer),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)

    print(f"[extract] saved -> {args.out}")
    print(f"[extract] shape={tuple(vec.shape)} dtype={vec.dtype} norm={vec.float().norm().item():.4f}")


if __name__ == "__main__":
    main()
