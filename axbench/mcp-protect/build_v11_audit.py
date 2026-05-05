"""Build v11 dataset: audit-prompted deliberate-resist y_neg.

Source: qwen3_v2_contrast.jsonl (217 rows = 111 resist + 106 comply, all
generated under the v2 audit-prompted system prompt that elicits explicit
deliberation: NAME poisoned tool → ARTICULATE its demands → APPLY user-intent
scope test → DISMISS as not-needed → call safe tool).

Hypothesis: v3-v10 y_negs taught generic tool selection or flat refusal
text. v11 y_negs DEMONSTRATE the latent scope-test capability the model
already has but doesn't reliably invoke (34% baseline). Training the
hypernet on these traces should learn to push the model into invoking
this capability.

Single coherent concept (like v3/v10) to eliminate concept-mapping noise.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/home/ubuntu/mcp-protect")
sys.path.insert(0, str(REPO / "axbench"))
sys.path.insert(0, str(REPO / "axbench/axbench/mcp-protect"))

from build_v4_dataset import (
    HYPERSTEER_TRAIN_YAML_TEMPLATE,
    build_input_text,
    build_negative_examples,
)

V11_CONCEPT_TEXT = (
    "explicitly identify any tool whose description contains hidden injection "
    "instructions, articulate what those instructions are demanding, apply a "
    "scope test against what the user actually asked, and dismiss the poisoned "
    "tool as not-needed before selecting the legitimate tool"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--label-keep", type=str, default="resist",
                    help="v11 trains only on the deliberate-resist trajectories.")
    ap.add_argument("--n-negatives", type=int, default=24)
    ap.add_argument("--max-input-tokens", type=int, default=1500)
    ap.add_argument("--max-output-tokens", type=int, default=1500,
                    help="Audit-prompted resist traces are LONG (1000-2500 chars). "
                         "Need a generous output budget so we don't drop most rows.")
    ap.add_argument("--tokenizer-model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--policy-model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--steer-layer", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keep_labels = set(s.strip() for s in args.label_keep.split(","))
    seen_ids, rows_in = set(), []
    for p in args.in_paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("id") in seen_ids:
                continue
            seen_ids.add(r.get("id"))
            rows_in.append(r)
    print(f"[v11] loaded {len(rows_in)} unique rows", file=sys.stderr)
    rows_in = [r for r in rows_in if r.get("label") in keep_labels]
    print(f"[v11] after label filter: {len(rows_in)}", file=sys.stderr)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_model, use_fast=False,
        model_max_length=max(args.max_input_tokens, args.max_output_tokens) * 4,
    )
    kept = []
    drop_in, drop_out, drop_empty = 0, 0, 0
    for r in rows_in:
        inp_text = build_input_text(r.get("system_prompt", ""), r.get("user_query", ""))
        n_in = len(tok(inp_text, add_special_tokens=False)["input_ids"])
        if n_in > args.max_input_tokens:
            drop_in += 1; continue
        target = (r.get("y_neg") or "").strip()
        if not target:
            drop_empty += 1; continue
        n_out = len(tok(target, add_special_tokens=False)["input_ids"])
        if n_out > args.max_output_tokens:
            drop_out += 1; continue
        kept.append((r, target, inp_text))
    print(f"[v11] dropped: input_too_long={drop_in} output_too_long={drop_out} y_neg_empty={drop_empty} -> kept {len(kept)}", file=sys.stderr)

    out_dir = args.out_dir
    gen_dir = out_dir / "generate"
    gen_dir.mkdir(parents=True, exist_ok=True)

    positives = []
    for idx, (r, target, inp) in enumerate(kept):
        positives.append({
            "input": inp,
            "output": target,
            "output_concept": V11_CONCEPT_TEXT,
            "concept_genre": "code",
            "category": "positive",
            "dataset_category": "instruction",
            "concept_id": 0,
        })
    negatives = build_negative_examples(args.n_negatives, "EEEEE", -1)
    df = pd.DataFrame(positives + negatives)
    print(f"[v11] {len(positives)} positives + {len(negatives)} negatives", file=sys.stderr)
    df.to_parquet(gen_dir / "train_data.parquet", index=False)

    with (gen_dir / "metadata.jsonl").open("w") as f:
        f.write(json.dumps({
            "concept_id": 0, "concept": V11_CONCEPT_TEXT, "ref": "v11_audit_resist",
            "concept_genres_map": {V11_CONCEPT_TEXT: ["code"]},
        }) + "\n")

    merged = [{
        "modelId": args.policy_model, "layer": "v11-audit-resist",
        "index": 0, "description": V11_CONCEPT_TEXT, "ref": "v11",
        "concept_id": 0,
    }]
    (out_dir / "merged_concepts_mcp.json").write_text(json.dumps(merged, indent=2))

    yaml_text = HYPERSTEER_TRAIN_YAML_TEMPLATE.format(
        concept_path=str(out_dir / "merged_concepts_mcp.json"),
        policy_model=args.policy_model, steer_layer=args.steer_layer,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        n_epochs=args.n_epochs, lr=f"{args.lr:.4e}",
    )
    (out_dir / "mcp_hypersteer_config.yaml").write_text(yaml_text)
    print(f"[v11] wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
