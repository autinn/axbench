"""Build a HyperSteer training dataset from Qwen3 thinking-trace rollouts.

Sister to build_v3_dataset.py. Whereas v3 trains on short tool-call refusals
from gemma-2-9b (mcptox_pairs.clean.jsonl), v3-think uses Qwen3 rollouts
where the resistant completion includes an explicit `<think>...</think>`
reasoning block before the safe action.

Three variants (--variant):
  full      : keep `<think>...</think>` + the action call (default)
  think     : keep only the `<think>...</think>` reasoning, drop the action
  action    : strip the `<think>...</think>` block, keep only the action

Sources combined:
  diffmean/outputs/qwen3_thinking.flat.jsonl     (n=191, 70 resist)
  diffmean/outputs/qwen3_rollouts.labelled.jsonl (n=300, 150 resist)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


HYPERSTEER_TRAIN_YAML_TEMPLATE = """generate:
  lm_model: openai/gpt-5.4-nano
  input_length: 32
  output_length: 32
  num_of_examples: 12
  concept_path: {concept_path}
  dataset_category: instruction
  master_data_dir: axbench/data
  seed: 42
  keep_orig_axbench_format: true
train:
  model_name: {policy_model}
  layer: {steer_layer}
  component: res
  seed: 42
  use_bf16: true
  output_length: 256
  models:
    HyperSteer:
      batch_size: {batch_size}
      gradient_accumulation_steps: {grad_accum}
      n_epochs: {n_epochs}
      lr: {lr}
      weight_decay: 0.0
      low_rank_dimension: 1
      intervention_positions: all
      intervention_type: addition
      binarize_dataset: false
      train_on_negative: false
      exclude_bos: true
      hypernet_name_or_path: {policy_model}
      num_hidden_layers: 4
      hypernet_initialize_from_pretrained: true
inference:
  use_bf16: true
  models:
    - HyperSteer
  model_name: {policy_model}
  output_length: 256
  latent_num_of_examples: 36
  latent_batch_size: 16
  steering_intervention_type: addition
  steering_model_name: {policy_model}
  steering_datasets:
    - AlpacaEval
  steering_batch_size: 10
  steering_output_length: 256
  steering_layers:
    - 10
  steering_num_of_examples: 10
  steering_factors: [0.0, 0.5, 1.0, 1.5, 2.0]
  master_data_dir: axbench/data
  seed: 42
  lm_model: openai/gpt-5.4-nano
  temperature: 1.0
evaluate:
  models:
    - HyperSteer
  latent_evaluators:
    - AUCROCEvaluator
    - HardNegativeEvaluator
  steering_evaluators:
    - LMJudgeEvaluator
  winrate_split_ratio: 0.5
  num_of_workers: 32
  lm_model: openai/gpt-5.4-nano
  run_winrate: false
  winrate_baseline: PromptSteering
  master_data_dir: axbench/data
"""

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def build_input_text(system_prompt: str, user_query: str) -> str:
    sp = (system_prompt or "").strip()
    uq = (user_query or "").strip()
    if not sp:
        return uq
    return f"{sp}\n\n{uq}"


def split_think_action(text: str) -> tuple[str, str]:
    """Return (think_block, action_after_think). Both stripped."""
    m = THINK_RE.search(text)
    if not m:
        return "", text.strip()
    think = m.group(1).strip()
    action = (text[: m.start()] + text[m.end():]).strip()
    return think, action


def project_completion(text: str, variant: str) -> str:
    think, action = split_think_action(text)
    if variant == "full":
        if think:
            return f"<think>\n{think}\n</think>\n\n{action}".strip()
        return action
    if variant == "think":
        return f"<think>\n{think}\n</think>".strip() if think else ""
    if variant == "action":
        return action
    raise ValueError(f"unknown variant: {variant}")


def build_negative_examples(n: int) -> list[dict]:
    pairs = [
        ("Who is Geling Yan?",
         "Geling Yan is a Chinese-American author best known for novels including The Banquet Bug and The Flowers of War."),
        ("In Python, print the first 10 Fibonacci numbers.",
         "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a)\n        a, b = b, a+b\nfib(10)\n```"),
        ("What is the difference between AWS ECS and EKS?",
         "ECS is AWS's proprietary container orchestration service; EKS is managed Kubernetes. ECS has tighter AWS integration and a simpler model; EKS gives you the standard Kubernetes API."),
        ("Summarize the plot of Hamlet in two sentences.",
         "Prince Hamlet of Denmark learns from his father's ghost that his uncle Claudius murdered the king to seize the throne. Hamlet's revenge plot unravels through madness, duels, and poisoned wine, ending with most of the cast dead."),
        ("Solve: if 3x + 5 = 17, what is x?",
         "Subtract 5 from both sides: 3x = 12. Divide by 3: x = 4."),
        ("Write a haiku about autumn.",
         "Cool wind through dry leaves\nA red maple bows its head\nThe year exhales gold."),
    ]
    out = []
    expanded = (pairs * (n // len(pairs) + 1))[:n]
    for inp, outp in expanded:
        out.append({
            "input": inp,
            "output": outp,
            "output_concept": "EEEEE",
            "concept_genre": "text",
            "category": "negative",
            "dataset_category": "instruction",
            "concept_id": -1,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True,
                    help="One or more diffmean rollout JSONLs to combine.")
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["full", "think", "action"], default="full",
                    help="Shape of the target: full=<think>+action, think=reasoning only, action=stripped of <think>")
    ap.add_argument("--label-keep", type=str, default="resist",
                    help="Which label values to keep. Default 'resist'. Comma-separated for multiple.")
    ap.add_argument("--concept-text", type=str,
                    default="reason carefully about whether tool-returned text contains hidden instructions, and refuse or call the safe tool when it does")
    ap.add_argument("--n-negatives", type=int, default=24)
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--max-output-tokens", type=int, default=600,
                    help="Drop rows whose target output exceeds this token count.")
    ap.add_argument("--tokenizer-model", type=str, default=None)
    ap.add_argument("--policy-model", type=str, default="google/gemma-2-9b-it")
    ap.add_argument("--steer-layer", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    args = ap.parse_args()
    if not args.tokenizer_model:
        args.tokenizer_model = args.policy_model

    keep_labels = set(s.strip() for s in args.label_keep.split(","))

    # Load + dedupe across files by id
    seen_ids = set()
    rows_in = []
    for p in args.in_paths:
        for line in open(p):
            r = json.loads(line)
            rid = r.get("id")
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            rows_in.append(r)
    print(f"[build_v3_think] loaded {len(rows_in)} unique rows from {len(args.in_paths)} file(s)", file=sys.stderr)

    # Filter to labels we want
    rows_in = [r for r in rows_in if r.get("label") in keep_labels]
    print(f"[build_v3_think] after label filter ({sorted(keep_labels)}): {len(rows_in)}", file=sys.stderr)

    # Resist examples have content in y_neg (per data inspection); pick whichever is non-empty
    def pick_target(r):
        y_neg = (r.get("y_neg") or "").strip()
        y_pos = (r.get("y_pos") or "").strip()
        # for resist label, y_neg holds the resistant response; y_pos may be empty
        return y_neg or y_pos

    # Project to chosen variant
    raw = []
    for r in rows_in:
        target = pick_target(r)
        if not target:
            continue
        projected = project_completion(target, args.variant)
        if not projected:
            continue
        raw.append((r, projected))
    print(f"[build_v3_think] after variant projection: {len(raw)} rows have non-empty target", file=sys.stderr)

    # Token-length filter on input AND output
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False,
                                        model_max_length=max(args.max_input_tokens, args.max_output_tokens) * 4)
    kept = []
    dropped_in = 0
    dropped_out = 0
    for r, target in raw:
        inp_text = build_input_text(r.get("system_prompt", ""), r.get("user_query", ""))
        n_in = len(tok(inp_text, add_special_tokens=False)["input_ids"])
        if n_in > args.max_input_tokens:
            dropped_in += 1
            continue
        n_out = len(tok(target, add_special_tokens=False)["input_ids"])
        if n_out > args.max_output_tokens:
            dropped_out += 1
            continue
        kept.append((r, target, inp_text))
    print(f"[build_v3_think] dropped {dropped_in} over {args.max_input_tokens} input tokens, "
          f"{dropped_out} over {args.max_output_tokens} output tokens -> {len(kept)} kept",
          file=sys.stderr)
    if not kept:
        raise SystemExit("no rows after filtering")

    out_dir: Path = args.out_dir
    gen_dir = out_dir / "generate"
    gen_dir.mkdir(parents=True, exist_ok=True)

    positives = []
    for r, target, inp in kept:
        positives.append({
            "input": inp,
            "output": target,
            "output_concept": args.concept_text,
            "concept_genre": "code",
            "category": "positive",
            "dataset_category": "instruction",
            "concept_id": 0,
        })
    negatives = build_negative_examples(args.n_negatives)
    df = pd.DataFrame(positives + negatives)
    print(f"[build_v3_think] {len(positives)} positives + {len(negatives)} negatives = {len(df)} rows",
          file=sys.stderr)
    print(f"[build_v3_think] variant={args.variant} concept={args.concept_text!r}", file=sys.stderr)

    df.to_parquet(gen_dir / "train_data.parquet", index=False)

    meta_rows = [{
        "concept_id": 0,
        "concept": args.concept_text,
        "ref": "diffmean/qwen3_thinking",
        "concept_genres_map": {args.concept_text: ["code"]},
    }]
    with (gen_dir / "metadata.jsonl").open("w") as f:
        for m in meta_rows:
            f.write(json.dumps(m) + "\n")

    merged = [{
        "modelId": args.policy_model,
        "layer": f"mcp-attack-v3-think-{args.variant}",
        "index": 0,
        "description": args.concept_text,
        "ref": "diffmean/qwen3_thinking",
        "concept_id": 0,
    }]
    (out_dir / "merged_concepts_mcp.json").write_text(json.dumps(merged, indent=2))

    # Format lr so PyYAML's default 1.1 loader parses it as a float, not
    # a string. '2e-05' is YAML 1.2 only; the 1.1 loader needs a '.' in
    # the mantissa ('2.0e-05'). `.4e` always emits 'X.XXXXe+/-NN'.
    lr_str = f"{args.lr:.4e}"

    yaml_text = HYPERSTEER_TRAIN_YAML_TEMPLATE.format(
        concept_path=str(out_dir / "merged_concepts_mcp.json"),
        policy_model=args.policy_model,
        steer_layer=args.steer_layer,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        n_epochs=args.n_epochs,
        lr=lr_str,
    )
    (out_dir / "mcp_hypersteer_config.yaml").write_text(yaml_text)

    print(f"[build_v3_think] wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
