"""Build a HyperSteer v4 dataset using Autinn's 133 MCP-attack concepts as
DIVERSITY (not as multiple output dimensions).

The v3 single-concept design trained the hypernet on one concept_text. v4
exposes the hypernet to all 133 concept_text values from
`axbench/data/mcpattack.jsonl` so it learns to map ANY defense-strategy
text to a steering vector. Per Autinn's `mcp_hypersteer.py:320`,
`low_rank_dimension=1` is correct — the hypernet outputs a single
direction conditioned on whichever concept_text the caller passes at
inference.

Mapping policy:
  --map-strategy random
    Each diffmean rollout gets a uniformly-random concept_id from 1..133.
    Maximum diversity, simplest. Default.
  --map-strategy roundrobin
    rollout i gets concept_id = (i % 133) + 1. Even coverage.
  --map-strategy security_risk
    Map by `tags.security_risk` to a hand-curated subset of Autinn
    concepts (uses semantic alignment between security_risk and
    Autinn's attack_type taxonomy).

Usage:
    python axbench/mcp-protect/build_v4_dataset.py \\
      --in   diffmean/outputs/qwen3_thinking.flat.jsonl \\
             diffmean/outputs/qwen3_rollouts.labelled.jsonl \\
      --out  axbench/outputs/mcp_hsteer_qwen3_8b_v4_random \\
      --policy-model Qwen/Qwen3-8B --steer-layer 24 --variant action
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

MCP_ATTACK_PATH = Path("/home/ubuntu/mcp-protect/axbench/axbench/data/mcpattack.jsonl")

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
    return f"{sp}\n\n{uq}" if sp else uq


def split_think_action(text: str) -> tuple[str, str]:
    m = THINK_RE.search(text)
    if not m:
        return "", text.strip()
    think = m.group(1).strip()
    action = (text[: m.start()] + text[m.end():]).strip()
    return think, action


def project_completion(text: str, variant: str) -> str:
    think, action = split_think_action(text)
    if variant == "full":
        return f"<think>\n{think}\n</think>\n\n{action}".strip() if think else action
    if variant == "think":
        return f"<think>\n{think}\n</think>".strip() if think else ""
    if variant == "action":
        return action
    raise ValueError(variant)


# Map diffmean security_risk → keywords in Autinn's mcpattack concept text
SECURITY_RISK_KEYWORDS = {
    "Credential Leakage": ["ssh", "credential", "secret", "key", "password", "env", ".ssh", "id_rsa"],
    "Privacy Leakage": ["privacy", "personal", "leak", "exfil", "send", "external", "remote"],
    "Information Manipulation": ["manipulat", "tamper", "modify", "alter", "fake", "spoof", "deception"],
    "Service Disruption": ["disrupt", "deny", "delete", "destroy", "crash", "service", "availability"],
    "Infrastructure Damage": ["destroy", "delete", "rm", "infrastructure", "system", "boot", "kernel"],
    "Code Injection": ["inject", "code", "exec", "command", "shell", "script"],
    "Data Tampering": ["tamper", "modif", "alter", "data", "overwrit"],
    "Message Hijacking": ["hijack", "redirect", "replace", "intercept", "message"],
    "Instruction Tampering": ["instruct", "command", "tamper", "modif"],
    "Financial Loss": ["financial", "money", "payment", "transfer"],
    "Other": [],
}


def build_negative_examples(n: int, concept_text: str, concept_id: int) -> list[dict]:
    """Generic Q&A negatives (concept_id=-1 in axbench convention)."""
    pairs = [
        ("Who is Geling Yan?",
         "Geling Yan is a Chinese-American author best known for novels including The Banquet Bug and The Flowers of War."),
        ("In Python, print the first 10 Fibonacci numbers.",
         "```python\ndef fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a)\n        a, b = b, a+b\nfib(10)\n```"),
        ("What is the difference between AWS ECS and EKS?",
         "ECS is AWS's proprietary container orchestration service; EKS is managed Kubernetes."),
        ("Summarize the plot of Hamlet in two sentences.",
         "Prince Hamlet learns from his father's ghost that his uncle Claudius murdered the king. Hamlet's revenge plot ends with most of the cast dead."),
        ("Solve: if 3x + 5 = 17, what is x?",
         "Subtract 5: 3x = 12. Divide by 3: x = 4."),
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


def assign_concept(row: dict, autinn_concepts: list[dict], strategy: str, idx: int, rng: random.Random) -> dict:
    """Assign one of Autinn's 133 concepts to this row.
    Returns the chosen Autinn concept dict (concept_id, concept, ...)."""
    if strategy == "random":
        return rng.choice(autinn_concepts)
    if strategy == "roundrobin":
        return autinn_concepts[idx % len(autinn_concepts)]
    if strategy == "security_risk":
        sr = (row.get("tags") or {}).get("security_risk", "Other")
        keywords = SECURITY_RISK_KEYWORDS.get(sr, [])
        if not keywords:
            return rng.choice(autinn_concepts)
        # Score each Autinn concept by keyword overlap
        scored = []
        for ac in autinn_concepts:
            ctxt = ac["concept"].lower()
            score = sum(1 for k in keywords if k in ctxt)
            if score > 0:
                scored.append((score, ac))
        if scored:
            scored.sort(key=lambda x: -x[0])
            top_k = [s[1] for s in scored[:5]]  # pick from top-5 matches randomly
            return rng.choice(top_k)
        return rng.choice(autinn_concepts)
    raise ValueError(strategy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--variant", choices=["full", "think", "action"], default="action")
    ap.add_argument("--label-keep", type=str, default="resist")
    ap.add_argument("--map-strategy", choices=["random", "roundrobin", "security_risk"], default="security_risk",
                    help="How to assign Autinn concept_ids to each diffmean row.")
    ap.add_argument("--mcpattack-jsonl", type=Path, default=MCP_ATTACK_PATH)
    ap.add_argument("--n-negatives", type=int, default=24)
    ap.add_argument("--max-input-tokens", type=int, default=1024)
    ap.add_argument("--max-output-tokens", type=int, default=600)
    ap.add_argument("--tokenizer-model", type=str, default=None)
    ap.add_argument("--policy-model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--steer-layer", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.tokenizer_model:
        args.tokenizer_model = args.policy_model

    rng = random.Random(args.seed)

    # Load Autinn's 133 concepts
    autinn_concepts = [json.loads(l) for l in open(args.mcpattack_jsonl)]
    print(f"[v4] loaded {len(autinn_concepts)} Autinn mcp-attack concepts", file=sys.stderr)

    # Load + filter diffmean rollouts
    keep_labels = set(s.strip() for s in args.label_keep.split(","))
    seen_ids = set()
    rows_in = []
    for p in args.in_paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("id") in seen_ids:
                continue
            seen_ids.add(r.get("id"))
            rows_in.append(r)
    print(f"[v4] loaded {len(rows_in)} unique rows", file=sys.stderr)
    rows_in = [r for r in rows_in if r.get("label") in keep_labels]
    print(f"[v4] after label filter: {len(rows_in)}", file=sys.stderr)

    # Project to chosen variant
    def pick_target(r):
        return ((r.get("y_neg") or "").strip()) or ((r.get("y_pos") or "").strip())
    raw = []
    for r in rows_in:
        target = pick_target(r)
        if not target:
            continue
        projected = project_completion(target, args.variant)
        if not projected:
            continue
        raw.append((r, projected))
    print(f"[v4] after variant projection: {len(raw)}", file=sys.stderr)

    # Token-length filter
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False,
                                        model_max_length=max(args.max_input_tokens, args.max_output_tokens) * 4)
    kept = []
    drop_in, drop_out = 0, 0
    for r, target in raw:
        inp_text = build_input_text(r.get("system_prompt", ""), r.get("user_query", ""))
        n_in = len(tok(inp_text, add_special_tokens=False)["input_ids"])
        if n_in > args.max_input_tokens:
            drop_in += 1; continue
        n_out = len(tok(target, add_special_tokens=False)["input_ids"])
        if n_out > args.max_output_tokens:
            drop_out += 1; continue
        kept.append((r, target, inp_text))
    print(f"[v4] dropped {drop_in} over input, {drop_out} over output -> kept {len(kept)}", file=sys.stderr)
    if not kept:
        raise SystemExit("no rows kept")

    out_dir = args.out_dir
    gen_dir = out_dir / "generate"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Build positives with diverse concept assignment
    import collections
    positives = []
    concept_counts = collections.Counter()
    for idx, (r, target, inp) in enumerate(kept):
        ac = assign_concept(r, autinn_concepts, args.map_strategy, idx, rng)
        cid = int(ac["concept_id"])
        ctxt = ac["concept"]
        concept_counts[cid] += 1
        positives.append({
            "input": inp,
            "output": target,
            "output_concept": ctxt,
            "concept_genre": "code",
            "category": "positive",
            "dataset_category": "instruction",
            "concept_id": cid,
        })
    negatives = build_negative_examples(args.n_negatives, "EEEEE", -1)
    df = pd.DataFrame(positives + negatives)
    print(f"[v4] {len(positives)} positives + {len(negatives)} negatives = {len(df)} rows", file=sys.stderr)
    print(f"[v4] strategy={args.map_strategy}, distinct concept_ids used: {len(concept_counts)}", file=sys.stderr)
    print(f"[v4] concept_id distribution (top-10): {concept_counts.most_common(10)}", file=sys.stderr)

    df.to_parquet(gen_dir / "train_data.parquet", index=False)

    # metadata.jsonl: one row per Autinn concept (all 133)
    with (gen_dir / "metadata.jsonl").open("w") as f:
        for ac in autinn_concepts:
            cid = int(ac["concept_id"])
            ctxt = ac["concept"]
            f.write(json.dumps({
                "concept_id": cid,
                "concept": ctxt,
                "ref": ac.get("ref", "MCPTox"),
                "concept_genres_map": {ctxt: ["code"]},
            }) + "\n")

    # merged_concepts_mcp.json: same 133 in Autinn's flat array shape
    merged = []
    for ac in autinn_concepts:
        merged.append({
            "modelId": args.policy_model,
            "layer": "mcp-attack-v4",
            "index": int(ac["concept_id"]) - 1,  # 0-based index
            "description": ac["concept"],
            "ref": ac.get("ref", "MCPTox"),
            "concept_id": int(ac["concept_id"]),
        })
    (out_dir / "merged_concepts_mcp.json").write_text(json.dumps(merged, indent=2))

    # YAML
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

    print(f"[v4] wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
