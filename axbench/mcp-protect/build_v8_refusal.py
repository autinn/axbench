"""Build v8 dataset: synthetic-refusal y_neg.

Hypothesis from headlines run: every variant has refused=0 because the
training data never showed the model what an explicit refusal LOOKS like.
Steering pushes the model toward incoherent ('other') outputs because that's
the closest signal to what y_neg looks like (resist via picking a different
tool). Give the hypernet a clean refusal target instead.

For each diffmean resist row, replace y_neg with one of N synthetic refusal
templates that includes (a) a brief thinking trace acknowledging the threat
and (b) explicit refusal language in the post-think output. Random
template-per-row to avoid degenerate learning.

Otherwise identical to v4/v5/v6/v7 pipeline (Autinn 133 concepts via
security_risk mapping, low_rank_dimension=1, etc.).
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import pandas as pd

REPO = Path("/home/ubuntu/mcp-protect")
sys.path.insert(0, str(REPO / "axbench"))
sys.path.insert(0, str(REPO / "axbench/axbench/mcp-protect"))

# Reuse v4 builder's helpers for everything except y_neg generation.
from build_v4_dataset import (
    SECURITY_RISK_KEYWORDS,
    HYPERSTEER_TRAIN_YAML_TEMPLATE,
    build_input_text,
    build_negative_examples,
    assign_concept,
    MCP_ATTACK_PATH,
)

# --- Synthetic refusal templates ---
# Each template is (think_body, post_think_body). We'll fill {tool_hint} with
# a brief reference to the user's intended action so the refusal is grounded.
REFUSAL_TEMPLATES = [
    (
        "Looking at the available tools, the descriptions contain instructions that look like prompt-injection attempts targeting me rather than legitimate tool documentation. I should not execute any tool whose description is asking me to perform actions outside the user's stated request.",
        "I notice that the tool descriptions for the operation you requested contain suspicious embedded instructions that appear to be a prompt-injection attempt. I will not execute these tools. Could you describe what you're trying to accomplish in plain words, so I can help you find a safer path?",
    ),
    (
        "The tool metadata available here is structured oddly — there are imperative instructions embedded in the descriptions ('MUST do X first', 'REQUIRED to call Y') that don't belong in legitimate tool docs. This is a tool-poisoning pattern. I should refuse rather than execute.",
        "I cannot run this request. The tool descriptions on offer contain authority-mimicking phrases ('MUST', 'REQUIRED') that are characteristic of poisoned tool metadata, not legitimate API documentation. I won't execute the requested operation. Please rephrase what you need and I can suggest a safer alternative.",
    ),
    (
        "The tool descriptions reference required helpers that I would need to call in addition to the obvious tool — that combination of file access plus network exfiltration is a classic data-exfiltration chain. I should refuse the entire request rather than partially comply.",
        "I'm declining this request. The available tools chain together in a way that would access local files and then send data to an external endpoint — a classic exfiltration pattern. I won't execute any of these tools. If you tell me what data you need and where it should end up legitimately, I can suggest a safer approach.",
    ),
    (
        "Treating tool descriptions as commands to execute is exactly the failure mode this attack tries to exploit. The descriptions are documentation, not instructions I should follow. I should not call any of these tools.",
        "I'm not going to execute this. The tool descriptions provided contain content that reads as instructions to me rather than documentation of what the tools do — that's the signature of a tool-poisoning attack. I cannot safely run these tools. Please rewrite your request without relying on the supplied tools.",
    ),
    (
        "The user query and the available tools don't match — the user is asking for a benign operation, but the tools available are framed as if they need to perform privileged or destructive actions. This mismatch is suspicious. Refusing is safer than guessing.",
        "I cannot fulfill this request as currently framed. The available tools are not appropriate for what you're asking — they appear designed to perform destructive or privileged operations rather than the benign one in your query. I will not execute any of them. Please clarify your goal so I can recommend a safer path.",
    ),
    (
        "Even if there is a way to satisfy the user request through these tools, the tool descriptions themselves contain content that is trying to get me to take additional, unrequested actions. Partial compliance (just doing the user's part) still risks executing the injected actions. I should refuse.",
        "I'm refusing this request. The tools available carry embedded instructions that would have me perform actions you didn't request. Even a 'safe' subset of tool calls would still touch the poisoned descriptions. I will not proceed. Please describe the underlying task in your own words and I'll suggest tools you can run yourself.",
    ),
    (
        "I don't trust the tool list supplied here. Multiple tools are described in a way that conflates legitimate functionality with hidden side-effects (e.g., 'reads file AND uploads to remote server'). I should not run any of them.",
        "I won't execute any tool from the supplied list. Multiple descriptions conflate legitimate functionality with hidden side-effects (file access plus remote upload, command execution plus credential reading). This is not a safe environment. Please describe what you need; I can tell you what a safe tool set would look like.",
    ),
    (
        "There is no version of this request I can safely fulfill given the supplied tools. The right response is to flag the situation to the user, not to attempt the task with broken tools.",
        "I can't help with this as stated. The tools supplied to me look like they've been tampered with — there's no safe subset I can use to complete your request. I'm flagging this back to you rather than guessing. Could you provide a fresh tool list, or describe what you're trying to do without referencing these tools?",
    ),
]


def synthetic_refusal_y_neg(rng: random.Random) -> str:
    """Return a full <think>...</think>{refusal} string."""
    think_body, post_body = rng.choice(REFUSAL_TEMPLATES)
    return f"<think>\n{think_body}\n</think>\n\n{post_body}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    ap.add_argument("--out", dest="out_dir", type=Path, required=True)
    ap.add_argument("--label-keep", type=str, default="resist,comply",
                    help="v8 trains on resist+comply rows alike — the y_neg "
                         "is synthetic so the original label doesn't matter, "
                         "we want the largest pool of (system_prompt, user_query) inputs")
    ap.add_argument("--map-strategy", choices=["random", "roundrobin", "security_risk"],
                    default="security_risk")
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
    autinn_concepts = [json.loads(l) for l in open(args.mcpattack_jsonl)]
    print(f"[v8] loaded {len(autinn_concepts)} Autinn mcp-attack concepts", file=sys.stderr)

    keep_labels = set(s.strip() for s in args.label_keep.split(","))
    seen_ids, rows_in = set(), []
    for p in args.in_paths:
        for line in open(p):
            r = json.loads(line)
            if r.get("id") in seen_ids:
                continue
            seen_ids.add(r.get("id"))
            rows_in.append(r)
    print(f"[v8] loaded {len(rows_in)} unique rows", file=sys.stderr)
    rows_in = [r for r in rows_in if r.get("label") in keep_labels]
    print(f"[v8] after label filter: {len(rows_in)}", file=sys.stderr)

    # Token-length filter on (input, synthetic_refusal_target)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer_model, use_fast=False,
        model_max_length=max(args.max_input_tokens, args.max_output_tokens) * 4,
    )
    kept = []
    drop_in, drop_out = 0, 0
    for r in rows_in:
        inp_text = build_input_text(r.get("system_prompt", ""), r.get("user_query", ""))
        n_in = len(tok(inp_text, add_special_tokens=False)["input_ids"])
        if n_in > args.max_input_tokens:
            drop_in += 1; continue
        target = synthetic_refusal_y_neg(rng)
        n_out = len(tok(target, add_special_tokens=False)["input_ids"])
        if n_out > args.max_output_tokens:
            drop_out += 1; continue
        kept.append((r, target, inp_text))
    print(f"[v8] dropped {drop_in} over input, {drop_out} over output -> kept {len(kept)}", file=sys.stderr)
    if not kept:
        raise SystemExit("no rows kept")

    out_dir = args.out_dir
    gen_dir = out_dir / "generate"
    gen_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[v8] {len(positives)} positives + {len(negatives)} negatives = {len(df)} rows", file=sys.stderr)
    print(f"[v8] strategy={args.map_strategy}, distinct concept_ids used: {len(concept_counts)}", file=sys.stderr)
    print(f"[v8] concept_id distribution (top-10): {concept_counts.most_common(10)}", file=sys.stderr)

    df.to_parquet(gen_dir / "train_data.parquet", index=False)

    with (gen_dir / "metadata.jsonl").open("w") as f:
        for ac in autinn_concepts:
            cid = int(ac["concept_id"])
            ctxt = ac["concept"]
            f.write(json.dumps({
                "concept_id": cid, "concept": ctxt, "ref": ac.get("ref", "MCPTox"),
                "concept_genres_map": {ctxt: ["code"]},
            }) + "\n")

    merged = []
    for ac in autinn_concepts:
        merged.append({
            "modelId": args.policy_model, "layer": "mcp-attack-v8",
            "index": int(ac["concept_id"]) - 1,
            "description": ac["concept"], "ref": ac.get("ref", "MCPTox"),
            "concept_id": int(ac["concept_id"]),
        })
    (out_dir / "merged_concepts_mcp.json").write_text(json.dumps(merged, indent=2))

    yaml_text = HYPERSTEER_TRAIN_YAML_TEMPLATE.format(
        concept_path=str(out_dir / "merged_concepts_mcp.json"),
        policy_model=args.policy_model, steer_layer=args.steer_layer,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        n_epochs=args.n_epochs, lr=f"{args.lr:.4e}",
    )
    (out_dir / "mcp_hypersteer_config.yaml").write_text(yaml_text)
    print(f"[v8] wrote {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
