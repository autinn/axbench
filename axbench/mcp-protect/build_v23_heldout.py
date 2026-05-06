"""v23 generalization-probe dataset builder.

Per HANDOFF C4 (v23): retrain v22 on 8 of 12 archetypes; hold out the 4 hardest.

KEPT (8 trained cids, original v22 indices):
  0  function-hijack
  1  prefix-injection
  2  suffix-injection
  3  role-confusion
  4  instruction-override
  5  output-manipulation
  6  fake-error
  10 scope-escalation

HELD OUT (4 cids — never seen in training, eval-only):
  7  fake-success
  8  schema-mutation
  9  double-call
  11 exfiltration

Important: we PRESERVE the original v22 concept_id integers so the same hypernet
inputs can be reused at eval time across all 12 cids. The trained set has 8
unique concept_ids drawn from {0,1,2,3,4,5,6,10}; the held-out set uses
{7,8,9,11}. Only the 8 trained concepts go into v23_concepts.jsonl /
v23_metadata.jsonl. v23_FULL_concepts.jsonl is also written for eval convenience
and contains all 12 concepts (original v22 set).

Inputs:
  axbench/axbench/data/v22_train_data.parquet
  axbench/axbench/data/v22_metadata.jsonl
  axbench/axbench/data/v22_concepts.jsonl

Outputs:
  axbench/axbench/data/v23_train_data.parquet (8 cids only)
  axbench/axbench/data/v23_metadata.jsonl     (8 cids)
  axbench/axbench/data/v23_concepts.jsonl     (8 cids — for training)
  axbench/axbench/data/v23_FULL_concepts.jsonl (12 cids — for eval-only probe)
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
IN_PARQUET = DATA_DIR / "v22_train_data.parquet"
IN_META = DATA_DIR / "v22_metadata.jsonl"
IN_CONCEPTS = DATA_DIR / "v22_concepts.jsonl"

OUT_PARQUET = DATA_DIR / "v23_train_data.parquet"
OUT_META = DATA_DIR / "v23_metadata.jsonl"
OUT_CONCEPTS = DATA_DIR / "v23_concepts.jsonl"
OUT_FULL_CONCEPTS = DATA_DIR / "v23_FULL_concepts.jsonl"

# Original v22 indices to KEEP for training (8 archetypes)
KEEP_CIDS = {0, 1, 2, 3, 4, 5, 6, 10}
# Held-out (eval-only) cids (4 archetypes)
HOLDOUT_CIDS = {7, 8, 9, 11}

NAME_BY_CID = {
    0: "function-hijack",
    1: "prefix-injection",
    2: "suffix-injection",
    3: "role-confusion",
    4: "instruction-override",
    5: "output-manipulation",
    6: "fake-error",
    7: "fake-success",
    8: "schema-mutation",
    9: "double-call",
    10: "scope-escalation",
    11: "exfiltration",
}


def main():
    assert IN_PARQUET.exists(), f"missing {IN_PARQUET}"
    assert IN_META.exists(), f"missing {IN_META}"
    assert IN_CONCEPTS.exists(), f"missing {IN_CONCEPTS}"

    df = pd.read_parquet(IN_PARQUET)
    n_in = len(df)

    sub = df[df["concept_id"].isin(KEEP_CIDS)].reset_index(drop=True)
    n_out = len(sub)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    sub.to_parquet(OUT_PARQUET, index=False)

    # metadata.jsonl — keep only the 8 trained concepts (preserve original concept_id)
    full_meta = [json.loads(line) for line in IN_META.read_text().splitlines() if line.strip()]
    kept_meta = [m for m in full_meta if m["concept_id"] in KEEP_CIDS]
    with OUT_META.open("w") as f:
        for m in kept_meta:
            f.write(json.dumps(m) + "\n")

    # concepts.jsonl is a SINGLE JSON ARRAY in v22 format; mirror that.
    full_concepts = json.loads(IN_CONCEPTS.read_text())
    train_concepts = [c for c in full_concepts if c["concept_id"] in KEEP_CIDS]
    with OUT_CONCEPTS.open("w") as f:
        f.write(json.dumps(train_concepts))

    # Full-12 concepts file for eval-only probing across held-out cids.
    with OUT_FULL_CONCEPTS.open("w") as f:
        f.write(json.dumps(full_concepts))

    # ---- report ----
    print(f"Read v22 train_data.parquet ({n_in} rows)")
    print(f"Wrote v23_train_data.parquet ({n_out} rows; 8 trained cids)")
    print()
    print("KEPT (trained) per-cid row counts:")
    print(sub["concept_id"].value_counts().sort_index())
    print()
    print(f"HELD OUT (eval-only, never trained on): {sorted(HOLDOUT_CIDS)}")
    for cid in sorted(HOLDOUT_CIDS):
        held_n = int((df["concept_id"] == cid).sum())
        print(f"  cid={cid:2d} {NAME_BY_CID[cid]:24s} v22 had {held_n} rows -> dropped from training")
    print()
    print(f"Wrote {OUT_META.name} ({len(kept_meta)} entries)")
    print(f"Wrote {OUT_CONCEPTS.name} (8 trained concepts; for hypernet training)")
    print(f"Wrote {OUT_FULL_CONCEPTS.name} (12 concepts; for eval across all archetypes)")


if __name__ == "__main__":
    main()
