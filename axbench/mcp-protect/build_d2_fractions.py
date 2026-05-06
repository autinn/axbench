"""C5/D2 — within-archetype data-ablation: build 25%, 50%, 75% fractional
versions of v17's existing 141-row training data (seed=42, deterministic).

Strategy: take the EXISTING v17 train_data.parquet (141 rows; concept_id 0 has
117 positives + concept_id -1 has 24 negatives) and produce three subsampled
parquets. We stratify the subsample so the positive/negative ratio is preserved.

Sizes (after rounding stratified per-class to nearest int, total may be ±1):
  25% -> ~35 rows  (29 pos + 6 neg)
  50% -> ~70 rows  (58 pos + 12 neg)
  75% -> ~105 rows (88 pos + 18 neg)
  100% (existing v17, no new file)

Outputs:
  axbench/axbench/data/v17_25pct.parquet
  axbench/axbench/data/v17_50pct.parquet
  axbench/axbench/data/v17_75pct.parquet
"""
from pathlib import Path

import pandas as pd

REPO_AX = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_AX / "data"

# v17's training parquet lives in its output dir (not in data/); prefer that, else fall back.
SRC_CANDIDATES = [
    REPO_AX / "outputs" / "mcp_hsteer_qwen3_8b_v17_terse" / "generate" / "train_data.parquet",
    DATA_DIR / "v17_train_data.parquet",
]
SEED = 42
FRACS = [0.25, 0.50, 0.75]


def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    parts = []
    for cid, sub in df.groupby("concept_id", sort=True):
        n = max(1, round(len(sub) * frac))
        n = min(n, len(sub))
        parts.append(sub.sample(n=n, random_state=seed))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    src = next((p for p in SRC_CANDIDATES if p.exists()), None)
    assert src is not None, f"no v17 parquet found; checked {SRC_CANDIDATES}"
    df = pd.read_parquet(src)
    print(f"Source: {src}")
    print(f"Total: {len(df)} rows")
    print("Per-cid:")
    print(df["concept_id"].value_counts().sort_index())
    print()

    for frac in FRACS:
        sub = stratified_sample(df, frac, SEED)
        pct_tag = f"{int(frac * 100)}pct"
        out = DATA_DIR / f"v17_{pct_tag}.parquet"
        sub.to_parquet(out, index=False)
        print(f"v17_{pct_tag}: {len(sub)} rows -> {out.name}")
        print(sub["concept_id"].value_counts().sort_index().to_dict())
        print()


if __name__ == "__main__":
    main()
