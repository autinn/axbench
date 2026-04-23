# HyperSteer + Neuronpedia 16k + `mcpattack.jsonl`

Companion guide for **`axbench/scripts/mcp_hypersteer.py`**.

## Rationale

- **HyperSteer** (see `axbench/models/hypersteer.py`) trains a **hypernetwork** that **cross-attends** into a chosen **layer** of a **frozen** base Gemma model, produces a steering direction, and applies it as a **residual add** at that layer. The main learned artifact is the **hypernet** (e.g. under `train/hyperreft/`), not a full finetune of the base LLM.
- This script merges a **GemmaScope / Neuronpedia** concept JSON (≈16k) with optional lines from **`mcpattack.jsonl`**, then runs stock AxBench **`generate.py`** and **`train.py`** so you keep the same `concept_path` contract as the rest of the repo.

## What the script does

1. **Parse** the base Neuronpedia JSON and normalize rows to `{ modelId, layer, index, description }` (dedupe by `index`, first wins) — `parse_neuronpedia_export`.
2. **Optionally** read **`mcpattack.jsonl`**: one JSON object per line; each line’s **steering string** is taken from **`description`** or **`concept`** (in that order) and written as **`description`** on a synthetic row with `layer: "mcp-attack"`.
3. **Append** MCP rows after the Neuronpedia list and **reindex** MCP so indices are unique (`merge_neuronpedia_with_mcp` starts at `max(Neuronpedia int index) + 1` — see *Caveats*).
4. Write **`merged_concepts_mcp.json`** and **`mcp_hypersteer_config.yaml`** into **`--dump-dir`**.
5. Run **`axbench/scripts/generate.py --config <yaml> --dump_dir <dir>`** (produces `generate/` with `metadata.jsonl`, `train_data.parquet`, etc.).
6. Run **`torchrun ... axbench/scripts/train.py`** with the same config (trains **HyperSteer** once on the full table; outputs under `.../train/`, e.g. **`hyperreft/`**).

**Gemma 2B vs 9B:** run the script **twice** with `--model 2b` and `--model 9b` and **separate** `--dump-dir` values. There is no single flag for both.

**Not run automatically:** `inference.py` and `evaluate.py`. The written YAML still includes `inference` and `evaluate` sections (same pattern as `demo/sweep/hypersteer_simple.yaml`); use them manually after training if you need steering evals.

## Prerequisites

- **Repo root** as the working context; subprocesses are started with `cwd=REPO_ROOT`. `PYTHONPATH` is extended with `axbench/scripts` for imports.
- **Python** with PyTorch (and **GPU** for `generate_training` and `train`), `transformers`, and project dependencies.
- **`OPENAI_API_KEY`** (or whatever LM the generated YAML’s `generate.lm_model` expects) for dataset generation in `generate.py`.
- **Base concept JSON** on disk, e.g. after:
  - `axbench/data/download-2b.sh` → `axbench/data/gemma-2-2b_20-gemmascope-res-16k.json`
  - `axbench/data/download-9b.sh` → `axbench/data/gemma-2-9b_31-gemmascope-res-16k.json`  
  Or pass **`--base-concept-json`**.

## MCP JSONL: which fields are read?

`mcpattack.jsonl` is **not** a Neuronpedia file. Per line, `_mcp_text_from_row` uses the first non-empty string in order:

1. **`description`**
2. **`concept`**

The checked-in **`mcpattack.jsonl`** uses **`concept`**; that is supported. Optional fields `ref`, `attack_stage`, `attack_type`, `concept_id` are **copied onto the merged record** for traceability; **`generate.load_concepts` only uses** `modelId`, `layer`, `index`, and `description` when building training data.

## Command-line interface

| Option | Default | Meaning |
|--------|---------|---------|
| `--model` | (required) | `2b` or `9b` — selects HF model, Neuronpedia slug, and default base JSON path. |
| `--base-concept-json` | 2B/9B default under `axbench/data/…16k.json` | Neuronpedia / GemmaScope **JSON array** (same as `generate.load_concepts` `.json` input). |
| `--mcp-jsonl` | `axbench/data/mcpattack.jsonl` | JSONL to merge. |
| `--no-mcp` | off | Do not load MCP; Neuronpedia only. |
| `--dump-dir` | (required) | Output run directory (see Artifacts). |
| `--max-concepts` | none | Passed through to `generate` (subsample + shuffle in `generate.py`). |
| `--num-of-examples` | `72` | Training pairs per concept in `generate`. |
| `--n-epochs` | `10` | HyperSteer training epochs. |
| `--batch-size` | `12` | Per-device batch size. |
| `--master-data-dir` | `axbench/data` | Shared seed data for `generate`. |
| `--nproc-per-node` | `1` | `torchrun` GPU count. |
| `--skip-generate` | off | Only write merge + YAML; do not run `generate` / `train`. |
| `--skip-train` | off | Run `generate` but not `train`. |

## Artifacts (under `--dump-dir`)

- **`merged_concepts_mcp.json`** — full merged list passed as `generate.concept_path` in the YAML.
- **`mcp_hypersteer_config.yaml`** — `generate` / `train` / `inference` / `evaluate` blocks for AxBench scripts.
- **`generate/`** — `metadata.jsonl`, `train_data.parquet`, state pickles (from `generate.py`).
- **`train/`** — HyperSteer outputs, including **`hyperreft/`** (hypernet + tokenizer) when training completes.

## Example commands (from repository root)

**Minimal (2B, default paths, 16k + MCP):**

```bash
export OPENAI_API_KEY=...

python axbench/scripts/mcp_hypersteer.py \
  --model 2b \
  --dump-dir axbench/outputs/mcp_hypersteer_2b
```

**9B, explicit MCP path:**

```bash
python axbench/scripts/mcp_hypersteer.py \
  --model 9b \
  --mcp-jsonl axbench/data/mcpattack.jsonl \
  --dump-dir axbench/outputs/mcp_hypersteer_9b
```

**Smaller / faster debug run:**

```bash
python axbench/scripts/mcp_hypersteer.py \
  --model 2b \
  --max-concepts 200 \
  --n-epochs 1 \
  --num-of-examples 8 \
  --dump-dir axbench/outputs/mcp_hsteer_smoke
```

**Write merge + config only (no API / no train):**

```bash
python axbench/scripts/mcp_hypersteer.py \
  --model 2b \
  --skip-generate \
  --dump-dir axbench/outputs/inspect_merged
```

**Neuronpedia only (no MCP):**

```bash
python axbench/scripts/mcp_hypersteer.py \
  --model 2b \
  --no-mcp \
  --dump-dir axbench/outputs/hypersteer_np_only
```

## Optional follow-up

- **Steering eval:** e.g. `torchrun --nproc_per_node=1 axbench/scripts/inference.py --config <dump-dir>/mcp_hypersteer_config.yaml --mode steering --dump_dir <dump-dir>` (adjust to your cluster).
- **Evaluation:** `python axbench/scripts/evaluate.py` with the same config and mode as in the project README / demo.

## Caveats

- **Scale:** Full 16k + 133 MCP lines is **expensive** (API + wall time + GPU). Use `--max-concepts` for development.
- **`max_index` in merge:** reindexing MCP after Neuronpedia uses **integer** `index` maxima. If your Neuronpedia file uses only **string** indices, verify in code or data that new MCP indices do not collide with existing rows (rare, but possible if the export mixes types).
- **Missing `mcpattack.jsonl`:** if `--mcp-jsonl` is not found and you did not pass `--no-mcp`, the script **prints a note** and runs **Neuronpedia only**.

## Historical note

Older docs referred to `mcp_hypersteer_train.py` and `merged_mcp_concepts.json`. The current entry point is **`mcp_hypersteer.py`**; the merged file name is **`merged_concepts_mcp.json`**.
