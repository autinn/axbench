# `mcp-protect/` (AxBench)

This directory adds a **self-contained pipeline** for this monorepo:

1. **`mcp_hypersteer.py`** — merge Neuronpedia / GemmaScope JSON with optional `mcpattack.jsonl`, write **`mcp_hypersteer_config.yaml`**, then call stock AxBench **`generate.py`** (remote LM for synthetic training text) and **`train.py`** (HyperSteer on **local** Gemma).
2. **`serve_mcp_hypersteer.py`** — after training, load that run and expose **`POST /v1/chat/completions`** so **prime-envs** `vf-eval` can use **`-b http://…/v1`**.

**Monorepo context:** [README at repo root](../../../../README.md) (axbench + prime-envs). **Shell:** all `python` paths below are from the **AxBench project root** (the directory that contains the inner `axbench/` package, e.g. `…/mcp-protect/axbench/`).

---

## 1. Environment

- **Python / AxBench:** use an env with **PyTorch** (CUDA for train/serve), **transformers**, and AxBench dependencies (see upstream `axbench` / your `pyproject` or conda env).
- **HuggingFace:** `huggingface-cli login` to pull **Gemma** (`google/gemma-2-2b-it` or `9b-it`).
- **Remote LM for `generate` (default `gpt-4o-mini` in the YAML):** set **`OPENAI_API_KEY`**. This only builds **training data**, not the `vf-eval` policy.
- **Optional HTTP server deps** (for `serve_mcp_hypersteer.py` only), from the **mcp-protect** repo root:

  `pip install -r axbench/axbench/mcp-protect/requirements-serve.txt`

- **Submodule (if `axbench` is a submodule):** `git submodule update --init --recursive`

---

## 2. Download data

- **Neuronpedia / GemmaScope JSON** (per model size), e.g. from `axbench/data/`:
  - `bash axbench/data/download-2b.sh` or `download-9b.sh` (or your own file; pass `--base-concept-json` if not using the default name).
- **Optional:** [../data/mcpattack.jsonl](../data/mcpattack.jsonl) — extra steering lines merged into the same concept list (omit with `--no-mcp` or if the file is missing).

---

## 3. Test YAML + merge only (no API, no GPU training)

This checks that **merge logic** and **YAML generation** work end-to-end for your paths:

```bash
python axbench/mcp-protect/mcp_hypersteer.py \
  --model gemma2b --layer 20 \
  --skip-generate \
  --dump-dir axbench/outputs/mcp_smoke_config
```

**You should get:**

- `axbench/outputs/mcp_smoke_config/merged_concepts_mcp.json`
- `axbench/outputs/mcp_smoke_config/mcp_hypersteer_config.yaml` (sections: `generate`, `train`, `inference`, `evaluate`)

No `OPENAI_API_KEY` and no `train/` until you run without `--skip-generate` (and then train).

---

## 4. Full run: generate → train

From the same AxBench project root, with **`OPENAI_API_KEY`** set if the written `generate.lm_model` is an OpenAI model:

```bash
export OPENAI_API_KEY=...
python axbench/mcp-protect/mcp_hypersteer.py \
  --model gemma2b --layer 20 \
  --dump-dir axbench/outputs/mcp_hsteer_2b
```

**Faster / smaller merge** (optional): `-c 2000 -a 80` (cap Neuronpedia vs MCP rows; see `--help`).

**Artifacts (under `--dump-dir`):** `generate/` (parquet, `metadata.jsonl`), `train/` (e.g. `train/hyperreft/`).  
**Iterating:** `--skip-train` runs `generate` only; `--skip-generate` is the smoke step above.

---

## 5. Serve (OpenAI-style local policy)

After `train/` and `generate/` exist, start the **policy** server (GPU expected):

```bash
python axbench/mcp-protect/serve_mcp_hypersteer.py \
  --dump-dir axbench/outputs/mcp_hsteer_2b \
  --port 8000
```

**Behavior (concise):** loads Gemma + HyperSteer from the run, fixes **one** steering **concept** per process (from `--concept-id` or `HYPERSTEER_CONCEPT_ID`, default: first id in `metadata.jsonl`), runs **one-rank `gloo`** so `predict_steer` works without `torchrun`, then handles **`POST /v1/chat/completions`**. It **prints** a base URL ending in **`/v1`** — use that in **prime-envs** with **`-k EMPTY`**. See `--help` for `HYPERSTEER_FACTOR`, etc.

---

## 6. CLI quick reference (`mcp_hypersteer.py`)

| Flag | Role |
|------|------|
| `--model` | `2b` / `9b` or `gemma2b` / `gemma9b` |
| `-l` / `--layer` | Decoder layer (default in preset, often 20) |
| `--base-concept-json` | Neuronpedia JSON path (else default under `axbench/data/`) |
| `-c` / `-a` | Cap Neuronpedia vs `mcpattack` rows in the **merge** |
| `--max-concepts` | Passed to **`generate.py`** (subsample after merge) |
| `--dump-dir` | Run directory (required) |
| `--skip-generate` | Only merge + YAML (smoke test) |
| `--skip-train` | Run `generate`, not `train` |

**GPU:** needed for a real **train** and for **serve**; **merge + YAML** with `--skip-generate` is CPU-friendly.
