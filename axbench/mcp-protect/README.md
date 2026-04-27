# MCP-Protect: HyperSteer (Gemma, Neuronpedia, `mcpattack.jsonl`)

This folder holds **this monorepo’s** pipeline to merge GemmaScope / Neuronpedia concepts with `mcpattack.jsonl`, run AxBench `generate.py` and `train.py` for **HyperSteer**, and **serve** an OpenAI-compatible **policy** for **prime-envs** `vf-eval` (see [../../../../prime-envs/README.md](../../../../prime-envs/README.md)). The goal is to add integration here without forking `axbench/models` more than necessary.

**End-to-end story** (env vars, train → serve → `vf-eval` copy-paste): [README.md at the repository root](../../../../README.md).

---

## What’s in this directory

| File | Role |
|------|------|
| `mcp_hypersteer.py` | Merge → `generate` → `torchrun train` (subprocesses: `axbench/scripts/generate.py` and `train.py`). |
| `serve_mcp_hypersteer.py` | `POST /v1/chat/completions` for a trained run; prints a `http://…/v1` base URL. |

The rest of this file is the **full** guide (prereqs, install, examples, and CLI reference) — a single place for how to run everything here.

---

## Where to run commands

Use the **AxBench project root**: the directory that **contains** the inner `axbench/` package and this `mcp-protect/` folder (e.g. `…/mcp-protect/axbench/` if the clone is `mcp-protect`). From there:

```bash
export OPENAI_API_KEY=...  # for generate if the generated YAML still uses a remote LM
python axbench/mcp-protect/mcp_hypersteer.py --model gemma2b --layer 20 --dump-dir axbench/outputs/mcp_hsteer_2b
```

Paths like `axbench/mcp-protect/...` and `axbench/outputs/...` are **relative to that** AxBench project root.

---

## Prerequisites

- **GPU:** CUDA is expected for training and for `serve_mcp_hypersteer`. Check `nvidia-smi` and use a **PyTorch** build that matches your driver (see [pytorch.org](https://pytorch.org)).
- **HuggingFace:** `huggingface-cli login` (or `HUGGINGFACE_HUB_TOKEN`) for `google/gemma-2-2b-it` / `9b-it`.
- **Data generation:** the generated config defaults to `lm_model: gpt-4o-mini` — set **`OPENAI_API_KEY`**. This is only for **training data**, not the `vf-eval` **policy**.

## Will the commands actually run?

| Step | Machine | Without GPU? |
|------|----------|--------------|
| Merge + YAML in `mcp_hypersteer.py` | CPU | **Yes** with `--skip-generate`. |
| `generate.py` | needs remote LM | **`OPENAI_API_KEY`** (default). |
| `train.py` (HyperSteer) | CUDA expected | **No** for a realistic run. |
| `serve_mcp_hypersteer.py` | CUDA | **No** as configured here. |

`torchrun` sets **process rank** for training. `predict_steer` can call `torch.distributed.get_rank()`; the server pre-initializes a **1-process `gloo` group** so you do not need to patch `axbench/models/hypersteer.py` for that.

## Repo layout and submodule

```bash
git submodule update --init --recursive
```

The directory that **contains** the `axbench` **package** is what must be on the path; `mcp_hypersteer.py` already extends `PYTHONPATH` for its subprocesses.

## Install

Use your normal AxBench / PyTorch environment. For the **HTTP server only** (from the monorepo root, where `axbench/axbench/mcp-protect/` lives):

```bash
pip install -r axbench/axbench/mcp-protect/requirements-serve.txt
```

## Neuronpedia / GemmaScope and MCP

- Download data into `axbench/data/` (e.g. `download-2b.sh` / `download-9b.sh`) or set `--base-concept-json`.
- [../../data/mcpattack.jsonl](../../data/mcpattack.jsonl) (optional) — extra steering lines; merged with Neuronpedia.

**`-c` / `--concepts-16k`** and **`-a` / `--mcp-attack`** cap how many rows enter the merge (after Neuronpedia dedupe and MCP line parsing). **`--max-concepts`** is different — it is passed to **`generate.py`**.

## Train, then serve (paths in this monorepo)

```bash
python axbench/mcp-protect/mcp_hypersteer.py \
  --model gemma2b --layer 20 --dump-dir axbench/outputs/mcp_hsteer_2b

python axbench/mcp-protect/mcp_hypersteer.py \
  --model gemma2b --layer 20 -c 2000 -a 80 --dump-dir axbench/outputs/mcp_hsteer_2b_subset
```

**Serve** (after `train/` exists):

```bash
python axbench/mcp-protect/serve_mcp_hypersteer.py \
  --dump-dir path/to/same/outputs/run \
  --port 8000
```

- Default config: `<dump-dir>/mcp_hypersteer_config.yaml`.
- Steering: `--concept-id` / `--factor` or `HYPERSTEER_*` env vars (see `serve_mcp_hypersteer.py --help`).

The server prints a **base URL** with **`/v1`**. For prime-envs use **`-b http://127.0.0.1:PORT/v1`**, **`-k EMPTY`**, and do not use **`-p openrouter`** for the **steered** policy.

**Skip steps:** `--skip-generate` (write merge + YAML only), `--skip-train` (run `generate` only).

## Optional: AxBench `inference.py` / `evaluate.py`

You can use the same YAML to validate a run before prime-envs (see upstream AxBench docs).

## Memory

- **2B:** most single-GPU setups. **9B:** more VRAM and/or a smaller batch.

---

## Full CLI reference (`mcp_hypersteer.py`)

| Option | Default | Meaning |
|--------|---------|---------|
| `--model` | (required) | `2b` / `9b` or `gemma2b` / `gemma9b` — model preset and default Neuronpedia JSON. |
| `-l` / `--layer` | preset (e.g. 20) | Decoder layer for the intervention. |
| `--base-concept-json` | from preset | Neuronpedia / GemmaScope JSON list. |
| `-c` / `--concepts-16k` | none | Max Neuronpedia rows (after index dedupe). |
| `-a` / `--mcp-attack` | none | Max `mcpattack.jsonl` rows. |
| `--mcp-jsonl` | `axbench/data/mcpattack.jsonl` | JSONL to merge. |
| `--no-mcp` | off | Neuronpedia only. |
| `--dump-dir` | (required) | Run output directory. |
| `--max-concepts` | none | Passed to **`generate.py`**. |
| `--num-of-examples` | 72 | Pairs per concept. |
| `--n-epochs` | 10 | Training epochs. |
| `--batch-size` | 12 | Per-device batch. |
| `--master-data-dir` | `axbench/data` | Seed data for `generate`. |
| `--nproc-per-node` | 1 | `torchrun` processes. |
| `--skip-generate` / `--skip-train` | | As above. |

**Artifacts in `--dump-dir`:** `merged_concepts_mcp.json`, `mcp_hypersteer_config.yaml`, `generate/`, `train/` (e.g. `train/hyperreft/`).
