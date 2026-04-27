#!/usr/bin/env python3
"""
MCP + HyperSteer: one-command pipeline from Neuronpedia / MCP concepts to a trained HyperSteer run.

Run from the **AxBench project root** (the directory that contains the inner ``axbench/`` package).
``main()`` sets ``PYTHONPATH`` (package root + ``axbench/scripts``) and spawns the stock AxBench
scripts with ``cwd`` = that root so ``generate`` and ``train`` run unchanged.

**End-to-end steps (how each is achieved)**

1. **Resolve model preset** (``--model`` → ``MODEL_PRESETS`` / ``resolve_model_key``) — maps
   ``2b`` / ``9b`` (aliases ``gemma2b`` / ``gemma9b``) to the HuggingFace Gemma id, Neuronpedia slug,
   and default GemmaScope JSON filename; ``--layer`` / ``-l`` overrides the preset training layer.

2. **Merge concept lists** — ``build_merged_concept_path`` loads the Neuronpedia export
   (``parse_neuronpedia_export``, dedupe by index), optionally loads ``mcpattack.jsonl``
   (``parse_mcpattack_jsonl``), applies optional caps ``-c`` / ``--concepts-16k`` and
   ``-a`` / ``--mcp-attack``, then ``merge_neuronpedia_with_mcp`` and writes
   ``<dump-dir>/merged_concepts_mcp.json`` in the same record shape as
   ``generate.load_concepts`` for JSON inputs (``modelId``, ``layer``, ``index``, ``description``).

3. **Write run config** — ``_build_run_config`` + ``_write_yaml`` emit
   ``<dump-dir>/mcp_hypersteer_config.yaml`` with ``generate`` (default ``lm_model: gpt-4o-mini`` for
   remote **dataset** synthesis), ``train`` (HyperSteer on local Gemma), and ``inference`` /
   ``evaluate`` blocks for AxBench.

4. **Generate training data (optional)** — unless ``--skip-generate``, ``_run`` executes
   ``python axbench/scripts/generate.py --config … --dump_dir …`` → ``<dump-dir>/generate/``
   (e.g. parquet, ``metadata.jsonl``). Requires API access for the configured ``lm_model``
   (typically ``OPENAI_API_KEY`` for the default OpenAI model).

5. **Train HyperSteer (optional)** — unless ``--skip-train``, ``_run`` launches
   ``torchrun ... axbench/scripts/train.py`` with the same YAML → ``<dump-dir>/train/``
   (e.g. checkpoints under ``train/hyperreft/``).

**Example**::

    export OPENAI_API_KEY=...  # for generate, if using default gpt-4o-mini in the YAML
    python axbench/mcp-protect/mcp_hypersteer.py --model gemma2b --layer 20 --dump-dir axbench/outputs/mcp_hsteer_2b
    # Optional merge caps: -c 500 -a 50
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# This file: axbench/axbench/mcp-protect/mcp_hypersteer.py -> AxBench project root is parents[2]
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "axbench" / "scripts"

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "2b": {
        "hf_model": "google/gemma-2-2b-it",
        "neuronpedia_slug": "gemma-2-2b",
        "train_layer": 20,
    },
    "9b": {
        "hf_model": "google/gemma-2-9b-it",
        "neuronpedia_slug": "gemma-2-9b-it",
        "train_layer": 20,
    },
}

# Default Neuronpedia / GemmaScope filenames (under axbench/data) per model preset.
DEFAULT_CONCEPT_JSON_NAME = {
    "2b": "gemma-2-2b_20-gemmascope-res-16k.json",
    "9b": "gemma-2-9b_31-gemmascope-res-131k.json",
}

# Aliases for --model (equivalent to 2b / 9b presets)
_MODEL_KEY_ALIASES: dict[str, str] = {
    "2b": "2b",
    "9b": "9b",
    "gemma2b": "2b",
    "gemma9b": "9b",
}


def resolve_model_key(raw: str) -> str:
    s = raw.strip().lower().replace("-", "").replace("_", "")
    if s in _MODEL_KEY_ALIASES:
        return _MODEL_KEY_ALIASES[s]
    if raw in MODEL_PRESETS:
        return raw
    raise SystemExit(
        f"Unknown --model {raw!r}; use 2b, 9b, gemma2b, or gemma9b "
        f"(see MODEL_PRESETS in this script)."
    )


def resolve_base_concept_json_path(
    base_arg: Path | None,
    model_key: str,
    repo_root: Path,
) -> Path:
    """
    Return an existing .json path for Neuronpedia / GemmaScope concepts.

    - If ``base_arg`` is None, use ``axbench/data/<DEFAULT_CONCEPT_JSON_NAME[model_key]>``.
    - If ``base_arg`` is a **directory** (e.g. ``axbench/data``), use that directory +
      ``<DEFAULT_CONCEPT_JSON_NAME[model_key]>`` so a folder path resolves to the default file.
    - If ``base_arg`` is a file path, use it as-is (after resolve relative to repo_root).
    """
    default_name = DEFAULT_CONCEPT_JSON_NAME[model_key]
    if base_arg is None:
        p = repo_root / "axbench" / "data" / default_name
    else:
        p = (repo_root / base_arg).resolve() if not base_arg.is_absolute() else base_arg.resolve()
        if p.is_dir():
            p = p / default_name

    if p.is_file():
        return p

    if base_arg is not None:
        root = (repo_root / base_arg).resolve() if not base_arg.is_absolute() else base_arg.resolve()
        if root.is_dir():
            raise FileNotFoundError(
                f"Neuronpedia JSON: under directory {root} expected file {default_name!r} but it is missing. "
                f"Download data (e.g. axbench/data/download-2b.sh / download-9b.sh) or pass the full path to a .json file, "
                f"e.g. --base-concept-json axbench/data/gemma-2-9b-it_31-gemmascope-res-131k.json"
            )
    raise FileNotFoundError(
        f"Missing Neuronpedia JSON: {p} — run axbench/data/download-*.sh or set --base-concept-json to a valid .json file."
    )


# ---------------------------------------------------------------------------
# Parsers: mirror generate.load_concepts() so merged JSON is valid concept_path
# (see load_concepts in axbench/scripts/generate.py, ".json" branch)
# ---------------------------------------------------------------------------


def parse_neuronpedia_export(path: Path) -> list[dict[str, Any]]:
    """
    Parse a Neuronpedia / GemmaScope JSON array exactly like `load_concepts` for `.json` inputs:
    dedupe by `index` (first wins), and keep modelId, layer, index, description.
    """
    with path.open("r", encoding="utf-8") as f:
        json_concepts: list[dict[str, Any]] = json.load(f)
    if not isinstance(json_concepts, list):
        raise ValueError(f"Expected a JSON list in {path}")
    out: list[dict[str, Any]] = []
    seen_index: set[int] = set()
    for row in json_concepts:
        subspace_id = row["index"]
        if subspace_id in seen_index:
            continue
        seen_index.add(subspace_id)
        out.append(
            {
                "modelId": row["modelId"],
                "layer": row["layer"],
                "index": subspace_id,
                "description": row["description"].strip(),
            }
        )
    return out


def _mcp_text_from_row(row: dict[str, Any]) -> str:
    """
    Steering text, comparable to Neuronpedia `description`.
    Order matches typical AxBench + MCP files: description, concept, steering_prompt.
    """
    for key in ("description", "concept"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def parse_mcpattack_jsonl(
    path: Path,
    model_id: str,
    layer: str = "mcp-attack",
    start_index: int = 0,
) -> list[dict[str, Any]]:
    """
    One JSON object per line. Maps each row into the same record shape as Neuronpedia export:
    { modelId, layer, index, description }.
    """
    out: list[dict[str, Any]] = []
    next_index = start_index
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            row = json.loads(raw)
            text = _mcp_text_from_row(row)
            if not text:
                continue
            rec: dict[str, Any] = {
                "modelId": model_id,
                "layer": layer,
                "index": next_index,
                "description": text,
            }
            for extra in (
                "ref",
                "attack_stage",
                "attack_type",
                "concept_id",
            ):
                if extra in row:
                    rec[extra] = row[extra]
            out.append(rec)
            next_index += 1
    return out


def max_index(records: list[dict[str, Any]]) -> int:
    m = -1
    for r in records:
        i = r.get("index")
        if isinstance(i, int):
            m = max(m, i)
    return m


def merge_neuronpedia_with_mcp(
    neuronpedia_records: list[dict[str, Any]],
    mcp_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Append MCP rows after Neuronpedia rows, renumbering MCP indices if their indices collide.
    (Neuronpedia indices are preserved; MCP indices are reassigned to follow max(NP index)+1.)
    """
    if not mcp_records:
        return list(neuronpedia_records)
    start = max_index(neuronpedia_records) + 1
    fixed_mcp: list[dict[str, Any]] = []
    i = start
    for r in mcp_records:
        nr = {**r, "index": i}
        fixed_mcp.append(nr)
        i += 1
    return list(neuronpedia_records) + fixed_mcp


def records_to_concept_tuples(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Reproduce the two parallel lists that `load_concepts` would return for this merged JSON
    (sae_concepts / Neuronpedia ref URLs) — for debugging and parity checks.
    """
    sae: list[str] = []
    urls: list[str] = []
    for c in records:
        model = c["modelId"]
        sae_model = c["layer"]
        subspace_id = c["index"]
        sae.append(c["description"].strip())
        urls.append(
            f"https://www.neuronpedia.org/{model}/{sae_model}/{subspace_id}"
        )
    return sae, urls


# ---------------------------------------------------------------------------
# Config + driver (HyperSteer train)
# ---------------------------------------------------------------------------


def _write_merged_concepts(merged: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


def _build_run_config(
    *,
    rel_merged_path: str,
    dump_dir: str,
    hf_model: str,
    train_layer: int,
    max_concepts: int | None,
    num_of_examples: int,
    n_epochs: int,
    batch_size: int,
    master_data_dir: str,
) -> dict[str, Any]:
    gen: dict[str, Any] = {
        "lm_model": "gpt-4o-mini",
        "input_length": 32,
        "output_length": 32,
        "num_of_examples": num_of_examples,
        "concept_path": rel_merged_path,
        "dataset_category": "instruction",
        "master_data_dir": master_data_dir,
        "seed": 42,
        "keep_orig_axbench_format": True,
    }
    if max_concepts is not None:
        gen["max_concepts"] = max_concepts
    train: dict[str, Any] = {
        "model_name": hf_model,
        "layer": train_layer,
        "component": "res",
        "seed": 42,
        "use_bf16": True,
        "output_length": 128,
        "models": {
            "HyperSteer": {
                "batch_size": batch_size,
                "gradient_accumulation_steps": 1,
                "n_epochs": n_epochs,
                "lr": 0.00008,
                "weight_decay": 0.0,
                "low_rank_dimension": 1,
                "intervention_positions": "all",
                "intervention_type": "addition",
                "binarize_dataset": False,
                "train_on_negative": False,
                "exclude_bos": True,
                "hypernet_name_or_path": hf_model,
                "num_hidden_layers": 4,
                "hypernet_initialize_from_pretrained": True,
            }
        },
    }
    inference: dict[str, Any] = {
        "use_bf16": True,
        "models": ["HyperSteer"],
        "model_name": hf_model,
        "output_length": 128,
        "latent_num_of_examples": 36,
        "latent_batch_size": 16,
        "steering_intervention_type": "addition",
        "steering_model_name": hf_model,
        "steering_datasets": ["AlpacaEval"],
        "steering_batch_size": 10,
        "steering_output_length": 128,
        "steering_layers": [10],
        "steering_num_of_examples": 10,
        "steering_factors": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
        "master_data_dir": master_data_dir,
        "seed": 42,
        "lm_model": "gpt-4o-mini",
        "temperature": 1.0,
    }
    evaluate: dict[str, Any] = {
        "models": ["HyperSteer"],
        "latent_evaluators": ["AUCROCEvaluator", "HardNegativeEvaluator"],
        "steering_evaluators": ["LMJudgeEvaluator"],
        "winrate_split_ratio": 0.5,
        "num_of_workers": 32,
        "lm_model": "gpt-4o-mini",
        "run_winrate": False,
        "winrate_baseline": "PromptSteering",
        "master_data_dir": master_data_dir,
    }
    return {
        "generate": gen,
        "train": train,
        "inference": inference,
        "evaluate": evaluate,
    }


def _write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_merged_concept_path(
    base_neuronpedia_json: Path,
    mcp_jsonl: Path | None,
    neuronpedia_slug: str,
    out_json: Path,
    *,
    concepts_16k_max: int | None = None,
    mcp_attack_max: int | None = None,
) -> list[dict[str, Any]]:
    """
    End-to-end merge: Neuronpedia ~16k export + optional MCP JSONL -> one list suitable for
    generate.py concept_path, written to out_json. MCP rows are reindexed after the max
    Neuronpedia index.

    ``concepts_16k_max`` — keep at most this many Neuronpedia records (order preserved, after dedupe).
    ``mcp_attack_max`` — keep at most this many MCP JSONL rows (in file order, after empty-line skip).
    """
    print(f"[mcp_hypersteer] Loading Neuronpedia: {base_neuronpedia_json}", flush=True)
    base = parse_neuronpedia_export(base_neuronpedia_json)
    print(f"[mcp_hypersteer]   -> {len(base)} concepts (deduped by index).", flush=True)
    if concepts_16k_max is not None and concepts_16k_max >= 0:
        base = base[:concepts_16k_max]
        print(
            f"[mcp_hypersteer]   -> after --concepts-16k {concepts_16k_max}: {len(base)} rows.",
            flush=True,
        )
    mcp: list[dict[str, Any]] = []
    if mcp_jsonl is not None and mcp_jsonl.is_file():
        print(f"[mcp_hypersteer] Loading MCP JSONL: {mcp_jsonl}", flush=True)
        mcp = parse_mcpattack_jsonl(
            mcp_jsonl,
            model_id=neuronpedia_slug,
            layer="mcp-attack",
            start_index=0,
        )
        if mcp_attack_max is not None and mcp_attack_max >= 0:
            mcp = mcp[:mcp_attack_max]
        print(
            f"[mcp_hypersteer]   -> {len(mcp)} MCP rows"
            + (
                f" (capped from file by --mcp-attack {mcp_attack_max})"
                if mcp_attack_max is not None
                else "."
            ),
            flush=True,
        )
    elif mcp_jsonl is not None:
        print(
            f"[mcp_hypersteer] MCP path not a file, skipping: {mcp_jsonl}", flush=True
        )
    else:
        print("[mcp_hypersteer] No MCP file (MCP list empty).", flush=True)
    merged = merge_neuronpedia_with_mcp(base, mcp)
    _write_merged_concepts(merged, out_json)
    print(
        f"[mcp_hypersteer] Wrote merged list ({len(merged)} records) -> {out_json}",
        flush=True,
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Neuronpedia concept JSON and mcpattack.jsonl, then run HyperSteer."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Preset: 2b, 9b, or aliases gemma2b, gemma9b (see MODEL_PRESETS).",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=None,
        help="Decoder layer for HyperSteer (default: 20 in MODEL_PRESETS).",
    )
    parser.add_argument(
        "--base-concept-json",
        type=Path,
        help="Neuronpedia / GemmaScope JSON (e.g. concept16k download).",
    )
    parser.add_argument(
        "--mcp-jsonl",
        type=Path,
        default=REPO_ROOT / "axbench" / "data" / "mcpattack.jsonl",
        help="Optional extra steering lines (JSONL). If the file is missing, only Neuronpedia data is used.",
    )
    parser.add_argument(
        "--no-mcp",
        action="store_true",
        help="Do not load mcpattack.jsonl even if it exists (Neuronpedia only).",
    )
    parser.add_argument(
        "-c",
        "--concepts-16k",
        type=int,
        default=None,
        dest="concepts_16k_max",
        metavar="N",
        help="After Neuronpedia parse + dedupe, keep at most the first N rows (order as in the JSON). "
        "Omit for all rows. Use with --mcp-attack to balance Neuronpedia vs mcpattack.jsonl for merge only; "
        "see also --max-concepts (applies in generate).",
    )
    parser.add_argument(
        "-a",
        "--mcp-attack",
        type=int,
        default=None,
        dest="mcp_attack_max",
        metavar="N",
        help="After reading mcpattack.jsonl, keep at most the first N non-empty lines (in file order). "
        "Omit for all valid rows.",
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="Run directory: merged_concepts_mcp.json, mcp_hypersteer_config.yaml, generate/, train/.",
    )
    parser.add_argument("--max-concepts", type=int, default=None)
    parser.add_argument("--num-of-examples", type=int, default=72)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--master-data-dir", type=str, default="axbench/data")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    model_key = resolve_model_key(args.model)
    preset = dict(MODEL_PRESETS[model_key])
    if args.layer is not None:
        preset["train_layer"] = int(args.layer)
    base_path = resolve_base_concept_json_path(
        args.base_concept_json, model_key, REPO_ROOT
    )
    mcp_path: Path | None = None
    if not args.no_mcp:
        mp = (
            (REPO_ROOT / args.mcp_jsonl).resolve()
            if not args.mcp_jsonl.is_absolute()
            else args.mcp_jsonl
        )
        if mp.is_file():
            mcp_path = mp
        else:
            print(
                f"Note: mcp jsonl not found at {mp}; using Neuronpedia concepts only.",
                flush=True,
            )
    dump_dir = (
        (REPO_ROOT / args.dump_dir).resolve()
        if not args.dump_dir.is_absolute()
        else args.dump_dir
    )

    if not base_path.is_file():
        raise FileNotFoundError(
            f"Missing Neuronpedia JSON: {base_path} — run axbench/data/download-*.sh or set --base-concept-json."
        )

    print(
        f"[mcp_hypersteer] repo root={REPO_ROOT}\n"
        f"  base JSON: {base_path}\n"
        f"  MCP: {mcp_path if mcp_path else None}\n"
        f"  dump_dir: {dump_dir}\n"
        f"  skip_generate={args.skip_generate} skip_train={args.skip_train}",
        flush=True,
    )

    dump_dir.mkdir(parents=True, exist_ok=True)
    merged_path = dump_dir / "merged_concepts_mcp.json"
    build_merged_concept_path(
        base_path,
        mcp_path,
        preset["neuronpedia_slug"],
        merged_path,
        concepts_16k_max=args.concepts_16k_max,
        mcp_attack_max=args.mcp_attack_max,
    )

    rel_merged = str(merged_path.relative_to(REPO_ROOT))
    rel_dump = str(dump_dir.relative_to(REPO_ROOT))
    cfg = _build_run_config(
        rel_merged_path=rel_merged,
        dump_dir=rel_dump,
        hf_model=preset["hf_model"],
        train_layer=preset["train_layer"],
        max_concepts=args.max_concepts,
        num_of_examples=args.num_of_examples,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        master_data_dir=args.master_data_dir,
    )
    cfg_path = dump_dir / "mcp_hypersteer_config.yaml"
    _write_yaml(cfg, cfg_path)
    print(f"[mcp_hypersteer] Wrote run config -> {cfg_path}", flush=True)

    env = os.environ.copy()
    # `import axbench` needs the axbench project root (parent of the `axbench` package);
    # `from args` in generate/train needs the scripts directory.
    _pyp = f"{REPO_ROOT}{os.pathsep}{SCRIPTS_DIR}"
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = _pyp + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = _pyp

    if args.skip_generate:
        print(
            f"Wrote {merged_path} and {cfg_path} (skip generate/train).", flush=True
        )
        print(
            "[mcp_hypersteer] Next: run without --skip-generate to call generate.py, "
            "then train (omit --skip-train). Or run those scripts manually with:",
            f"\n  {cfg_path}",
            flush=True,
        )
        return

    print("[mcp_hypersteer] Step: generate (LM dataset) …", flush=True)
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "generate.py"),
            "--config",
            str(cfg_path.relative_to(REPO_ROOT)),
            "--dump_dir",
            rel_dump,
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    if args.skip_train:
        print("Skipping train (--skip-train).", flush=True)
        return
    print("[mcp_hypersteer] Step: train (HyperSteer) …", flush=True)
    torchrun = shutil.which("torchrun")
    if not torchrun:
        raise RuntimeError("torchrun not on PATH; install a PyTorch build with distributed tools.")
    _run(
        [
            torchrun,
            f"--nproc_per_node={args.nproc_per_node}",
            str(SCRIPTS_DIR / "train.py"),
            "--config",
            str(cfg_path.relative_to(REPO_ROOT)),
            "--dump_dir",
            rel_dump,
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    print(
        f"Done. Artifacts: {dump_dir} (train/ for checkpoints).",
        flush=True,
    )


if __name__ == "__main__":
    main()