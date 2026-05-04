#!/usr/bin/env python3
"""
OpenAI-compatible **local policy** server for a **trained HyperSteer** run: loads Gemma + the
hypernet from ``<dump-dir>/train/`` and answers ``POST /v1/chat/completions`` with
``HyperSteer.predict_steer`` so **prime-envs** ``vf-eval`` can use **``-b http://HOST:PORT/v1``**
(with **``-k EMPTY``**). The repository root **README** describes the full train → serve →
benchmark story.

Run from the **AxBench project root** (parent of the inner ``axbench/`` package). The script
inserts that root on ``sys.path`` (same pattern as ``mcp_hypersteer``). Requires
``fastapi``, ``uvicorn``, ``pydantic`` (see ``requirements-serve.txt``) plus a CUDA-capable PyTorch
install for practical latency.

**End-to-end steps (how each is achieved)**

1. **``main()`` — CLI and steering defaults** — Parses ``--dump-dir`` (a run with ``train/`` and
   ``generate/`` from ``mcp_hypersteer.py``), optional ``--config`` (default
   ``<dump-dir>/mcp_hypersteer_config.yaml``), ``--host`` / ``--port`` / ``--device``,
   ``--concept-id`` and ``--factor``. Missing concept/factor fall back to **``HYPERSTEER_*``** env
   vars; if ``concept_id`` is unset, the **first** row in ``generate/metadata.jsonl`` is used.
   ``HYPERSTEER_MAX_TOKENS`` caps generation length (default 1024). Resolves the run root with
   ``_axbench_dump_root`` then calls ``_serve``.

2. **``_serve()`` — process group for ``predict_steer``** — Calls ``_ensure_dist_for_predict_steer`` so
   ``torch.distributed.get_rank()`` works (single-rank ``gloo``) without running under ``torchrun``;
   register ``atexit`` to tear down the group. Loads the merged YAML: ``train`` and ``inference``
   sections drive model construction.

3. **``_build_hsteer()`` — load base LM + HyperSteer checkpoint** — Reads ``generate/metadata.jsonl``
   via ``_load_metadata_rows``, instantiates the HuggingFace causal LM and tokenizer
   (``train.model_name``, ``use_bf16``), builds ``axbench.HyperSteer`` with
   ``_hypersteer_model_params`` from the YAML, then ``h.load`` from ``train/`` in steering mode
   (mirrors the preload path in ``inference.py``).

4. **Fix one steering concept per process** — Builds a ``concept_id`` → text map from
   ``metadata.jsonl``, checks ``--concept-id`` (or default) is valid, sets
   ``h._fixed_concept_text``, ``h._steer_concept_id``, ``h._steer_factor`` for each request
   (single concept per server; restart to change, or use env/CLI at startup).

5. **FastAPI app and OpenAI surface** — Registers ``GET /v1/models`` (e.g. ``local-hypersteer``),
   ``GET /healthz``, and ``POST /v1/chat/completions`` (no streaming). The handler flattens chat
   messages with ``_messages_to_prompt`` when the base model is a chat model, builds a one-row
   pandas frame for ``predict_steer``, and returns a JSON body shaped like OpenAI’s chat
   completion. On startup, prints a copy-paste **base URL** ending in **``/v1``** for
   ``vf-eval -b ...``.

6. **``uvicorn.run``** — Serves the app; blocks until stopped.

**Environment (optional; CLI wins when set)**

* ``HYPERSTEER_CONCEPT_ID`` — ``concept_id`` in ``metadata.jsonl`` to steer on.
* ``HYPERSTEER_FACTOR`` — steering strength (default ``1.0``).
* ``HYPERSTEER_MAX_TOKENS`` — default max new tokens (default ``1024``).

**Example**::

    python axbench/mcp-protect/serve_mcp_hypersteer.py --dump-dir axbench/outputs/mcp_hsteer_2b --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
import atexit
import datetime
import socket
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch.distributed as dist

# Package root: .../mcp-protect/axbench (parent of the `axbench` package)
_SCRIPT = Path(__file__).resolve()
_AX_PKG_PARENT = _SCRIPT.parents[2]  # .../axbench (contains axbench/ package)
if str(_AX_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_AX_PKG_PARENT))

import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

import axbench
from axbench.utils.constants import CHAT_MODELS
from axbench.utils.model_utils import get_prefix_length
from axbench.scripts.args.training_args import ModelParams

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise SystemExit(
        "serve_mcp_hypersteer requires fastapi, uvicorn, and pydantic. "
        "Install: pip install 'fastapi' 'uvicorn[standard]'\n"
    ) from e


class _ChatMessage(BaseModel):
    role: str
    content: str


class _ChatCompletionsRequest(BaseModel):
    model: str = "hypersteer-local"
    messages: list[_ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False


def _axbench_dump_root(dump_dir: Path) -> Path:
    """Run directory containing `generate/` and `train/`."""
    p = dump_dir.resolve()
    if (p / "train").is_dir() and (p / "generate").is_dir():
        return p
    if (p.parent / "train").is_dir() and (p.parent / "generate").is_dir():
        return p.parent
    raise FileNotFoundError(
        f"Expected --dump-dir to be a run dir with `train/` and `generate/`, got: {dump_dir}"
    )


def _load_metadata_rows(generate_dir: Path) -> list[dict[str, Any]]:
    path = generate_dir / "metadata.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _hypersteer_model_params(
    train_cfg: dict[str, Any], model_name: str = "HyperSteer"
) -> ModelParams:
    raw = (train_cfg.get("models") or {})[model_name]
    data = {k: v for k, v in raw.items() if k in {f.name for f in fields(ModelParams)}}
    return ModelParams(**data)


def _build_hsteer(
    train_cfg: dict[str, Any],
    inference_cfg: dict[str, Any] | None,
    train_dir: Path,
    generate_dir: Path,
    device: str,
) -> axbench.HyperSteer:
    """Match `infer_steering` for HyperSteer in inference.py (preload path)."""
    model_name = train_cfg["model_name"]
    layer = int(train_cfg["layer"])
    use_bf16 = bool(train_cfg.get("use_bf16", True))
    hparams = _hypersteer_model_params(train_cfg)
    meta_rows = _load_metadata_rows(generate_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=False, model_max_length=1024
    )
    tokenizer.padding_side = "right"
    need_resize = False
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    dtype = torch.bfloat16 if use_bf16 else None
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    model_instance = model_instance.eval()
    if need_resize:
        model_instance.resize_token_embeddings(len(tokenizer))

    is_chat = model_name in CHAT_MODELS
    prefix_length = 1
    if is_chat:
        prefix_length = int(get_prefix_length(tokenizer))
    h = axbench.HyperSteer(
        model_instance,
        tokenizer,
        layer=layer,
        low_rank_dimension=len(meta_rows),
        device=device,
        training_args=hparams,
        lm_model_name=model_name,
    )
    h.load(
        dump_dir=str(train_dir),
        low_rank_dimension=1,
        mode="steering",
        hypernet_initialize_from_pretrained=hparams.hypernet_initialize_from_pretrained,
        hypernet_name_or_path=hparams.hypernet_name_or_path,
        num_hidden_layers=hparams.num_hidden_layers,
    )
    h.concept_embedding.eval()
    h._inference_steer_kwargs = {  # type: ignore[attr-defined]
        "prefix_length": prefix_length,
        "batch_size": 1,
        "intervention_positions": hparams.intervention_positions,
    }
    return h


def _ensure_dist_for_predict_steer() -> None:
    """
    :meth:`HyperSteer.predict_steer` calls ``torch.distributed.get_rank()``. Training
    always uses ``torchrun`` (process group already initialized). This process does not,
    so we start a 1-rank ``gloo`` group here—**no edits to** ``axbench/models/hypersteer.py``
    are required for that API contract. If you only ever call ``predict_steer`` under
    ``torchrun``, this is unnecessary; it is safe to call once per server process.
    """
    if not dist.is_available() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        world_size=1,
        rank=0,
        timeout=datetime.timedelta(seconds=120),
    )

    def _destroy() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    atexit.register(_destroy)


def _messages_to_prompt(
    model_name: str, tokenizer, messages: list[dict[str, str]]
) -> str:
    if not messages:
        return ""
    as_objs = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        # Qwen3 chat template accepts enable_thinking=True; other model
        # families either ignore it or raise on unknown kwarg.
        try:
            return tokenizer.apply_chat_template(
                as_objs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                as_objs, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        # Gemma chat template has no system role; fold system into the first user turn.
        sys_parts = [m["content"] for m in as_objs if m["role"] == "system"]
        rest = [m for m in as_objs if m["role"] != "system"]
        if sys_parts and rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": "\n\n".join(sys_parts) + "\n\n" + rest[0]["content"]}
        elif sys_parts:
            rest = [{"role": "user", "content": "\n\n".join(sys_parts)}] + rest
        return tokenizer.apply_chat_template(
            rest, tokenize=False, add_generation_prompt=True
        )


def _serve(
    dump_dir: Path,
    config_path: Path,
    host: str,
    port: int,
    device: str,
    concept_id: int,
    factor: float,
    default_max_tokens: int,
) -> None:
    _ensure_dist_for_predict_steer()
    with config_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg.get("train") or {}
    inference_cfg = full_cfg.get("inference")
    run_root = _axbench_dump_root(dump_dir)
    train_d = run_root / "train"
    gen_d = run_root / "generate"

    h = _build_hsteer(train_cfg, inference_cfg, train_d, gen_d, device)
    meta = _load_metadata_rows(gen_d)
    by_id = {int(r["concept_id"]): r["concept"] for r in meta if "concept_id" in r and "concept" in r}
    if concept_id not in by_id:
        raise SystemExit(
            f"concept_id {concept_id} not in {gen_d / 'metadata.jsonl'} (available: {sorted(by_id.keys())[:20]}...)"
        )
    model_name = train_cfg["model_name"]
    is_chat = model_name in CHAT_MODELS
    h._fixed_concept_text = by_id[concept_id]  # type: ignore[attr-defined]
    h._steer_concept_id = concept_id  # type: ignore[attr-defined]
    h._steer_factor = float(factor)  # type: ignore[attr-defined]

    app = FastAPI(title="HyperSteer local policy", version="0.1.0")
    kws = getattr(h, "_inference_steer_kwargs", {})
    if inference_cfg and "steering_output_length" in inference_cfg:
        kws["eval_output_length"] = int(inference_cfg["steering_output_length"])

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "local-hypersteer",
                    "object": "model",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: _ChatCompletionsRequest) -> JSONResponse:
        if req.stream:
            raise HTTPException(400, "Streaming not supported yet; use stream=false")
        if not req.messages:
            raise HTTPException(400, "messages is required")
        if is_chat:
            user_prompt = _messages_to_prompt(
                model_name, h.tokenizer, [m.model_dump() for m in req.messages]
            )
        else:
            user_prompt = req.messages[-1].content
        # Accept either max_tokens (legacy) or max_completion_tokens (newer OpenAI).
        # Fall back to default_max_tokens (set on serve startup) if neither given.
        req_cap = req.max_completion_tokens if req.max_completion_tokens else req.max_tokens
        try:
            max_toks = int(req_cap) if req_cap else default_max_tokens
        except (TypeError, ValueError):
            max_toks = default_max_tokens
        row = {
            "input": user_prompt,
            "factor": float(h._steer_factor),  # type: ignore[attr-defined]
            "concept_id": int(h._steer_concept_id),  # type: ignore[attr-defined]
            "input_concept": h._fixed_concept_text,  # type: ignore[attr-defined]
        }
        df = pd.DataFrame([row])
        with torch.inference_mode():
            out = h.predict_steer(
                df,
                **{
                    **kws,
                    "eval_output_length": max(1, max_toks),
                    "temperature": float(req.temperature) if req.temperature else 1.0,
                },
            )
        text = (out.get("steered_generation") or [""])[0]
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return JSONResponse(
            {
                "id": cid,
                "object": "chat.completion",
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # Print URL for copy-paste into prime-envs `vf-eval -b ...`
    base = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1"
    print(
        f"\n[serve_mcp_hypersteer] OpenAI API base (use with vf-eval: -b {base} -k EMPTY):\n  {base}\n",
        flush=True,
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    p = argparse.ArgumentParser(description="HyperSteer OpenAI-compatible local server")
    p.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="Run directory that contains train/ and generate/ (same as mcp_hypersteer --dump-dir).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML written by mcp_hypersteer (default: <dump-dir>/mcp_hypersteer_config.yaml).",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--concept-id",
        type=int,
        default=None,
        help="Concept id in metadata.jsonl (or env HYPERSTEER_CONCEPT_ID).",
    )
    p.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Steering strength (or env HYPERSTEER_FACTOR, default 1.0).",
    )
    args = p.parse_args()
    run = _axbench_dump_root(args.dump_dir)
    cfg = args.config or (run / "mcp_hypersteer_config.yaml")
    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg} — pass --config or run mcp_hypersteer to generate it.")
    def _getint(k: str, a: int | None, default: int) -> int:
        if a is not None:
            return int(a)
        v = os.environ.get(k)
        if v is not None and str(v).strip() != "":
            return int(v)
        return default

    def _getfloat(k: str, a: float | None, default: float) -> float:
        if a is not None:
            return float(a)
        v = os.environ.get(k)
        return float(v) if v is not None else default

    meta0 = _load_metadata_rows(run / "generate")
    if not meta0:
        raise SystemExit("No rows in generate/metadata.jsonl")
    first_id = int(meta0[0]["concept_id"])
    cid = _getint("HYPERSTEER_CONCEPT_ID", args.concept_id, first_id)
    fact = _getfloat("HYPERSTEER_FACTOR", args.factor, 1.0)
    max_toks = int(os.environ.get("HYPERSTEER_MAX_TOKENS", "1024"))

    _serve(
        args.dump_dir,
        Path(cfg),
        args.host,
        args.port,
        args.device,
        cid,
        fact,
        max_toks,
    )


if __name__ == "__main__":
    main()
