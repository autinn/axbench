#!/usr/bin/env python3
"""Phase G inference server with concept-space search.

Model id: ``vG-search-local``. Same OpenAI-compatible chat-completions API as
``serve_mcp_hypersteer.py`` (drop-in for ``vf-eval -b ...``).

What this serves:
    1. At startup, load the trained vG HyperSteer (via the same path as
       ``serve_mcp_hypersteer.py``).
    2. Build a concept-fingerprint cache: for each training concept text, run
       the hypernet's ``concept_embedding`` with a *zero* base-encoder hidden
       state to get a prompt-agnostic concept vector. Stack as
       ``C ∈ [N_concepts, hidden_dim]``.
    3. On each request: take the chosen *embed mode* (system, system+user, or
       user-only) of the request, run it through the base model to get the
       L20 hidden state mean-pooled over tokens. Cosine-search ``q @ C.T`` for
       top-k concepts.
    4. For each top-k concept, run the *real* hypernet forward with the actual
       prompt hidden state; sum ``Σ v_i × VG_FACTOR`` and install at the
       intervention site via ``ax._update_v``. Generate.
    5. Log the top-k selections + similarities for each request to
       ``serve.log`` (stderr).

Knobs (env-var; defaults shown):
    VG_TOPK=3            top-k concepts to sum (hard cap: 5)
    VG_FACTOR=0.25       per-concept steering magnitude
    VG_EMBED_MODE=system  one of {system, system_user, user}
    HYPERSTEER_MAX_TOKENS=1024  default max-tokens for unconstrained requests
    MCP_ENABLE_THINKING=1  Qwen3 thinking mode (matches serve_mcp_hypersteer)

Sanity assertions performed at startup:
    * tokenizer (base) and hypernet_tokenizer agree on vocab size (Qwen3 == Qwen3 → ok)
    * concept-embedding rank ≥ 10 (otherwise mode-collapse risk; see §15-d)
    * concept count > 0
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import json
import logging
import os
import socket
import sys
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

# Package root: .../mcp-protect/axbench (parent of the `axbench` package)
_SCRIPT = Path(__file__).resolve()
_AX_PKG_PARENT = _SCRIPT.parents[2]
if str(_AX_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_AX_PKG_PARENT))

import pandas as pd
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
        "serve_mcp_vG_search requires fastapi, uvicorn, and pydantic. "
        "Install: pip install 'fastapi' 'uvicorn[standard]'"
    ) from e


_LOG = logging.getLogger("vG-search")
_LOG.setLevel(logging.INFO)
_h = logging.StreamHandler(sys.stderr)
_h.setFormatter(logging.Formatter("[%(asctime)s] [vG] %(message)s"))
_LOG.addHandler(_h)


class _ChatMessage(BaseModel):
    role: str
    content: str


class _ChatCompletionsRequest(BaseModel):
    model: str = "vG-search-local"
    messages: list[_ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False


def _axbench_dump_root(dump_dir: Path) -> Path:
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


def _hypersteer_model_params(train_cfg: dict[str, Any]) -> ModelParams:
    raw = (train_cfg.get("models") or {})["HyperSteer"]
    data = {k: v for k, v in raw.items() if k in {f.name for f in fields(ModelParams)}}
    return ModelParams(**data)


def _build_hsteer(train_cfg, train_dir, generate_dir, device) -> axbench.HyperSteer:
    """Mirrors serve_mcp_hypersteer._build_hsteer."""
    model_name = train_cfg["model_name"]
    layer = int(train_cfg["layer"])
    use_bf16 = bool(train_cfg.get("use_bf16", True))
    hparams = _hypersteer_model_params(train_cfg)
    meta_rows = _load_metadata_rows(generate_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=8192)
    tokenizer.padding_side = "right"
    need_resize = False
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        need_resize = True

    dtype = torch.bfloat16 if use_bf16 else None
    model_instance = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()
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
    h._inference_steer_kwargs = {
        "prefix_length": prefix_length,
        "batch_size": 1,
        "intervention_positions": hparams.intervention_positions,
    }
    h._is_chat = is_chat
    return h


def _ensure_dist() -> None:
    if not dist.is_available() or dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in os.environ:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        s.close()
    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=1, rank=0,
        timeout=datetime.timedelta(seconds=120),
    )

    def _destroy() -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
    atexit.register(_destroy)


def _messages_to_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    as_objs = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        _think = os.environ.get("MCP_ENABLE_THINKING", "1").lower() not in ("0", "false", "no")
        try:
            return tokenizer.apply_chat_template(
                as_objs, tokenize=False, add_generation_prompt=True, enable_thinking=_think,
            )
        except TypeError:
            return tokenizer.apply_chat_template(as_objs, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys_parts = [m["content"] for m in as_objs if m["role"] == "system"]
        rest = [m for m in as_objs if m["role"] != "system"]
        if sys_parts and rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": "\n\n".join(sys_parts) + "\n\n" + rest[0]["content"]}
        elif sys_parts:
            rest = [{"role": "user", "content": "\n\n".join(sys_parts)}] + rest
        return tokenizer.apply_chat_template(rest, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Concept-space search components
# ---------------------------------------------------------------------------
@torch.no_grad()
def _concept_fingerprint(h: axbench.HyperSteer, concept_text: str, hidden_dim: int) -> torch.Tensor:
    """Run concept_embedding(...) with a zero base-encoder hidden state to
    extract a prompt-agnostic concept fingerprint. Returns shape [hidden_dim].
    """
    tok = h.hypernet_tokenizer(
        [concept_text], return_tensors="pt",
        add_special_tokens=True, padding=True, truncation=True,
    ).to(h.device)
    bsz = tok["input_ids"].shape[0]
    # 1-token zero base-encoder state — content doesn't matter, we only want
    # the concept-side fingerprint. (The encoder cross-attends to it; with
    # zeros it contributes nothing meaningful and the regression head depends
    # only on the concept-input pathway.)
    zero_base = torch.zeros(bsz, 1, hidden_dim, device=h.device, dtype=torch.bfloat16)
    zero_mask = torch.ones(bsz, 1, device=h.device, dtype=tok["attention_mask"].dtype)
    v = h.concept_embedding(
        input_ids=tok["input_ids"],
        inputs_embeds=None,
        attention_mask=tok["attention_mask"],
        base_encoder_hidden_states=zero_base,
        base_encoder_attention_mask=zero_mask,
        output_hidden_states=False,
    ).last_hidden_state
    # v is [1, low_rank_dim, hidden_dim]; flatten to [hidden_dim] (rank-1 in our config).
    return v.squeeze(0).squeeze(0).to(torch.float32).cpu()


@torch.no_grad()
def _build_concept_cache(h: axbench.HyperSteer, concepts: list[dict], hidden_dim: int) -> tuple[torch.Tensor, list[dict]]:
    """Return (C [N, H] float32 cpu, ordered concept list)."""
    rows: list[torch.Tensor] = []
    for c in concepts:
        fp = _concept_fingerprint(h, c["concept"], hidden_dim)
        rows.append(fp)
    C = torch.stack(rows, dim=0)  # [N, H]
    return C, list(concepts)


@torch.no_grad()
def _embed_prompt(h: axbench.HyperSteer, prompt: str, layer: int) -> torch.Tensor:
    """Encode a prompt to a single [hidden_dim] vector by mean-pooling
    layer-`layer` hidden states (matches the L20 site of the steering hook).
    """
    inputs = h.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(h.device)
    out = h.model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        output_hidden_states=True,
    )
    hs = out.hidden_states[layer]                 # [1, T, H]
    mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)  # [1, T, 1]
    pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [1, H]
    return pooled.squeeze(0).to(torch.float32).cpu()


def _topk_concepts(q: torch.Tensor, C: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    """Cosine-sim. Return (indices, sims) of top-k descending."""
    qn = q / (q.norm() + 1e-8)
    Cn = C / (C.norm(dim=1, keepdim=True) + 1e-8)
    sims = (Cn @ qn).tolist()  # [N]
    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    return order, [sims[i] for i in order]


@torch.no_grad()
def _hypernet_v_for_prompt(
    h: axbench.HyperSteer, prompt: str, concept_text: str, layer: int
) -> torch.Tensor:
    """Run the *real* hypernet forward (prompt + concept) to get v.
    Returns [1, low_rank_dim, H] tensor on h.device, ready for ax._update_v.
    """
    pinputs = h.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(h.device)
    base_hidden = h.model(
        input_ids=pinputs["input_ids"], attention_mask=pinputs["attention_mask"],
        output_hidden_states=True,
    ).hidden_states[layer]
    cinputs = h.hypernet_tokenizer(
        [concept_text], return_tensors="pt",
        add_special_tokens=True, padding=True, truncation=True,
    ).to(h.device)
    v = h.concept_embedding(
        input_ids=cinputs["input_ids"],
        inputs_embeds=None,
        attention_mask=cinputs["attention_mask"],
        base_encoder_hidden_states=base_hidden,
        base_encoder_attention_mask=pinputs["attention_mask"],
        output_hidden_states=False,
    ).last_hidden_state
    return v


def _select_embed_text(messages: list[dict], mode: str) -> str:
    sys_parts = [m["content"] for m in messages if m["role"] == "system"]
    user_parts = [m["content"] for m in messages if m["role"] == "user"]
    if mode == "system":
        return "\n\n".join(sys_parts) if sys_parts else "\n\n".join(user_parts)
    if mode == "user":
        return "\n\n".join(user_parts) if user_parts else "\n\n".join(sys_parts)
    # default: system + user
    return "\n\n".join(sys_parts + user_parts)


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
def _serve(dump_dir: Path, config_path: Path, host: str, port: int, device: str,
           topk: int, factor: float, embed_mode: str, default_max_tokens: int) -> None:
    _ensure_dist()
    with config_path.open("r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg.get("train") or {}
    inference_cfg = full_cfg.get("inference")
    run_root = _axbench_dump_root(dump_dir)
    train_d = run_root / "train"
    gen_d = run_root / "generate"

    h = _build_hsteer(train_cfg, train_d, gen_d, device)
    layer = int(train_cfg["layer"])
    hidden_dim = int(h.model.config.hidden_size)

    # ---- assertions
    if h.tokenizer.vocab_size != h.hypernet_tokenizer.vocab_size:
        _LOG.warning(
            "tokenizer vocab mismatch: base=%d hypernet=%d (concept text path may differ)",
            h.tokenizer.vocab_size, h.hypernet_tokenizer.vocab_size,
        )

    meta = _load_metadata_rows(gen_d)
    if not meta:
        raise SystemExit("metadata.jsonl is empty — cannot build concept cache")
    _LOG.info("loaded %d concepts from %s", len(meta), gen_d / "metadata.jsonl")
    _LOG.info("building concept fingerprint cache (this may take ~1-2 min for 400 concepts)...")
    C, ordered_concepts = _build_concept_cache(h, meta, hidden_dim)
    _LOG.info("concept cache shape: %s", tuple(C.shape))

    # Sanity check: rank of C (mode-collapse diagnostic, §15-d).
    try:
        rank = int(torch.linalg.matrix_rank(C, tol=1e-3).item())
    except Exception:
        rank = -1
    _LOG.info("concept-cache effective rank: %s (low rank ⇒ mode-collapse risk)", rank)
    if rank != -1 and rank < 10:
        _LOG.warning("concept manifold may have collapsed — kNN search may not differentiate concepts")

    topk = max(1, min(int(topk), 5))  # hard cap top-k=5 (§15-b)
    _LOG.info(
        "serve config: top-k=%d factor=%.3f embed_mode=%s layer=%d max_tokens_default=%d",
        topk, factor, embed_mode, layer, default_max_tokens,
    )

    app = FastAPI(title="vG concept-space-search server", version="0.1.0")
    kws = getattr(h, "_inference_steer_kwargs", {})
    if inference_cfg and "steering_output_length" in inference_cfg:
        kws["eval_output_length"] = int(inference_cfg["steering_output_length"])

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "vG-search-local", "object": "model"}]}

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    def chat_completions(req: _ChatCompletionsRequest) -> JSONResponse:
        if req.stream:
            raise HTTPException(400, "Streaming not supported; use stream=false")
        if not req.messages:
            raise HTTPException(400, "messages is required")

        msgs = [m.model_dump() for m in req.messages]

        # Choose generation prompt (chat template applied if base model is chat-tuned).
        if h._is_chat:
            user_prompt = _messages_to_prompt(h.tokenizer, msgs)
        else:
            user_prompt = msgs[-1]["content"]

        # Choose embedding text (may be just system, system+user, or user).
        embed_text = _select_embed_text(msgs, embed_mode)
        if not embed_text.strip():
            embed_text = user_prompt  # fallback

        # ---- concept search
        with torch.inference_mode():
            q = _embed_prompt(h, embed_text, layer)
            top_idx, top_sims = _topk_concepts(q, C, topk)
            top_concepts = [ordered_concepts[i] for i in top_idx]

            _LOG.info(
                "topk=%s sims=%s",
                [int(c["concept_id"]) for c in top_concepts],
                [round(s, 3) for s in top_sims],
            )
            for c, s in zip(top_concepts, top_sims):
                txt = (c.get("concept") or "")[:80].replace("\n", " ")
                _LOG.info("  cid=%-3d sim=%.3f text=%s%s",
                          int(c["concept_id"]), s, txt, "..." if len(c.get("concept", "")) > 80 else "")

            # ---- compute summed v for top-k
            v_sum = None
            for c in top_concepts:
                v_i = _hypernet_v_for_prompt(h, user_prompt, c["concept"], layer)
                v_sum = v_i if v_sum is None else (v_sum + v_i)

            # Cap max_tokens per request
            req_cap = req.max_completion_tokens if req.max_completion_tokens else req.max_tokens
            try:
                max_toks = int(req_cap) if req_cap else default_max_tokens
            except (TypeError, ValueError):
                max_toks = default_max_tokens

            # ---- install summed v at intervention site, generate
            inputs = h.tokenizer(
                user_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048,
            ).to(h.device)
            mag = torch.tensor([float(factor)], device=h.device)
            idx = torch.tensor([int(top_concepts[0]["concept_id"])], device=h.device)
            h.ax._update_v(v_sum)
            try:
                _, generations = h.ax_model.generate(
                    inputs, unit_locations=None, intervene_on_prompt=True,
                    subspaces=[{"idx": idx, "mag": mag, "prefix_length": kws["prefix_length"]}] * h.num_of_layers,
                    max_new_tokens=max(1, max_toks),
                    do_sample=True, temperature=float(req.temperature) if req.temperature else 1.0,
                )
            finally:
                h.ax._reset_v()

            input_lens = [len(ids) for ids in inputs.input_ids]
            generated_text = h.tokenizer.decode(
                generations[0][input_lens[0]:], skip_special_tokens=True,
            )

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": cid,
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop",
            }],
            "vG_top_concepts": [
                {"concept_id": int(c["concept_id"]), "sim": round(s, 4),
                 "concept": (c.get("concept") or "")[:120]}
                for c, s in zip(top_concepts, top_sims)
            ],
        })

    base = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1"
    print(
        f"\n[serve_mcp_vG_search] OpenAI API base (use with vf-eval: -b {base} -k EMPTY):\n  {base}\n",
        flush=True,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    p = argparse.ArgumentParser(description="vG concept-search HyperSteer server")
    p.add_argument("--dump-dir", type=Path, required=True,
                   help="Run dir with train/ and generate/ subdirs.")
    p.add_argument("--config", type=Path, default=None,
                   help="YAML written by mcp_hypersteer (default: <dump-dir>/mcp_hypersteer_config.yaml)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--topk", type=int, default=None,
                   help="(or env VG_TOPK; default 3, hard cap 5)")
    p.add_argument("--factor", type=float, default=None,
                   help="(or env VG_FACTOR; default 0.25)")
    p.add_argument("--embed-mode", default=None,
                   help="(or env VG_EMBED_MODE; default 'system'; options: system, system_user, user)")
    args = p.parse_args()

    run = _axbench_dump_root(args.dump_dir)
    cfg = args.config or (run / "mcp_hypersteer_config.yaml")
    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")

    def _getint(env_key: str, cli: int | None, default: int) -> int:
        if cli is not None:
            return int(cli)
        v = os.environ.get(env_key)
        return int(v) if v is not None and str(v).strip() else default

    def _getfloat(env_key: str, cli: float | None, default: float) -> float:
        if cli is not None:
            return float(cli)
        v = os.environ.get(env_key)
        return float(v) if v is not None and str(v).strip() else default

    def _getstr(env_key: str, cli: str | None, default: str) -> str:
        if cli is not None:
            return str(cli)
        v = os.environ.get(env_key)
        return str(v) if v is not None and str(v).strip() else default

    topk = _getint("VG_TOPK", args.topk, 3)
    factor = _getfloat("VG_FACTOR", args.factor, 0.25)
    embed_mode = _getstr("VG_EMBED_MODE", args.embed_mode, "system")
    if embed_mode not in {"system", "system_user", "user"}:
        raise SystemExit(f"--embed-mode must be one of system, system_user, user (got {embed_mode!r})")
    max_toks = int(os.environ.get("HYPERSTEER_MAX_TOKENS", "1024"))

    _serve(args.dump_dir, Path(cfg), args.host, args.port, args.device,
           topk, factor, embed_mode, max_toks)


if __name__ == "__main__":
    main()
