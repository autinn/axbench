#!/usr/bin/env python3
"""OpenAI-compatible local server that loads a trained **LoReFT** checkpoint
(``<dump-dir>/LoReFT_weight.pt`` + ``<dump-dir>/LoReFT_bias.pt``) and applies
the LoReFT intervention at inference.

Counterpart to ``serve_mcp_hypersteer.py`` (which is hardcoded for
HyperSteer's ``train/hyperreft/model.safetensors`` layout) and to
``serve_mcp_multivec.py`` (which assumes a static additive vector).

LoReFT's intervention is

    h' = h + R^T (W h + b - R h)

which depends on ``h`` itself, so it cannot be reduced to a constant
additive vector — the actual intervention must run inside the forward pass.

Both ``axbench.LoReFT.save`` (see ``axbench/models/reft.py``) and
``axbench.LoReFT.load`` round-trip the parameters as 3D tensors
``[n_concepts, embed_dim, low_rank_dim]`` (and ``[n_concepts, low_rank_dim]``
for bias). For the single-concept ReFT-CE runs in D4a/D4b, ``n_concepts=1``
and ``low_rank_dim=1``.

Implementation: rather than reconstructing the full pyreft
``IntervenableModel`` (which requires distributed init + a configured
``LoReFT`` axbench wrapper), we materialize the three tensors directly into
plain ``nn.Parameter``s and register a forward hook on
``model.model.layers[L]`` that applies

    delta = factor * R^T ((W h + b) - R h)
    new_hidden = h + delta

at every token position (matching ``intervention_positions: all``).

CLI / env mirrors ``serve_mcp_hypersteer.py`` for chain-script compatibility:

* ``--dump-dir``  — directory containing ``LoReFT_weight.pt`` + ``LoReFT_bias.pt``
* ``HYPERSTEER_FACTOR`` — intervention strength (default 1.0)
* ``HYPERSTEER_CONCEPT_ID`` — concept index into ``n_concepts`` axis (default 0)
* ``HYPERSTEER_MAX_TOKENS`` — default max new tokens (default 1024)
* ``MCP_ENABLE_THINKING`` — Qwen3 chat-template ``enable_thinking`` toggle

Model id reported on ``/v1/models`` is ``loreft-local``.

Usage::

    python axbench/mcp-protect/serve_mcp_loreft.py \
        --dump-dir axbench/outputs/reft_v17_d4a --port 8000 \
        --model-name Qwen/Qwen3-8B --layer 20
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Same package-root insertion as the other serve scripts.
_SCRIPT = Path(__file__).resolve()
_AX_PKG_PARENT = _SCRIPT.parents[2]
if str(_AX_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_AX_PKG_PARENT))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from axbench.utils.constants import CHAT_MODELS  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise SystemExit(
        "serve_mcp_loreft requires fastapi, uvicorn, pydantic. "
        "Install: pip install 'fastapi' 'uvicorn[standard]'\n"
    ) from e


class _ChatMessage(BaseModel):
    role: str
    content: str


class _ChatCompletionsRequest(BaseModel):
    model: str = "loreft-local"
    messages: list[_ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False


def _messages_to_prompt(model_name: str, tokenizer, messages: list[dict[str, str]]) -> str:
    """Verbatim copy of serve_mcp_hypersteer._messages_to_prompt."""
    if not messages:
        return ""
    as_objs = [{"role": m["role"], "content": m["content"]} for m in messages]
    try:
        _think = os.environ.get("MCP_ENABLE_THINKING", "1").lower() not in ("0", "false", "no")
        try:
            return tokenizer.apply_chat_template(
                as_objs, tokenize=False, add_generation_prompt=True,
                enable_thinking=_think,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                as_objs, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        sys_parts = [m["content"] for m in as_objs if m["role"] == "system"]
        rest = [m for m in as_objs if m["role"] != "system"]
        if sys_parts and rest and rest[0]["role"] == "user":
            rest[0] = {"role": "user", "content": "\n\n".join(sys_parts) + "\n\n" + rest[0]["content"]}
        elif sys_parts:
            rest = [{"role": "user", "content": "\n\n".join(sys_parts)}] + rest
        return tokenizer.apply_chat_template(
            rest, tokenize=False, add_generation_prompt=True
        )


class _LoreftSlice(nn.Module):
    """A single-concept slice of a saved LoReFT checkpoint, ready to be applied
    in a forward hook.

    Parameters (matching ``ConceptReFTIntervention`` and the LoReFT save layout
    in ``axbench/models/reft.py``):

    * ``proj_weight`` — ``R^T``'s factor: ``[embed_dim, low_rank_dim]``
    * ``source_weight`` — ``W^T``'s factor: ``[embed_dim, low_rank_dim]``
    * ``source_bias`` — ``b``: ``[low_rank_dim]``

    Forward returns ``delta = R^T ((W h + b) - R h)`` (same shape as ``h``).
    The caller multiplies by ``factor`` and adds to ``h``.
    """

    def __init__(self, proj_weight: torch.Tensor, source_weight: torch.Tensor,
                 source_bias: torch.Tensor):
        super().__init__()
        # Buffers (not Parameters) — frozen at inference; no autograd state.
        self.register_buffer("proj_weight", proj_weight)      # [E, R]
        self.register_buffer("source_weight", source_weight)  # [E, R]
        self.register_buffer("source_bias", source_bias)      # [R]

    @torch.no_grad()
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, S, E]; cast intermediates to float32 for numerical parity with
        # ConceptReFTIntervention.forward (which upcasts via base.float() for
        # rotated_base). Final delta is recast back to h.dtype by the caller.
        proj = self.proj_weight.to(dtype=torch.float32)        # [E, R]
        source_w = self.source_weight.to(dtype=torch.float32)  # [E, R]
        source_b = self.source_bias.to(dtype=torch.float32)    # [R]

        h32 = h.to(torch.float32)
        rotated = torch.matmul(h32, proj)                      # [B, S, R]
        sourced = torch.matmul(h32, source_w) + source_b       # [B, S, R]
        delta = torch.matmul(sourced - rotated, proj.transpose(-1, -2))  # [B, S, E]
        return delta


def _load_loreft_slice(dump_dir: Path, concept_id: int, layer: int,
                       expected_embed_dim: int) -> _LoreftSlice:
    """Load LoReFT_weight.pt + LoReFT_bias.pt and slice out one concept.

    Saved tensors (see ``LoReFT.save`` in ``axbench/models/reft.py``):

    * ``<key>.proj_weight``   shape ``[n_concepts, E, R]``
    * ``<key>.source_weight`` shape ``[n_concepts, E, R]``
    * ``<key>.bias``          shape ``[n_concepts, R]``

    The dict key prefix encodes the pyreft intervention name (e.g.
    ``layer.20.comp.block_output.unit.pos.nunit.1#0``). For a single-layer
    LoReFT run there is exactly one prefix; we take it without trying to
    match exactly (``layer`` is informational here, used for logs).
    """
    weight_path = dump_dir / "LoReFT_weight.pt"
    bias_path = dump_dir / "LoReFT_bias.pt"
    if not weight_path.is_file():
        raise SystemExit(f"Missing {weight_path}")
    if not bias_path.is_file():
        raise SystemExit(f"Missing {bias_path}")

    weight_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
    bias_dict = torch.load(bias_path, map_location="cpu", weights_only=False)

    # Group keys by their `<intervention_name>` prefix.
    prefixes = sorted({k.rsplit(".", 1)[0] for k in weight_dict.keys()})
    if not prefixes:
        raise SystemExit(f"{weight_path} has no entries")
    if len(prefixes) > 1:
        # Multi-layer LoReFT: pick the prefix matching the requested layer if
        # we can identify it (`layer.<L>.` substring), else first.
        match = [p for p in prefixes if f"layer.{layer}." in p]
        prefix = match[0] if match else prefixes[0]
        print(f"[loreft] {len(prefixes)} intervention prefixes found; using {prefix!r}",
              flush=True)
    else:
        prefix = prefixes[0]

    proj = weight_dict[f"{prefix}.proj_weight"]      # [n_concepts, E, R]
    src_w = weight_dict[f"{prefix}.source_weight"]   # [n_concepts, E, R]
    src_b = bias_dict[f"{prefix}.bias"]              # [n_concepts, R]

    n_concepts = int(proj.shape[0])
    embed_dim = int(proj.shape[1])
    low_rank = int(proj.shape[2])

    if embed_dim != expected_embed_dim:
        raise SystemExit(
            f"embed_dim mismatch: checkpoint has {embed_dim}, model has {expected_embed_dim}"
        )
    if not (0 <= concept_id < n_concepts):
        raise SystemExit(
            f"concept_id={concept_id} out of range; checkpoint has n_concepts={n_concepts}"
        )

    print(
        f"[loreft] loaded prefix={prefix!r} n_concepts={n_concepts} embed_dim={embed_dim} "
        f"low_rank={low_rank}; using concept_id={concept_id}",
        flush=True,
    )

    return _LoreftSlice(
        proj_weight=proj[concept_id].clone(),     # [E, R]
        source_weight=src_w[concept_id].clone(),  # [E, R]
        source_bias=src_b[concept_id].clone(),    # [R]
    )


def _make_hook(slice_module: _LoreftSlice, factor_ref: dict[str, float]):
    """Forward hook on a Qwen3 decoder layer.

    HF Qwen3 / Llama / Gemma decoder layers return a tuple
    ``(hidden_states, ...)``. We compute the LoReFT delta on
    ``hidden_states`` and rebuild the tuple. ``factor_ref`` is a 1-element
    dict so we can mutate the factor at runtime (currently unused — server
    fixes factor at startup — but cheap to keep).
    """
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            delta = slice_module(hidden)
            scaled = (factor_ref["factor"] * delta).to(hidden.dtype)
            return (hidden + scaled,) + output[1:]
        hidden = output
        delta = slice_module(hidden)
        scaled = (factor_ref["factor"] * delta).to(hidden.dtype)
        return hidden + scaled

    return hook


def _serve(
    dump_dir: Path,
    host: str,
    port: int,
    device: str,
    model_name: str,
    layer: int,
    use_bf16: bool,
    concept_id: int,
    factor: float,
    default_max_tokens: int,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=8192)
    tokenizer.padding_side = "left"
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dtype = torch.bfloat16 if use_bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    ).eval()

    hidden_dim = int(model.config.hidden_size)
    slice_mod = _load_loreft_slice(dump_dir, concept_id=concept_id, layer=layer,
                                   expected_embed_dim=hidden_dim)
    slice_mod = slice_mod.to(device=device)

    factor_ref = {"factor": float(factor)}
    print(
        f"[loreft] startup factor={factor_ref['factor']} concept_id={concept_id} "
        f"layer={layer} model={model_name}",
        flush=True,
    )

    # Resolve target layer module (Qwen3 / Llama / Gemma layout).
    try:
        layer_mod = model.model.layers[layer]
    except AttributeError:
        layer_mod = model.layers[layer]  # type: ignore[attr-defined]
    handle = layer_mod.register_forward_hook(_make_hook(slice_mod, factor_ref))

    is_chat = model_name in CHAT_MODELS
    if not is_chat:
        print(
            f"[loreft] WARNING: model_name={model_name!r} not in CHAT_MODELS; "
            f"chat template will be bypassed (raw last-message string used). "
            f"This is the silent-bypass bug noted in feedback memory.",
            flush=True,
        )

    app = FastAPI(title="LoReFT local policy", version="0.1.0")

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "loreft-local", "object": "model"}]}

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "factor": str(factor_ref["factor"]),
            "concept_id": str(concept_id),
            "layer": str(layer),
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: _ChatCompletionsRequest) -> JSONResponse:
        if req.stream:
            raise HTTPException(400, "Streaming not supported; use stream=false")
        if not req.messages:
            raise HTTPException(400, "messages is required")
        if is_chat:
            prompt = _messages_to_prompt(
                model_name, tokenizer, [m.model_dump() for m in req.messages]
            )
        else:
            prompt = req.messages[-1].content

        req_cap = req.max_completion_tokens if req.max_completion_tokens else req.max_tokens
        try:
            max_toks = int(req_cap) if req_cap else default_max_tokens
        except (TypeError, ValueError):
            max_toks = default_max_tokens

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max(1, max_toks),
            "do_sample": True,
            "temperature": float(req.temperature) if req.temperature else 1.0,
        }
        if tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

        with torch.inference_mode():
            out_ids = model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": cid,
            "object": "chat.completion",
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        })

    base = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}/v1"
    print(
        f"\n[serve_mcp_loreft] OpenAI API base (vf-eval -b {base} -k EMPTY):\n  {base}\n",
        flush=True,
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        handle.remove()


def _getint(env_key: str, cli_val: int | None, default: int) -> int:
    if cli_val is not None:
        return int(cli_val)
    v = os.environ.get(env_key)
    if v is not None and str(v).strip() != "":
        return int(v)
    return default


def _getfloat(env_key: str, cli_val: float | None, default: float) -> float:
    if cli_val is not None:
        return float(cli_val)
    v = os.environ.get(env_key)
    if v is not None and str(v).strip() != "":
        return float(v)
    return default


def main() -> None:
    p = argparse.ArgumentParser(description="LoReFT OpenAI-compatible local server")
    p.add_argument(
        "--dump-dir",
        type=Path,
        required=True,
        help="Directory containing LoReFT_weight.pt and LoReFT_bias.pt.",
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=20,
                   help="Decoder layer index where the LoReFT intervention applies.")
    p.add_argument("--no-bf16", action="store_true", help="Disable bf16 (default on)")
    p.add_argument(
        "--concept-id",
        type=int,
        default=None,
        help="Index into n_concepts axis (or env HYPERSTEER_CONCEPT_ID, default 0).",
    )
    p.add_argument(
        "--factor",
        type=float,
        default=None,
        help="Intervention strength (or env HYPERSTEER_FACTOR, default 1.0).",
    )
    args = p.parse_args()

    if not args.dump_dir.is_dir():
        raise SystemExit(f"--dump-dir is not a directory: {args.dump_dir}")

    cid = _getint("HYPERSTEER_CONCEPT_ID", args.concept_id, 0)
    fact = _getfloat("HYPERSTEER_FACTOR", args.factor, 1.0)
    max_toks = int(os.environ.get("HYPERSTEER_MAX_TOKENS", "1024"))

    _serve(
        dump_dir=args.dump_dir,
        host=args.host,
        port=args.port,
        device=args.device,
        model_name=args.model_name,
        layer=int(args.layer),
        use_bf16=not args.no_bf16,
        concept_id=cid,
        factor=fact,
        default_max_tokens=max_toks,
    )


if __name__ == "__main__":
    main()
