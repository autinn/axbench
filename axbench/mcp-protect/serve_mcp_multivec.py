#!/usr/bin/env python3
"""OpenAI-compatible local server that adds N pre-extracted steering vectors at
a single layer of the base policy LM.

This is the *inference-time composition* twin of ``serve_mcp_hypersteer.py``.
Instead of running the full HyperSteer hypernet, it consumes a list of
``(vec_path, factor)`` pairs (each ``.pt`` produced by
``extract_hypersteer_vec.py``) and registers a single forward hook on
``model.model.layers[L]`` that adds ``sum(f_i * v_i)`` to the output's first
tensor (the residual hidden state) at *all* token positions — matching
``intervention_positions: all`` from the trained HyperSteer config.

Used for:

* **D12** (multi-style additive defense): v17 + v19 simultaneously at L20.
* **D7**  (cross-attack composition): 3-5 v22 cids each at low factor.

API surface mirrors ``serve_mcp_hypersteer.py`` so ``vf-eval -b
http://HOST:PORT/v1`` Just Works (model id is ``local-multivec``).

Usage::

    python axbench/mcp-protect/serve_mcp_multivec.py \
        --vec-spec '[{"path":"/root/v17_vec.pt","factor":0.5},{"path":"/root/v19_vec.pt","factor":0.3}]' \
        --port 8000 --layer 20 --model-name Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Same package-root insertion as serve_mcp_hypersteer.py.
_SCRIPT = Path(__file__).resolve()
_AX_PKG_PARENT = _SCRIPT.parents[2]
if str(_AX_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_AX_PKG_PARENT))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from axbench.utils.constants import CHAT_MODELS  # noqa: E402

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise SystemExit(
        "serve_mcp_multivec requires fastapi, uvicorn, pydantic. "
        "Install: pip install 'fastapi' 'uvicorn[standard]'\n"
    ) from e


class _ChatMessage(BaseModel):
    role: str
    content: str


class _ChatCompletionsRequest(BaseModel):
    model: str = "multivec-local"
    messages: list[_ChatMessage] = Field(default_factory=list)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 1.0
    stream: bool = False


def _messages_to_prompt(model_name: str, tokenizer, messages: list[dict[str, str]]) -> str:
    """Verbatim copy of serve_mcp_hypersteer._messages_to_prompt — same chat
    template handling, same MCP_ENABLE_THINKING env hook, same Gemma fallback."""
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


def _load_specs(spec_json: str) -> list[dict[str, Any]]:
    try:
        spec = json.loads(spec_json)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--vec-spec is not valid JSON: {e}") from e
    if not isinstance(spec, list) or not spec:
        raise SystemExit("--vec-spec must be a non-empty JSON array")
    out: list[dict[str, Any]] = []
    for i, item in enumerate(spec):
        if "path" not in item or "factor" not in item:
            raise SystemExit(f"--vec-spec[{i}] needs both 'path' and 'factor': {item!r}")
        path = Path(item["path"]).resolve()
        if not path.is_file():
            raise SystemExit(f"--vec-spec[{i}] path does not exist: {path}")
        out.append({"path": path, "factor": float(item["factor"])})
    return out


def _load_vectors(specs: list[dict[str, Any]], hidden_dim: int):
    """Load each .pt and validate shape. Returns list of dicts with 'vec',
    'factor', 'meta'."""
    loaded = []
    for s in specs:
        payload = torch.load(s["path"], map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "vec" in payload:
            vec = payload["vec"]
            meta = {k: v for k, v in payload.items() if k != "vec"}
        else:
            vec = payload
            meta = {}
        if not isinstance(vec, torch.Tensor):
            raise SystemExit(f"{s['path']}: 'vec' is not a tensor (got {type(vec)})")
        if vec.dim() != 1 or vec.shape[0] != hidden_dim:
            raise SystemExit(
                f"{s['path']}: vec shape {tuple(vec.shape)} mismatches hidden_dim={hidden_dim}"
            )
        loaded.append({"vec": vec.float(), "factor": s["factor"], "meta": meta, "path": s["path"]})
    return loaded


def _make_hook(combined_vec: torch.Tensor):
    """Returns a forward hook that adds ``combined_vec`` to the residual
    hidden state at every token position. Decoder layer outputs in HF Qwen3
    are tuples ``(hidden_states, ...)``; we add to ``out[0]`` and rebuild."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            add = combined_vec.to(device=hidden.device, dtype=hidden.dtype)
            new_hidden = hidden + add  # broadcast over [batch, seq, hidden]
            return (new_hidden,) + output[1:]
        # Bare-tensor case (some models return tensor directly).
        add = combined_vec.to(device=output.device, dtype=output.dtype)
        return output + add

    return hook


def _serve(
    vec_spec: str,
    host: str,
    port: int,
    device: str,
    model_name: str,
    layer: int,
    use_bf16: bool,
    default_max_tokens: int,
) -> None:
    specs = _load_specs(vec_spec)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, model_max_length=8192)
    tokenizer.padding_side = "left"
    if tokenizer.unk_token is None and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    dtype = torch.bfloat16 if use_bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()

    hidden_dim = int(model.config.hidden_size)
    loaded = _load_vectors(specs, hidden_dim)

    # Pre-combine into a single [hidden_dim] vector since this is a linear sum
    # — saves a few flops vs. summing inside the hook on every forward.
    combined = torch.zeros(hidden_dim, dtype=torch.float32)
    for item in loaded:
        combined = combined + item["factor"] * item["vec"]

    print(f"[multivec] loaded {len(loaded)} vec(s) at layer {layer}:")
    for item in loaded:
        m = item["meta"]
        cid = m.get("concept_id", "?")
        ctx = (m.get("concept_text") or "")[:60]
        src = Path(m.get("source_dir", "?")).name
        print(f"  - {item['path'].name}  factor={item['factor']:+.3f}  cid={cid}  src={src}  '{ctx}'")
    print(f"[multivec] combined vec: shape={tuple(combined.shape)} norm={combined.norm().item():.4f}")

    # Resolve target layer module. Qwen3 / Llama / Gemma all expose
    # `model.model.layers[L]`. Fall back to `model.layers` for bare LM heads.
    try:
        layer_mod = model.model.layers[layer]
    except AttributeError:
        layer_mod = model.layers[layer]  # type: ignore[attr-defined]
    handle = layer_mod.register_forward_hook(_make_hook(combined))

    is_chat = model_name in CHAT_MODELS
    if not is_chat:
        print(
            f"[multivec] WARNING: model_name={model_name!r} not in CHAT_MODELS; "
            f"chat template will be bypassed (raw last-message string used). "
            f"This is the same silent-bypass bug noted in feedback memory.",
            flush=True,
        )

    app = FastAPI(title="MultiVec local policy", version="0.1.0")

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": "local-multivec", "object": "model"}]}

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "n_vecs": str(len(loaded)), "layer": str(layer)}

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
        gen_kwargs = {
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
        f"\n[serve_mcp_multivec] OpenAI API base (vf-eval -b {base} -k EMPTY):\n  {base}\n",
        flush=True,
    )
    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        handle.remove()


def main() -> None:
    p = argparse.ArgumentParser(description="MultiVec OpenAI-compatible local server")
    p.add_argument("--vec-spec", required=True,
                   help='JSON array, e.g. \'[{"path":"/root/v17.pt","factor":0.5}]\'')
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-name", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--no-bf16", action="store_true", help="Disable bf16 (default on)")
    args = p.parse_args()

    default_max_tokens = int(os.environ.get("HYPERSTEER_MAX_TOKENS", "1024"))

    _serve(
        vec_spec=args.vec_spec,
        host=args.host,
        port=args.port,
        device=args.device,
        model_name=args.model_name,
        layer=int(args.layer),
        use_bf16=not args.no_bf16,
        default_max_tokens=default_max_tokens,
    )


if __name__ == "__main__":
    main()
