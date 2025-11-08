#!/usr/bin/env python3
"""
Lightweight HTTP server that exposes a minimal vLLM API compatible with
``apis.vllm_api.VLLMServerAPI`` legacy endpoints.

Endpoints
---------
- POST /generate/text
- POST /generate/json
- POST /generate/text_with_image
- POST /generate/json_with_image

Each endpoint accepts a JSON body containing:
    prompt: str
    max_tokens: int (optional, default 512)
    temperature: float (optional, default 0.7 or 0.0 for JSON)
    top_p: float (optional, default 0.95)
    json_schema: mapping (required for the JSON endpoints)
    image: optional data URL / path (currently ignored with a warning)

The server instantiates a single vLLM ``LLM`` object and services requests by
calling ``llm.generate`` with the provided sampling parameters. JSON responses
make use of ``GuidedDecodingParams`` to enforce the supplied schema.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

LOGGER = logging.getLogger("vllm_server")

app = FastAPI(title="Custom vLLM Server", version="0.1.0")


# ---------- CLI configuration ----------
@dataclass
class ServerConfig:
    model: str
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


CONFIG: Optional[ServerConfig] = None
LLM_INSTANCE: Optional[LLM] = None


# ---------- Request models ----------
class BaseGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    image: Optional[str] = None
    model: Optional[str] = None  # accepted but ignored; kept for compatibility


class JSONGenerateRequest(BaseGenerateRequest):
    json_schema: Dict[str, Any]
    temperature: float = 0.0
    top_p: float = Field(0.95, gt=0.0, le=1.0)


class BatchTextGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    # Images are not supported in batch mode for simplicity; ignored if present
    image: Optional[str] = None
    model: Optional[str] = None


# REST-style request model for unified /generate endpoint
class RESTGenerateRequest(BaseModel):
    prompt: Union[str, List[str]]
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    guided_decoding: Optional[Dict[str, Any]] = None
    image: Optional[str] = None


# ---------- Helper utilities ----------
def _ensure_llm() -> LLM:
    if LLM_INSTANCE is None:
        raise RuntimeError("LLM not initialised. Did you start the server correctly?")
    return LLM_INSTANCE


def _decode_image_reference(image: Optional[str]) -> None:
    """
    Decode a data URL or base64 payload to verify it is well-formed.
    Currently we do not feed images into vLLM; this helper simply validates
    the payload and logs the first 16 bytes to aid debugging.
    """
    if not image:
        return
    if image.startswith("data:"):
        try:
            header, encoded = image.split(",", 1)
            raw = base64.b64decode(encoded, validate=True)
            LOGGER.debug("Received data URL image (%s bytes)", len(raw))
        except Exception as exc:
            LOGGER.warning("Failed to decode data URL image: %s", exc)
    else:
        LOGGER.debug("Ignoring non-inline image reference: %s", image[:64])


def _run_llm_generate(prompt: str, sampling_params: SamplingParams) -> str:
    llm = _ensure_llm()
    outputs = llm.generate([prompt], sampling_params)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs.")
    return outputs[0].outputs[0].text


def _run_llm_generate_batch(prompts: List[str], sampling_params: SamplingParams) -> List[str]:
    llm = _ensure_llm()
    if not prompts:
        return []
    outputs = llm.generate(list(prompts), sampling_params)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs.")
    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
        else:
            texts.append(out.outputs[0].text)
    return texts


async def _handle_text_request(req: BaseGenerateRequest) -> JSONResponse:
    _decode_image_reference(req.image)
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    try:
        text = await run_in_threadpool(_run_llm_generate, req.prompt, sampling)
        return JSONResponse({"ok": True, "text": text})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Text generation failed: prompt_preview=%r", req.prompt[:120])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _handle_json_request(req: JSONGenerateRequest) -> JSONResponse:
    _decode_image_reference(req.image)
    guided = GuidedDecodingParams(json=req.json_schema)
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        guided_decoding=guided,
    )
    try:
        text = await run_in_threadpool(_run_llm_generate, req.prompt, sampling)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("JSON generation failed: prompt_preview=%r", req.prompt[:120])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.error("JSON parse failed under guided decoding. Preview=%r", text[:160])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        # Also print full raw output for debugging
        LOGGER.error("Raw model output: %r", text)
        return JSONResponse({
            "ok": False,
            "error_code": "JSON_PARSE_FAILED",
            "text": text,
        })
    return JSONResponse({
        "ok": True,
        "json": parsed,
        "text": text,
    })


# ---------- Routes ----------
@app.post("/generate/text")
async def generate_text(req: BaseGenerateRequest) -> JSONResponse:
    return await _handle_text_request(req)


@app.post("/generate/json")
async def generate_json(req: JSONGenerateRequest) -> JSONResponse:
    return await _handle_json_request(req)


@app.post("/generate/text_with_image")
async def generate_text_with_image(req: BaseGenerateRequest) -> JSONResponse:
    return await _handle_text_request(req)


@app.post("/generate/json_with_image")
async def generate_json_with_image(req: JSONGenerateRequest) -> JSONResponse:
    return await _handle_json_request(req)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": CONFIG.model if CONFIG else "unknown"}


@app.post("/generate/text_batch")
async def generate_text_batch(req: BatchTextGenerateRequest) -> JSONResponse:
    """
    Batch text generation endpoint accepting a list of prompts and shared sampling params.
    Returns a JSON object with a "text" list matching the order of inputs.
    """
    _decode_image_reference(req.image)
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    try:
        texts = await run_in_threadpool(_run_llm_generate_batch, req.prompts, sampling)
        return JSONResponse({"ok": True, "text": texts})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Batch text generation failed: prompts=%d", len(req.prompts) if req.prompts else 0)
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate_rest(req: RESTGenerateRequest) -> JSONResponse:
    """
    REST-compatible endpoint used by VLLMServerAPI in 'rest' mode.
    - Accepts a single prompt (str) or a batch (list[str]) under the same route.
    - Optional guided_decoding: {"json": schema} enables JSON guidance.
    - Optional image is accepted but ignored by this server variant.
    """
    # Validate/ignore image for parity with client payloads
    _decode_image_reference(req.image)

    guided = None
    if isinstance(req.guided_decoding, dict):
        schema = req.guided_decoding.get("json")
        if isinstance(schema, dict):
            try:
                guided = GuidedDecodingParams(json=schema)
            except Exception:
                guided = None

    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        guided_decoding=guided,
    )
    try:
        if isinstance(req.prompt, list):
            texts = await run_in_threadpool(_run_llm_generate_batch, list(req.prompt), sampling)
            return JSONResponse({"ok": True, "text": texts})
        text = await run_in_threadpool(_run_llm_generate, str(req.prompt), sampling)
        # If guided decoding present, include parse status to aid debugging
        if guided is not None:
            try:
                _ = json.loads(text)
                return JSONResponse({"ok": True, "text": text})
            except json.JSONDecodeError:
                LOGGER.error("REST /generate JSON parse failed. Preview=%r", text[:160])
                try:
                    LOGGER.error("Request body: %s", req.dict())
                except Exception:
                    pass
                LOGGER.error("Raw model output: %r", text)
                return JSONResponse({"ok": False, "error_code": "JSON_PARSE_FAILED", "text": text})
        return JSONResponse({"ok": True, "text": text})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("/generate failed")
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------- Startup ----------
def parse_args(argv: Optional[List[str]] = None) -> ServerConfig:
    parser = argparse.ArgumentParser(description="Custom vLLM server.")
    parser.add_argument("--model", required=True, help="Model name or path to load.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism degree.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g. float16, bfloat16).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="VRAM utilisation fraction.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override maximum model length.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="TCP port to bind.")
    parser.add_argument("--log-level", default="info", help="Logging level.")
    args = parser.parse_args(argv)
    return ServerConfig(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


def initialise_llm(config: ServerConfig) -> LLM:
    LOGGER.info("Loading vLLM model: %s", config.model)
    llm_kwargs: Dict[str, Any] = {
        "model": config.model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "dtype": config.dtype,
        "enforce_eager": True,
    }
    if config.gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
    if config.max_model_len is not None:
        llm_kwargs["max_model_len"] = config.max_model_len
    return LLM(**llm_kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_args(argv)
    global CONFIG, LLM_INSTANCE  # pylint: disable=global-statement
    CONFIG = config
    try:
        LLM_INSTANCE = initialise_llm(config)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to load vLLM model.")
        raise SystemExit(1) from exc

    LOGGER.info("Server configuration: %s", asdict(config))
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )


if __name__ == "__main__":
    main()
