#!/usr/bin/env python3
"""
Image-capable vLLM HTTP server compatible with api_layer.vllm_api legacy endpoints.

Endpoints
---------
- POST /generate/text              (plain text)
- POST /generate/json              (guided JSON)
- POST /generate/text_with_image   (plain text with image)
- POST /generate/json_with_image   (guided JSON with image)
- POST /generate/text_batch        (batch text; images not supported)
- GET  /health

Intended to be used with VLLMServerAPI; when an 'image' payload is present,
the client will hit the *_with_image routes.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from PIL import Image  # type: ignore
 # type: ignore

LOGGER = logging.getLogger("vllm_server_image")

app = FastAPI(title="Image vLLM Server", version="0.1.0")


# ---------- CLI configuration ----------
@dataclass
class ServerConfig:
    model: str
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    gpu_memory_utilization: Optional[float] = None
    max_model_len: Optional[int] = None
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"
    allowed_local_media_paths: Optional[List[str]] = None


CONFIG: Optional[ServerConfig] = None
LLM_INSTANCE: Optional[LLM] = None


# ---------- Request models ----------
class BaseGenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    image: Optional[str] = None
    model: Optional[str] = None


class JSONGenerateRequest(BaseGenerateRequest):
    json_schema: Dict[str, Any]
    temperature: float = 0.0
    top_p: float = Field(0.95, gt=0.0, le=1.0)


class BatchTextGenerateRequest(BaseModel):
    prompts: List[str]
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    image: Optional[str] = None
    model: Optional[str] = None


class BatchTextGenerateWithImagesRequest(BaseModel):
    prompts: List[str]
    images: List[Optional[str]]  # one image (optional) per prompt
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    model: Optional[str] = None


class RESTGenerateRequest(BaseModel):
    prompt: Union[str, List[str]]
    max_tokens: int = Field(512, ge=1)
    temperature: float = 0.7
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    guided_decoding: Optional[Dict[str, Any]] = None
    image: Optional[str] = None


# ---------- Helpers ----------
def _ensure_llm() -> LLM:
    if LLM_INSTANCE is None:
        raise RuntimeError("LLM not initialised. Did you start the server correctly?")
    return LLM_INSTANCE


def _decode_image_reference(image: Optional[str]) -> Optional[str]:
    """
    Normalize an image reference to a form vLLM chat() can consume without
    requiring --allowed-local-media-path. Prefer inline data URLs; accept http(s).
    For local filesystem paths (or file:// URIs), read and embed as data URLs.
    """
    if not image:
        return None
    try:
        if image.startswith("data:"):
            # Validate base64 payload minimally
            try:
                header, encoded = image.split(",", 1)
                _ = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                LOGGER.warning("Invalid data URL image: %s", exc)
                return None
            return image
        if image.startswith("http://") or image.startswith("https://"):
            return image
        # Local path/URI ⇒ embed as data URL (avoid --allowed-local-media-path)
        p = Path(image[7:]) if image.startswith("file://") else Path(image)
        if not p.is_absolute():
            p = p.resolve()
        if not (p.exists() and p.is_file()):
            LOGGER.warning("Image file not found: %s", p)
            return None
        try:
            data = p.read_bytes()
            # Lazy import to avoid a top-level dependency; fallback to PNG
            import mimetypes  # noqa: WPS433

            mime = mimetypes.guess_type(str(p))[0] or "image/png"
            encoded = base64.b64encode(data).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except Exception as exc:
            LOGGER.warning("Failed to embed local image %s: %s", p, exc)
            return None
    except Exception as exc:
        LOGGER.warning("Failed to normalize image reference: %s", exc)
        return None


def _run_llm_generate(prompt: str, sampling: SamplingParams) -> str:
    llm = _ensure_llm()
    outputs = llm.generate([prompt], sampling)
    if not outputs:
        raise RuntimeError("vLLM returned no outputs.")
    return outputs[0].outputs[0].text


def _run_llm_chat_with_image(prompt: str, sampling: SamplingParams, image_url: str) -> str:
    """Use vLLM chat() with multimodal content for a single prompt+image."""
    llm = _ensure_llm()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    LOGGER.info("Submitting single image via chat() (vision path)")
    outs = llm.chat(messages=[msgs], sampling_params=[sampling])
    if not outs:
        raise RuntimeError("vLLM chat returned no outputs.")
    return outs[0].outputs[0].text


def _run_llm_generate_batch(prompts: List[str], sampling: SamplingParams) -> List[str]:
    llm = _ensure_llm()
    if not prompts:
        return []
    outputs = llm.generate(list(prompts), sampling)
    texts: List[str] = []
    for out in outputs:
        if not out.outputs:
            texts.append("")
        else:
            texts.append(out.outputs[0].text)
    return texts


def _run_llm_chat_batch_with_images(
    prompts: List[str], sampling: SamplingParams, images: List[Optional[str]]
) -> List[str]:
    """Use vLLM chat() for a batch of prompt+optional-image items."""
    llm = _ensure_llm()
    if not prompts:
        return []
    messages_batch: List[List[Dict[str, Any]]] = []
    for prompt, img in zip(prompts, images):
        content = [{"type": "text", "text": prompt}]
        if img is not None:
            content.append({"type": "image_url", "image_url": {"url": img}})
        messages_batch.append([{ "role": "user", "content": content }])
    LOGGER.info("Submitting chat batch with images | prompts=%d", len(messages_batch))
    outs = llm.chat(
        messages=messages_batch,
        sampling_params=[sampling for _ in messages_batch],
    )
    texts: List[str] = []
    for out in outs:
        if not out.outputs:
            texts.append("")
        else:
            texts.append(out.outputs[0].text)
    return texts


# ---------- Handlers ----------
async def _handle_text(req: BaseGenerateRequest) -> JSONResponse:
    if req.image:
        LOGGER.warning("/generate/text received unexpected image payload; ignoring for this route. Use /generate/text_with_image")
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


async def _handle_json(req: JSONGenerateRequest) -> JSONResponse:
    if req.image:
        LOGGER.warning("/generate/json received unexpected image payload; ignoring for this route. Use /generate/json_with_image")
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
        LOGGER.error("Raw model output: %r", text)
        return JSONResponse({"ok": False, "error_code": "JSON_PARSE_FAILED", "text": text})
    return JSONResponse({"ok": True, "json": parsed, "text": text})


async def _handle_text_with_image(req: BaseGenerateRequest) -> JSONResponse:
    image_url = _decode_image_reference(req.image)
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    try:
        if image_url is not None:
            text = await run_in_threadpool(_run_llm_chat_with_image, req.prompt, sampling, image_url)
        else:
            text = await run_in_threadpool(_run_llm_generate, req.prompt, sampling)
        return JSONResponse({"ok": True, "text": text})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Text+image generation failed: prompt_preview=%r", req.prompt[:120])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _handle_json_with_image(req: JSONGenerateRequest) -> JSONResponse:
    image_url = _decode_image_reference(req.image)
    guided = GuidedDecodingParams(json=req.json_schema)
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        guided_decoding=guided,
    )
    try:
        if image_url is not None:
            text = await run_in_threadpool(_run_llm_chat_with_image, req.prompt, sampling, image_url)
        else:
            text = await run_in_threadpool(_run_llm_generate, req.prompt, sampling)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("JSON+image generation failed: prompt_preview=%r", req.prompt[:120])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    parsed: Optional[Dict[str, Any]] = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        LOGGER.error("JSON+image parse failed under guided decoding. Preview=%r", text[:160])
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        LOGGER.error("Raw model output: %r", text)
        return JSONResponse({"ok": False, "error_code": "JSON_PARSE_FAILED", "text": text})
    return JSONResponse({"ok": True, "json": parsed, "text": text})


async def _handle_text_batch(req: BatchTextGenerateRequest) -> JSONResponse:
    # Batch endpoint does not support images for simplicity
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    try:
        texts = await run_in_threadpool(_run_llm_generate_batch, req.prompts or [], sampling)
        return JSONResponse({"ok": True, "text": texts})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Batch text generation failed: prompts=%d", len(req.prompts) if req.prompts else 0)
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------- Routes ----------
@app.post("/generate/text")
async def generate_text(req: BaseGenerateRequest) -> JSONResponse:
    return await _handle_text(req)


@app.post("/generate/json")
async def generate_json(req: JSONGenerateRequest) -> JSONResponse:
    return await _handle_json(req)


@app.post("/generate/text_with_image")
async def generate_text_with_image(req: BaseGenerateRequest) -> JSONResponse:
    return await _handle_text_with_image(req)


@app.post("/generate/json_with_image")
async def generate_json_with_image(req: JSONGenerateRequest) -> JSONResponse:
    return await _handle_json_with_image(req)


@app.post("/generate/text_batch")
async def generate_text_batch(req: BatchTextGenerateRequest) -> JSONResponse:
    return await _handle_text_batch(req)


@app.post("/generate/text_batch_with_image")
async def generate_text_batch_with_image(req: BatchTextGenerateWithImagesRequest) -> JSONResponse:
    if len(req.prompts or []) != len(req.images or []):
        raise HTTPException(status_code=400, detail="Length of prompts must equal length of images")
    # Decode images in order
    decoded: List[Optional[str]] = []
    for ref in req.images or []:
        decoded.append(_decode_image_reference(ref))
    sampling = SamplingParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    try:
        texts = await run_in_threadpool(
            _run_llm_chat_batch_with_images, req.prompts or [], sampling, decoded
        )
        return JSONResponse({"ok": True, "text": texts})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception(
            "Batch text+image generation failed: prompts=%d", len(req.prompts) if req.prompts else 0
        )
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "model": CONFIG.model if CONFIG else "unknown"}


@app.post("/generate")
async def generate_rest(req: RESTGenerateRequest) -> JSONResponse:
    """
    REST-compatible endpoint: single or batch prompts, optional image and JSON guidance.
    Returns {"text": str} or {"text": [str, ...]}.
    """
    image_url = _decode_image_reference(req.image)
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
            # Batch path: if an image is provided, reuse it for all prompts via chat(); else plain generate()
            if image_url is not None:
                imgs = [image_url for _ in req.prompt]
                texts = await run_in_threadpool(_run_llm_chat_batch_with_images, list(req.prompt), sampling, imgs)
            else:
                texts = await run_in_threadpool(_run_llm_generate_batch, list(req.prompt), sampling)
            return JSONResponse({"ok": True, "text": texts})
        # Single prompt
        if image_url is not None:
            text = await run_in_threadpool(_run_llm_chat_with_image, str(req.prompt), sampling, image_url)
        else:
            text = await run_in_threadpool(_run_llm_generate, str(req.prompt), sampling)
        # Guided decoding parse check (if any)
        if guided is not None:
            try:
                _ = json.loads(text)
            except json.JSONDecodeError:
                LOGGER.error("REST /generate JSON parse failed (image server). Preview=%r", text[:160])
                try:
                    LOGGER.error("Request body: %s", req.dict())
                except Exception:
                    pass
                LOGGER.error("Raw model output: %r", text)
                return JSONResponse({"ok": False, "error_code": "JSON_PARSE_FAILED", "text": text})
        return JSONResponse({"ok": True, "text": text})
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("REST /generate failed: prompts=%s", type(req.prompt).__name__)
        try:
            LOGGER.error("Request body: %s", req.dict())
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------- Startup ----------
def parse_args(argv: Optional[List[str]] = None) -> ServerConfig:
    parser = argparse.ArgumentParser(description="vLLM server with basic image support.")
    parser.add_argument("--model", required=True, help="Model name or path to load.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism degree.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype (e.g. float16, bfloat16).")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="VRAM utilisation fraction.")
    parser.add_argument("--max-model-len", type=int, default=None, help="Override maximum model length.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8001, help="TCP port to bind.")
    parser.add_argument("--log-level", default="info", help="Logging level.")
    parser.add_argument(
        "--allowed-local-media-path",
        dest="allowed_local_media_paths",
        action="append",
        default=None,
        help="Allow vLLM chat() to load local file:// media from this path (may be specified multiple times).",
    )
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
        allowed_local_media_paths=args.allowed_local_media_paths,
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
    # Permit local file:// media for chat multimodal when configured.
    # vLLM ModelConfig expects a single string for `allowed_local_media_path`.
    if config.allowed_local_media_paths:
        try:
            norm_paths: List[str] = []
            for p in config.allowed_local_media_paths:
                try:
                    norm_paths.append(str(Path(p).resolve()))
                except Exception:
                    norm_paths.append(str(p))
            value: str = norm_paths[0] if len(norm_paths) == 1 else ",".join(norm_paths)
            llm_kwargs["allowed_local_media_path"] = value
            LOGGER.info("Allowing local media path(s): %s", value)
        except Exception:
            LOGGER.warning(
                "Failed to apply allowed_local_media_path; proceeding without local media access.")
    return LLM(**llm_kwargs)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="% (asctime)s | % (levelname)s | % (name)s | % (message)s".replace("% ", "%"),
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
