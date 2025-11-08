from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import requests
from requests import HTTPError

from .base import BaseLLMAPI


class VLLMServerAPI(BaseLLMAPI):
    """
    Client for interacting with vLLM HTTP servers.

    Supported compat modes:
    - ``rest``: ``python -m vllm.entrypoints.api_server`` (POST /generate)
    - ``legacy``: custom servers that expose /generate/text and /generate/json
    - ``openai``: OpenAI-compatible server (not implemented yet)

    ``compat_mode="auto"`` attempts the available strategies in sequence and
    caches the first one that succeeds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout: int = 1000,
        text_endpoint: Optional[str] = None,
        json_endpoint: Optional[str] = None,
        session: Optional[requests.Session] = None,
        model: Optional[str] = None,
        compat_mode: str = "auto",
    ) -> None:
        if compat_mode != "auto" and compat_mode not in {"rest", "legacy", "openai"}:
            raise ValueError(f"Unsupported compat_mode {compat_mode!r}.")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._custom_text_endpoint = text_endpoint
        self._custom_json_endpoint = json_endpoint
        self.session = session or requests.Session()
        self.model = model
        self.compat_mode = compat_mode
        self._resolved_mode: Optional[str] = None
        self._warned_image_drop = False

    # ---------- Public API ----------
    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        image: Optional[str] = None,
    ) -> str:
        last_error: Optional[Exception] = None
        for mode in self._candidate_modes(image=image is not None):
            try:
                result = self._generate_text_mode(
                    mode,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    image=image,
                )
            except HTTPError as exc:
                if self._should_ignore_http_error(exc):
                    last_error = exc
                    continue
                raise
            except NotImplementedError:
                last_error = None
                continue
            if result is not None:
                self._resolved_mode = mode
                return result
        if last_error is not None:
            raise last_error
        raise RuntimeError("No compatible vLLM text generation endpoint responded successfully.")

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for mode in self._candidate_modes(image=image is not None):
            try:
                result = self._generate_json_mode(
                    mode,
                    prompt=prompt,
                    schema=schema,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    image=image,
                )
            except HTTPError as exc:
                if self._should_ignore_http_error(exc):
                    last_error = exc
                    continue
                raise
            except NotImplementedError:
                last_error = None
                continue
            if result is not None:
                self._resolved_mode = mode
                return result
        if last_error is not None:
            raise last_error
        raise RuntimeError("No compatible vLLM JSON generation endpoint responded successfully.")

    def generate_text_with_image(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        image: str,
    ) -> str:
        return self.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            image=image,
        )

    def generate_json_with_image(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        image: str,
    ) -> Dict[str, Any]:
        return self.generate_json(
            prompt,
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            image=image,
        )

    def generate_text_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        image: Optional[str] = None,
        images: Optional[list[Optional[str]]] = None,
    ) -> list[str]:
        """Batch text generation for multiple prompts (REST mode only)."""
        if not prompts:
            return []
        # If images are provided per prompt, use image-capable batch endpoint
        if images is not None:
            payload_img: Dict[str, Any] = {
                "prompts": list(prompts),
                "images": list(images),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            try:
                endpoint = self._custom_text_endpoint or "/generate/text_batch_with_image"
                response = self._post(endpoint, payload_img)
            except HTTPError:
                # Treat any HTTP error from the image batch endpoint as unsupported
                # and signal the caller to fall back to a sequential loop.
                raise NotImplementedError
            if not isinstance(response, dict) or "text" not in response:
                raise ValueError("Image batch endpoint returned unexpected format.")
            texts = response.get("text")
            if not isinstance(texts, list) or len(texts) != len(prompts):
                raise ValueError("Image batch response length mismatch.")
            return [str(t) for t in texts]

        # Attempt REST multi-prompt endpoint first (no images)
        payload: Dict[str, Any] = {
            "prompt": list(prompts),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        try:
            response = self._post(self._custom_text_endpoint or "/generate", payload)
        except HTTPError as exc:
            if self._should_ignore_http_error(exc):
                # Fall back to legacy custom batch endpoint
                try:
                    legacy_payload: Dict[str, Any] = {
                        "prompts": list(prompts),
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                    legacy_endpoint = self._custom_text_endpoint or "/generate/text_batch"
                    legacy_resp = self._post(legacy_endpoint, legacy_payload)
                    if not isinstance(legacy_resp, dict) or not isinstance(legacy_resp.get("text"), list):
                        raise ValueError("Legacy batch endpoint returned unexpected format.")
                    texts = legacy_resp["text"]
                    if len(texts) != len(prompts):
                        raise ValueError("Legacy batch response length mismatch.")
                    # Remember legacy mode worked for batching
                    self._resolved_mode = "legacy"
                    return [str(t) for t in texts]
                except Exception:
                    # Signal unsupported batching to caller
                    raise NotImplementedError
            raise
        if not isinstance(response, dict):
            raise ValueError("Unexpected response type from REST vLLM batch endpoint.")
        texts = response.get("text")
        if not isinstance(texts, list) or len(texts) != len(prompts):
            raise ValueError("REST batch response missing text list matching prompt count.")
        outputs: list[str] = []
        for prompt, completion in zip(prompts, texts):
            c = str(completion)
            if c.startswith(prompt):
                c = c[len(prompt) :]
            outputs.append(c)
        return outputs

    # ---------- Mode dispatch ----------
    def _candidate_modes(self, *, image: bool) -> Tuple[str, ...]:
        if self._resolved_mode:
            return (self._resolved_mode,)
        if self.compat_mode == "auto":
            if image:
                # REST mode cannot honour multimodal payloads; try older endpoints first.
                return ("legacy", "rest")
            return ("rest", "legacy")
        return (self.compat_mode,)

    @staticmethod
    def _should_ignore_http_error(exc: HTTPError) -> bool:
        response = exc.response
        if response is None:
            return False
        return response.status_code in (404, 405)

    def _generate_text_mode(
        self,
        mode: str,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> str:
        if mode == "legacy":
            return self._legacy_generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                image=image,
            )
        if mode == "rest":
            return self._rest_generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                image=image,
            )
        if mode == "openai":
            raise NotImplementedError
        raise ValueError(f"Unsupported mode {mode!r}")

    def _generate_json_mode(
        self,
        mode: str,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> Dict[str, Any]:
        if mode == "legacy":
            return self._legacy_generate_json(
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                image=image,
            )
        if mode == "rest":
            return self._rest_generate_json(
                prompt=prompt,
                schema=schema,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                image=image,
            )
        if mode == "openai":
            raise NotImplementedError
        raise ValueError(f"Unsupported mode {mode!r}")

    # ---------- Legacy endpoints (/generate/text, /generate/json) ----------
    def _legacy_generate_text(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> str:
        # Use image-capable endpoint when an image payload is provided
        endpoint = self._custom_text_endpoint or (
            "/generate/text_with_image" if image is not None else "/generate/text"
        )
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.model:
            payload["model"] = self.model
        if image is not None:
            payload["image"] = image
        response = self._post(endpoint, payload)
        if isinstance(response, dict):
            if "text" in response:
                return str(response["text"])
            if "completion" in response:
                return str(response["completion"])
        raise ValueError("Unexpected response format from legacy vLLM text endpoint.")

    def _legacy_generate_json(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> Dict[str, Any]:
        # Use image-capable endpoint when an image payload is provided
        endpoint = self._custom_json_endpoint or (
            "/generate/json_with_image" if image is not None else "/generate/json"
        )
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "json_schema": schema,
        }
        if self.model:
            payload["model"] = self.model
        if image is not None:
            payload["image"] = image
        response = self._post(endpoint, payload)
        if isinstance(response, dict):
            if "json" in response and isinstance(response["json"], dict):
                return response["json"]
            if "text" in response:
                return self.parse_json_object(str(response["text"]))
        raise ValueError("Unexpected response format from legacy vLLM JSON endpoint.")

    # ---------- REST endpoint (/generate) ----------
    def _rest_generate_text(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> str:
        payload = self._rest_common_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if image is not None:
            payload["image"] = image
        endpoint = self._custom_text_endpoint or (
            "/generate/text_with_image" if image is not None else "/generate"
        )
        response = self._post(endpoint, payload)
        return self._rest_extract_text(response, prompt)

    def _rest_generate_json(
        self,
        *,
        prompt: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        temperature: float,
        top_p: float,
        image: Optional[str],
    ) -> Dict[str, Any]:
        payload = self._rest_common_payload(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        payload["guided_decoding"] = {"json": schema}
        if image is not None:
            payload["image"] = image
        endpoint = self._custom_json_endpoint or (
            "/generate/json_with_image" if image is not None else "/generate"
        )
        response = self._post(endpoint, payload)
        completion = self._rest_extract_text(response, prompt)
        return self.parse_json_object(completion)

    def _rest_common_payload(
        self,
        *,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        return payload

    @staticmethod
    def _rest_extract_text(response: Any, prompt: str) -> str:
        if not isinstance(response, dict):
            raise ValueError("Unexpected response type from REST vLLM endpoint.")
        texts = response.get("text")
        if isinstance(texts, list) and texts:
            completion = texts[0]
        elif isinstance(texts, str):
            completion = texts
        else:
            raise ValueError("REST response missing text field.")
        completion = str(completion)
        if completion.startswith(prompt):
            return completion[len(prompt) :]
        return completion

    # ---------- HTTP helper ----------
    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}{endpoint}"
        response = self.session.post(url, json=dict(payload), timeout=self.timeout)
        response.raise_for_status()
        return response.json()


'''

/data/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/3d729a084f14c9502775d59d95c71385293f5518
google/gemma-3-27b-it
/data/huggingface/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/9fe72e5d5c4e325af1653d160b4212c5dc49a299/
/data/huggingface/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/33101ce6ccc08fa6249c10a543ebfcac65173393

python api_layer/vllm_server.py \
  --model google/gemma-3-27b-it\
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000


  python api_layer/vllm_server.py \
  --model /data/huggingface/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/3d729a084f14c9502775d59d95c71385293f5518\
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000

'''
