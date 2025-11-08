from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, List
from pathlib import Path
import base64
import mimetypes

from .base import BaseLLMAPI

try:
    import jsonschema  # type: ignore
except ImportError:
    jsonschema = None  # type: ignore


class OpenAIChatAPI(BaseLLMAPI):
    """
    Thin wrapper around the OpenAI Chat Completion endpoint that enforces JSON-only replies.

    Parameters
    ----------
    client:
        Optional pre-configured OpenAI client. If omitted, the module will attempt to
        import and use the global `openai` SDK.
    model:
        Model name to query, e.g. ``gpt-4o-mini`` or ``gpt-4.1``.
    system_prompt:
        Optional system message prepended to every request.
    """

    def __init__(
        self,
        *,
        client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        system_prompt: str = "You are a helpful assistant that follows instructions precisely.",
        api_key: Optional[str] = None,
    ) -> None:
        if client is None:
            try:
                import openai  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The openai package is required to use OpenAIChatAPI."
                ) from exc
            # Prefer new SDK client if available
            new_client: Optional[Any] = getattr(openai, "OpenAI", None)
            if new_client is not None:
                # openai>=1.0 style
                if api_key is not None:
                    client = new_client(api_key=api_key)
                else:
                    client = new_client()
            else:
                # Legacy SDK: set module-level api_key when provided
                if api_key is not None:
                    setattr(openai, "api_key", api_key)
                client = openai
        self.client = client
        self.model = model
        self.system_prompt = system_prompt

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        image: Optional[str] = None,
    ) -> str:
        response = self._create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image=image,
        )
        return self._extract_message_content(response)

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        response = self._create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            image=image,
        )
        content = self._extract_message_content(response)
        #print(f"Raw JSON content: {content}")
        payload = self.parse_json_object(content)
        #print(f"Parsed JSON payload: {payload}")
        _validate_against_schema(payload, schema)
        #print(f"Validated JSON payload: {payload}")
        return payload

    # ---------- Internal helpers ----------
    def _create_completion(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        response_format: Optional[Dict[str, Any]] = None,
        image: Optional[str] = None,
    ) -> Any:
        # Build user content; if image is provided, use multi-part content per Chat Completions
        # Normalise local file paths to data URLs for OpenAI vision models
        image = self._normalise_image_ref(image)
        user_content: Any
        if image:
            # Treat provided image string as a URL or data URL
            user_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Compatible with both openai<1 and openai>=1 clients.
        if hasattr(self.client, "chat") and hasattr(self.client.chat, "completions"):
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )

        if hasattr(self.client, "ChatCompletion"):
            return self.client.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )

        raise RuntimeError("Unsupported OpenAI client; expected chat completion interface.")

    # ---------- Image helpers ----------
    @staticmethod
    def _normalise_image_ref(image: Optional[str]) -> Optional[str]:
        """
        Convert a local file path (or file:// URI) to a data URL so that
        OpenAI Chat Completions can consume it as an image_url. URLs and
        existing data: URIs are passed through unchanged.
        """
        if image is None:
            return None
        ref = str(image).strip()
        if not ref:
            return None
        # Pass through URLs and existing data URIs
        if ref.startswith("data:") or ref.startswith("http://") or ref.startswith("https://"):
            return ref
        # Handle file:// URIs and bare filesystem paths
        path = Path(ref[7:]) if ref.startswith("file://") else Path(ref)
        try:
            if path.exists() and path.is_file():
                data = path.read_bytes()
                mime = mimetypes.guess_type(str(path))[0] or "image/png"
                encoded = base64.b64encode(data).decode("ascii")
                return f"data:{mime};base64,{encoded}"
        except Exception:
            # If anything goes wrong, fall back to returning the original reference
            return ref
        # Unknown format or missing file: return original (may still be a URL)
        return ref

    # ---------- Batch helpers ----------
    def generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        images: Optional[List[Optional[str]]] = None,
    ) -> List[str]:
        """
        Batched text generation implemented as a simple loop for OpenAI Chat API
        to preserve interface compatibility with vLLM clients. If images are
        provided, each prompt may carry its own image reference.
        """
        if not prompts:
            return []
        if images is not None and len(images) != len(prompts):
            raise ValueError("Length of images must match prompts length.")
        outputs: List[str] = []
        for i, prompt in enumerate(prompts):
            img = images[i] if images is not None else None
            text = self.generate_text(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                image=img,
            )
            outputs.append(text)
        print(outputs[0])
        return outputs

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("No choices returned by OpenAI response.")
            return choices[0]["message"]["content"].strip()

        if hasattr(response, "choices"):
            choices = response.choices
            if not choices:
                raise ValueError("No choices returned by OpenAI response.")
            message = getattr(choices[0], "message", None)
            if message and hasattr(message, "content"):
                return message.content.strip()
            if isinstance(choices[0], dict):
                return choices[0]["message"]["content"].strip()

        raise ValueError("Unrecognized response format from OpenAI client.")


def _validate_against_schema(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    if jsonschema is None:
        return
    try:
        jsonschema.validate(payload, schema)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Payload failed JSON schema validation: {exc}") from exc
