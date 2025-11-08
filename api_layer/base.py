from __future__ import annotations

import json
from typing import Any, Dict, Mapping


class BaseLLMAPI:
    """
    Minimal interface expected by CollaborativeAgent.
    Concrete subclasses must provide text and JSON generation helpers.
    """

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    # ---------- Optional batch helper ----------
    def generate_text_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[str]:  # pragma: no cover
        """
        Generate text for multiple prompts. Implementations may raise
        NotImplementedError if batch requests are not supported.
        """
        raise NotImplementedError

    @staticmethod
    def parse_json_object(text: str) -> Dict[str, Any]:
        """
        Extract a JSON object payload from a model response.
        Subclasses can reuse this helper before applying schema validation.
        """
        text = text.strip()
        if not text:
            raise ValueError("Empty response; cannot parse JSON.")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            start = text.find("{")
            end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("No JSON object found in model output.") from exc
            return json.loads(text[start : end + 1])
