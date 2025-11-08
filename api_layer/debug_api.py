from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .base import BaseLLMAPI

PLACEHOLDER_TEXT = "[debug-api] response not generated; replace with real model output."


class DebugLLMAPI(BaseLLMAPI):
    """
    Lightweight mock LLM API for pipeline debugging without external dependencies.

    - Text generation returns a deterministic placeholder string.
    - JSON generation picks a valid value at random from enum constraints (or defaults).
    - If an image path/data URL is provided, the API attempts to load it for parity with real calls.
    """

    def __init__(self, *, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random()

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        image: Optional[str] = None,
    ) -> str:
        if image:
            self._touch_image_reference(image)
        return PLACEHOLDER_TEXT

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Mapping[str, Any],
        max_tokens: int = 512,
        temperature: float = 0.0,
        image: Optional[str] = None,
    ) -> Dict[str, Any]:
        if image:
            self._touch_image_reference(image)
        return self._sample_from_schema(schema)

    def generate_text_batch(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **_: Any,
    ) -> list[str]:
        """Mock batch API: returns placeholder text per prompt; ignores images."""
        return [PLACEHOLDER_TEXT for _ in (prompts or [])]

    # ---------- Helpers ----------
    def _sample_from_schema(self, schema: Mapping[str, Any]) -> Dict[str, Any]:
        if schema.get("type") != "object":
            raise ValueError("DebugLLMAPI expects an object schema for generate_json.")
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties.keys()))
        result: Dict[str, Any] = {}
        for key in required:
            prop_schema = properties.get(key, {})
            result[key] = self._sample_value(prop_schema)
        return result

    def _sample_value(self, prop_schema: Mapping[str, Any]) -> Any:
        if "enum" in prop_schema:
            return self._rng.choice(list(prop_schema["enum"]))
        prop_type = prop_schema.get("type")
        if prop_type == "boolean":
            return self._rng.choice([True, False])
        if prop_type == "integer":
            if "minimum" in prop_schema and "maximum" in prop_schema:
                return self._rng.randint(prop_schema["minimum"], prop_schema["maximum"])
            return self._rng.randint(0, 10)
        if prop_type == "number":
            minimum = prop_schema.get("minimum", 0.0)
            maximum = prop_schema.get("maximum", minimum + 10.0)
            return minimum if minimum == maximum else self._rng.uniform(minimum, maximum)
        if prop_type == "string":
            enum = prop_schema.get("enum")
            if enum:
                return self._rng.choice(list(enum))
            return f"debug-{self._rng.randint(0, 9999)}"
        if prop_type == "object":
            return self._sample_from_schema(prop_schema)
        if prop_type == "array":
            items = prop_schema.get("items", {})
            min_items = prop_schema.get("minItems", 1)
            max_items = prop_schema.get("maxItems", min_items)
            length = min_items if min_items == max_items else self._rng.randint(min_items, max_items)
            return [self._sample_value(items) for _ in range(length)]
        return f"debug-{prop_type or 'value'}"

    def _touch_image_reference(self, image: str) -> None:
        if image.startswith("data:"):
            return
        if image.startswith("http://") or image.startswith("https://"):
            return
        path = Path(image[7:]) if image.startswith("file://") else Path(image)
        try:
            if path.exists() and path.is_file():
                _ = path.read_bytes()
        except Exception:
            pass
