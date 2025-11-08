from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from problem_layer import ProblemDefinition

from .agent import CollaborativeAgent


class ThinkingAgent(CollaborativeAgent):
    """
    Agent that separates "thinking" (free-form text) from final JSON decision.
    The protocol orchestrates thinking (optionally batched) and then calls the
    sequential decide_from_thought stage to keep Gauss–Seidel updates.
    """

    def prepare_thinking_prompt(
        self,
        problem: ProblemDefinition,
        context: Mapping[str, Any],
        images: Optional[Sequence[str]] = None,
    ) -> str:
        base_prompt = self._build_decision_prompt(problem, context)
        thinking_prompt = self._build_thinking_prompt(base_prompt)
        # Log prompt for traceability; no API call here
        self.prompt_history.append(thinking_prompt)
        return thinking_prompt

    def decide_from_thought(
        self,
        problem: ProblemDefinition,
        context: Mapping[str, Any],
        thought: str,
        images: Optional[Sequence[str]] = None,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        base_prompt = self._build_decision_prompt(problem, context)
        decision_prompt = self._build_json_prompt_with_thinking(base_prompt, thought or "")
        self.prompt_history.append(decision_prompt)
        self._last_prompt = decision_prompt
        image_payload, image_source, image_error = self._prepare_image_payload(images or ())
        try:
            response = self.api.generate_json(
                decision_prompt,
                schema=problem.agent_schema(self.agent_id),
                max_tokens=max_tokens,
                temperature=temperature,
                image=image_payload,
            )
        except Exception as exc:  # pragma: no cover - fall back path mirrors CollaborativeAgent
            response = self._fallback_decision(problem)
            self.decision_history.append(
                {
                    "source": "fallback_thinking",
                    "mode": "json_only",
                    "decision": response,
                    "context": dict(context),
                    "attempt_errors": [
                        {"attempt": 1, "error": str(exc), "mode": "json_only"}
                    ],
                    "decision_prompt": decision_prompt,
                    "thinking_prompt": None,
                    "thought": thought,
                    "image_source": image_source,
                    "image_error": image_error,
                }
            )
            return response

        # Log successful decision
        self.decision_history.append(
            {
                "source": "llm",
                "mode": "json_only",
                "decision": response,
                "context": dict(context),
                "attempt": 1,
                "decision_prompt": decision_prompt,
                "thinking_prompt": None,
                "thought": thought,
                "image_source": image_source,
                "image_error": image_error,
            }
        )
        return response

