from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Literal, Tuple

from api_layer.base import BaseLLMAPI
from problem_layer import ProblemDefinition


class CollaborativeAgent:
    """
    Agent wrapper that stores metadata, neighbouring decisions, and interacts with an LLM API.
    The agent supports multiple generation modes (JSON-only or think-then-JSON) and retries
    before falling back to heuristic choices, always striving to produce schema-compliant output.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        instruction: str,
        api: "BaseLLMAPI",
        decision_schema: Dict[str, Any],
        *,
        mode: Literal["json_only", "thinking"] = "json_only",
        max_retries: int = 2,
    ) -> None:
        self.agent_id = agent_id
        self.name = name
        self.instruction = instruction.strip()
        self.api = api
        self.decision_schema = decision_schema
        if mode not in {"json_only", "thinking"}:
            raise ValueError(f"Unsupported decision mode {mode!r}.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        self.mode = mode
        self.max_retries = int(max_retries)

        self.neighbours: Set[str] = set()
        self.neighbour_decisions: Dict[str, Dict[str, Any]] = {}
        self.decision_history: List[Dict[str, Any]] = []
        self.prompt_history: List[str] = []
        self.thinking_history: List[str] = []
        self._last_prompt: Optional[str] = None

    # ---------- Neighbour management ----------
    def add_neighbour(self, neighbour_id: str) -> None:
        if neighbour_id != self.agent_id:
            self.neighbours.add(neighbour_id)

    def set_neighbours(self, neighbour_ids: Sequence[str]) -> None:
        self.neighbours = {nid for nid in neighbour_ids if nid != self.agent_id}

    def notify_neighbour_decision(
        self,
        neighbour_id: str,
        decision: Mapping[str, Any],
        round_index: int,
    ) -> None:
        if neighbour_id not in self.neighbours:
            return
        self.neighbour_decisions[neighbour_id] = {
            "round": round_index,
            "decision": dict(decision),
        }

    # ---------- Decision ----------
    def decide(
        self,
        problem: ProblemDefinition,
        context: Optional[Mapping[str, Any]] = None,
        *,
        images: Optional[Sequence[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        context = dict(context or {})
        image_refs: Tuple[str, ...] = tuple(images or ())
        base_prompt = self._build_decision_prompt(problem, context)
        attempt_errors: List[Dict[str, Any]] = []
        total_attempts = self.max_retries + 1
        image_payload, image_source, image_error = self._prepare_image_payload(image_refs)
        if image_error:
            attempt_errors.append(
                {
                    "attempt": 0,
                    "error": f"image_prepare:{image_error}",
                    "mode": self.mode,
                    "image_source": image_source,
                }
            )

        last_metadata: Dict[str, Any] = {}

        for attempt_index in range(1, total_attempts + 1):
            try:
                llm_response, metadata = self._attempt_decision_generation(
                    base_prompt=base_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    image_payload=image_payload,
                    image_source=image_source,
                )
                last_metadata = metadata
            except NotImplementedError:
                attempt_errors.append(
                    {
                        "attempt": attempt_index,
                        "error": "not_implemented",
                        "mode": self.mode,
                    }
                )
                decision = self._fallback_decision(problem)
                self.decision_history.append(
                    {
                        "source": "fallback_not_implemented",
                        "decision": decision,
                        "context": context,
                        "mode": self.mode,
                        "attempt_errors": attempt_errors,
                        "image_source": image_source,
                        "image_error": image_error,
                        "image_refs": list(image_refs),
                        "decision_prompt": last_metadata.get("decision_prompt"),
                        "thinking_prompt": last_metadata.get("thinking_prompt"),
                        "thought": last_metadata.get("thought"),
                    }
                )
                return decision
            except Exception as exc:  # pylint: disable=broad-except
                attempt_errors.append(
                    {
                        "attempt": attempt_index,
                        "error": str(exc),
                        "mode": self.mode,
                    }
                )
                continue

            if not isinstance(llm_response, dict):
                attempt_errors.append(
                    {
                        "attempt": attempt_index,
                        "error": "invalid_shape",
                        "mode": self.mode,
                        "decision_prompt": metadata.get("decision_prompt"),
                        "returned_preview": repr(llm_response),
                        "image_source": image_source,
                    }
                )
                continue

            if not self._decision_matches_schema(problem, llm_response):
                attempt_errors.append(
                    {
                        "attempt": attempt_index,
                        "error": "schema_mismatch",
                        "mode": self.mode,
                        "decision_prompt": metadata.get("decision_prompt"),
                        "returned_preview": repr(llm_response),
                        "image_source": image_source,
                    }
                )
                continue

            self.decision_history.append(
                {
                    "source": "llm",
                    "mode": self.mode,
                    "decision": llm_response,
                    "context": context,
                    "attempt": attempt_index,
                    "thought": metadata.get("thought"),
                    "decision_prompt": metadata.get("decision_prompt"),
                    "thinking_prompt": metadata.get("thinking_prompt"),
                    "image_source": metadata.get("image_source", image_source),
                    "image_error": image_error,
                    "image_refs": list(image_refs),
                }
            )
            return llm_response

        decision = self._fallback_decision(problem)
        self.decision_history.append(
            {
                "source": "fallback_max_retries",
                "decision": decision,
                "context": context,
                "mode": self.mode,
                "attempt_errors": attempt_errors,
                "image_source": image_source,
                "image_error": image_error,
                "image_refs": list(image_refs),
                "decision_prompt": last_metadata.get("decision_prompt"),
                "thinking_prompt": last_metadata.get("thinking_prompt"),
                "thought": last_metadata.get("thought"),
            }
        )
        return decision

    def _attempt_decision_generation(
        self,
        *,
        base_prompt: str,
        max_tokens: int,
        temperature: float,
        image_payload: Optional[str],
        image_source: Optional[str],
    ) -> Tuple[Any, Dict[str, Any]]:
        if self.mode == "thinking":
            return self._attempt_thinking_decision(
                base_prompt=base_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                image_payload=image_payload,
                image_source=image_source,
            )
        return self._attempt_json_only_decision(
            base_prompt=base_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image_payload=image_payload,
            image_source=image_source,
        )

    def _attempt_json_only_decision(
        self,
        *,
        base_prompt: str,
        max_tokens: int,
        temperature: float,
        image_payload: Optional[str],
        image_source: Optional[str],
    ) -> Tuple[Any, Dict[str, Any]]:
        decision_prompt = base_prompt
        self.prompt_history.append(decision_prompt)
        self._last_prompt = decision_prompt
        response = self.api.generate_json(
            decision_prompt,
            schema=self.decision_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            image=image_payload,
        )
        metadata = {
            "mode": "json_only",
            "decision_prompt": decision_prompt,
            "image_source": image_source,
        }
        return response, metadata

    def _attempt_thinking_decision(
        self,
        *,
        base_prompt: str,
        max_tokens: int,
        temperature: float,
        image_payload: Optional[str],
        image_source: Optional[str],
    ) -> Tuple[Any, Dict[str, Any]]:
        thinking_prompt = self._build_thinking_prompt(base_prompt)
        self.prompt_history.append(thinking_prompt)
        thought = self.api.generate_text(
            thinking_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            image=image_payload,
        )
        self.thinking_history.append(thought)
        decision_prompt = self._build_json_prompt_with_thinking(base_prompt, thought)
        self.prompt_history.append(decision_prompt)
        self._last_prompt = decision_prompt
        response = self.api.generate_json(
            decision_prompt,
            schema=self.decision_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            image=image_payload,
        )
        metadata = {
            "mode": "thinking",
            "thought": thought,
            "decision_prompt": decision_prompt,
            "thinking_prompt": thinking_prompt,
            "image_source": image_source,
        }
        return response, metadata

    # ---------- Prompt builder ----------
    def _build_decision_prompt(
        self,
        problem: ProblemDefinition,
        context: Mapping[str, Any],
    ) -> str:
        variables = problem.agent_variables(self.agent_id)
        lines = [
            f"You are {self.name} (agent id: {self.agent_id}).",
            problem.description,
            self.instruction,
            "",
            "Decide assignments for the variables you control:",
        ]
        for spec in variables:
            domain_preview = ", ".join(str(value) for value in spec.domain)
            lines.append(
                f"- {spec.name}: choose from [{domain_preview}]"
                f"{' — ' + spec.description if spec.description else ''}"
            )

        # Intentionally omit the historical neighbour decision echo to avoid
        # duplicating information with the current shared assignment.

        vision_hint = context.get("vision_hint")
        if vision_hint:
            lines.extend(["", vision_hint])
            image_sources = context.get("image_sources")
            if image_sources:
                for entry in image_sources:
                    lines.append(f"- {entry}")

        neighbour_messages: List[Tuple[str, str, Any]] = []
        for key, value in context.items():
            if not key.startswith("message_from::") or not value:
                continue
            parts = key.split("::", 2)
            if len(parts) == 3:
                _, neighbour_id, label = parts
            elif len(parts) == 2:
                _, neighbour_id = parts
                label = "message"
            else:
                neighbour_id = "?"
                label = "message"
            neighbour_messages.append((neighbour_id, label, value))
        if neighbour_messages:
            lines.append("")
            lines.append("Messages from neighbours:")
            for neighbour_id, label, message in sorted(neighbour_messages):
                label_text = f" ({label})" if label and label != "message" else ""
                lines.append(f"- {neighbour_id}{label_text}: {message}")

        # Render your own natural communication messages without technical labels
        your_messages: List[str] = []
        for key, value in context.items():
            if key.startswith("your_message::") and value:
                # Ignore the label suffix (e.g., "naturalcomm"); show only the message text
                try:
                    your_messages.append(str(value))
                except Exception:
                    pass
        if your_messages:
            lines.append("")
            lines.append("Your preference messages:")
            for message in your_messages:
                lines.append(f"- {message}")

        # Objective & Strategy (if provided by PromptBuilder)
        obj = context.get("objective")
        rubric = context.get("scoring_rubric_generic")
        trade = context.get("tradeoff_guidance")
        round_strategy = context.get("round_strategy")
        if any([obj, rubric, trade, round_strategy]):
            lines.append("")
            lines.append("Objective & Scoring:")
            if obj:
                lines.append(f"- Objective: {obj}")
            if rubric:
                lines.append(f"- Scoring rubric: {rubric}")
            if trade:
                lines.append(f"- Trade-offs: {trade}")
            if round_strategy:
                lines.append(f"- Round strategy: {round_strategy}")
        
        # Problem-specific guidance (separate block from generic objective)
        problem_guidance = context.get("problem_guidance")
        if isinstance(problem_guidance, Mapping):
            title = str(problem_guidance.get("title") or "Problem guidance:").strip()
            items = problem_guidance.get("lines")
            if isinstance(items, Sequence) and items:
                lines.append("")
                lines.append(title)
                for item in items:
                    lines.append(str(item))

        # Place Action after high-level rubric/guidance so the model sees goals first
        algorithm_instruction = context.get("algorithm_instruction")
        if algorithm_instruction:
            lines.extend(["", "Action:", algorithm_instruction])

        # Optional factor information block (problem-provided)
        if context.get("show_factor_information", False):
            info = context.get("factor_information")
            if isinstance(info, Mapping):
                title = str(info.get("title") or "Factor information:").strip()
                content = info.get("lines")
                if isinstance(content, Sequence) and content:
                    lines.append("")
                    lines.append(title)
                    for item in content:
                        lines.append(str(item))

        # Optional neighbour assignments block
        if context.get("show_neighbour_assignments", True):
            shared_assignment = context.get("shared_assignment")
            if isinstance(shared_assignment, Mapping):
                controlled = {spec.name for spec in variables}
                # Factor-based filtering: only show variables that co-occur in at
                # least one factor with any of the agent's controlled variables.
                neighbour_vars: Set[str] = set()
                try:
                    for factor in problem.list_factors():
                        scope = set(factor.scope)
                        if scope & controlled:
                            neighbour_vars.update(scope)
                except Exception:
                    neighbour_vars = set()

                def _is_visible(var_name: str) -> bool:
                    if var_name in controlled:
                        return False
                    if neighbour_vars:
                        return var_name in neighbour_vars
                    # Fallback: if we couldn't compute, show all non-controlled
                    return True

                render_map = {}
                try:
                    rm = context.get("neighbour_assignment_render")
                    if isinstance(rm, Mapping):
                        render_map = {str(k): str(v) for k, v in rm.items()}
                except Exception:
                    render_map = {}

                visible_items = [
                    (name, render_map.get(name, shared_assignment[name]))
                    for name in sorted(shared_assignment)
                    if _is_visible(name)
                ]
                if visible_items:
                    lines.append("")
                    lines.append("Current neighbour assignments:")
                    for name, value in visible_items:
                        lines.append(f"- {name}: {value}")

        lines.extend(
            [
                "",
                "Output requirements:",
                "1. Respond with a single JSON object and nothing else.",
                "2. Use the exact keys defined in the schema below.",
                "3. Ensure every value complies with the allowed domain.",
                "",
                "JSON schema:",
                json.dumps(self.decision_schema, indent=2, ensure_ascii=False),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _split_prompt_sections(base_prompt: str) -> Tuple[str, str, str]:
        action_marker = "\nAction:\n"
        requirements_marker = "\nOutput requirements:\n"

        core_and_action = base_prompt
        requirements_block = ""
        if requirements_marker in base_prompt:
            core_and_action, tail = base_prompt.split(requirements_marker, 1)
            requirements_block = "Output requirements:\n" + tail.strip("\n")

        action_block = ""
        if action_marker in core_and_action:
            core_part, action_part = core_and_action.split(action_marker, 1)
            action_block = action_part.strip()
        else:
            core_part = core_and_action

        return core_part.rstrip(), action_block, requirements_block

    @staticmethod
    def _build_thinking_prompt(base_prompt: str) -> str:
        core_text, action_text, _ = CollaborativeAgent._split_prompt_sections(base_prompt)
        sections: List[str] = [core_text]
        if action_text:
            sections.extend(["", "Action:", action_text])
        sections.extend(
            [
                "",
                "Before producing your final decision, reason step by step in natural language.",
                "Do not output JSON or schema-formatted content during this response.",
            ]
        )
        return "\n".join(sections)

    @staticmethod
    def _build_json_prompt_with_thinking(base_prompt: str, thought: str) -> str:
        core_text, action_text, requirements_text = CollaborativeAgent._split_prompt_sections(base_prompt)
        sections: List[str] = [core_text]
        if thought:
            sections.extend(["", "Prior reasoning:", thought.strip()])
        if action_text:
            sections.extend(["", "Action:", action_text])
        sections.extend(
            [
                "",
                "Use the reasoning above to finalise your decision, then follow the output requirements.",
            ]
        )
        if requirements_text:
            sections.extend(["", requirements_text])
        return "\n".join(sections)

    # ---------- Image helpers ----------
    def _prepare_image_payload(
        self,
        image_refs: Sequence[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if not image_refs:
            return None, None, None
        source = image_refs[0]
        if not source:
            return None, source, "empty_reference"
        try:
            payload = self._load_image_reference(source)
        except Exception as exc:  # pylint: disable=broad-except
            return None, source, str(exc)
        return payload, source, None

    def _load_image_reference(self, reference: str) -> str:
        ref = reference.strip()
        if not ref:
            raise ValueError("Empty image reference.")
        if ref.startswith("data:"):
            return ref
        if ref.startswith("http://") or ref.startswith("https://"):
            return ref
        if ref.startswith("file://"):
            path = Path(ref[7:])
        else:
            path = Path(ref)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Image reference is not a file: {path}")
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    # ---------- Validation & fallback ----------
    def _decision_matches_schema(
        self,
        problem: ProblemDefinition,
        decision: Mapping[str, Any],
    ) -> bool:
        controlled_variables = {spec.name: spec for spec in problem.agent_variables(self.agent_id)}
        if set(decision.keys()) != set(controlled_variables.keys()):
            return False
        for name, spec in controlled_variables.items():
            error = spec.validate(decision[name])
            if error:
                return False
        return True

    def _fallback_decision(self, problem: ProblemDefinition) -> Dict[str, Any]:
        decision: Dict[str, Any] = {}
        for spec in problem.agent_variables(self.agent_id):
            best_value = spec.domain[0]
            best_score = float("-inf")
            for candidate in spec.domain:
                test_assignment = {spec.name: candidate}
                score = 0.0
                for factor in problem.personal_preference_factors(self.agent_id):
                    try:
                        score += factor.evaluate(test_assignment)
                    except KeyError:
                        continue
                if score > best_score:
                    best_value = candidate
                    best_score = score
            decision[spec.name] = best_value
        return decision

    # ---------- Introspection ----------
    @property
    def last_prompt(self) -> Optional[str]:
        return self._last_prompt

    @property
    def last_thought(self) -> Optional[str]:
        if not self.thinking_history:
            return None
        return self.thinking_history[-1]
