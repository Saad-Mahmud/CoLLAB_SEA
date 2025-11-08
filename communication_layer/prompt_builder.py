from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent_layer import CollaborativeAgent
from problem_layer import ProblemDefinition
import warnings
import importlib


@dataclass
class PromptInputs:
    context: Dict[str, Any]
    images: Sequence[str]


class PromptBuilder:
    """
    Centralised prompt/context construction for all protocol phases:
    - Decision turns (ICC variants)
    - Preference summaries (ICC one-shot, CC)
    - Decision + short message (decision-message ICC)
    """

    # ---------- Decision turns ----------
    def prepare_decision_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        round_index: int,
        shared_assignment: Mapping[str, Any],
        protocol_name: str,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> PromptInputs:
        base: Dict[str, Any] = {
            "protocol": protocol_name,
            "round": round_index,
            "shared_assignment": dict(shared_assignment),
            "algorithm_instruction": (
                "Decide values for your controlled variables only. Output strictly valid JSON matching your schema."
            ),
        }
        if static_context:
            base.update(dict(static_context))
        # Add generic objective, rubric, and strategy guidance (domain-agnostic)
        try:
            factors = list(problem.list_factors())
            unary_count = sum(1 for f in factors if len(f.scope) == 1)
            coord_count = sum(1 for f in factors if len(f.scope) > 1)
            max_utility = float(sum(1 if len(f.scope) == 1 else 2 for f in factors))
        except Exception:
            unary_count = 0
            coord_count = 0
            max_utility = None  # type: ignore[assignment]

        base.setdefault("objective", "Maximise total utility (sum of all factor utilities).")
        base.setdefault(
            "scoring_rubric_generic",
            "Utility aggregates all factors. Unary factors capture local goals/constraints; multi-variable factors capture coordination among agents. Avoid violating constraints; favour choices that increase total utility.",
        )
        base.setdefault(
            "tradeoff_guidance",
            "Favour assignments that increase joint utility even if personal utility decreases slightly, "
            "provided the net gain is positive.",
        )
        total_rounds = None
        if isinstance(static_context, Mapping):
            total_rounds = static_context.get("total_rounds")
        base.setdefault(
            "round_strategy",
            "Early rounds: choose strong personal options that likely coordinate well. "
            "Later rounds: respect neighbour decisions and adjust only when joint utility improves; avoid oscillation.",
        )
        base.setdefault(
            "factor_summary",
            {
                "unary_factors": unary_count,
                "coordination_factors": coord_count,
                "min_utility": 0.0,
                "max_utility": max_utility,
                "total_rounds": total_rounds,
            },
        )
        images = tuple(self.collect_images(agent=agent, static_context=static_context or {}))
        if images:
            base.setdefault("vision_hint", "You may inspect provided images to inform your decision.")
            base.setdefault("image_sources", images)

        # Optional problem-specific factor information summary
        if base.get("show_factor_information", False):
            try:
                modname = f"problem_layer.{problem.name}.problem"
                mod = importlib.import_module(modname)
            except Exception as exc:
                warnings.warn(f"Factor info: failed to import {modname}: {exc}")
            else:
                formatter = getattr(mod, "format_factor_information", None)
                if callable(formatter):
                    try:
                        info = formatter(
                            problem=problem,
                            agent_id=agent.agent_id,
                            shared_assignment=dict(shared_assignment),
                        )
                    except Exception as exc:
                        warnings.warn(f"Factor info: formatter raised: {exc}")
                        info = None
                    if info:
                        base["factor_information"] = info
                    else:
                        warnings.warn(
                            f"Factor info: empty for problem={problem.name} agent={agent.agent_id}"
                        )
                else:
                    warnings.warn(
                        f"Factor info: problem={problem.name} has no format_factor_information()"
                    )
        # Optional problem-specific guidance block (separate from generic Objective & Strategy)
        try:
            modname = f"problem_layer.{problem.name}.problem"
            mod = importlib.import_module(modname)
            guidance_fn = getattr(mod, "format_problem_guidance", None)
            if callable(guidance_fn):
                try:
                    guidance = guidance_fn(problem=problem, agent_id=agent.agent_id)
                except Exception as exc:
                    warnings.warn(f"Problem guidance: formatter raised: {exc}")
                    guidance = None
                if guidance:
                    base["problem_guidance"] = guidance
        except Exception:
            # Non-fatal if problem has no guidance hook
            pass

        # Optional problem-specific neighbour assignment rendering
        if base.get("show_neighbour_assignments", True):
            try:
                modname = f"problem_layer.{problem.name}.problem"
                mod = importlib.import_module(modname)
                renderer = getattr(mod, "format_neighbour_assignments", None)
                if callable(renderer):
                    try:
                        rendered = renderer(problem=problem, shared_assignment=dict(shared_assignment))
                    except Exception as exc:
                        warnings.warn(f"Neighbour render: formatter raised: {exc}")
                        rendered = None
                    if isinstance(rendered, Mapping) and rendered:
                        base["neighbour_assignment_render"] = dict(rendered)
            except Exception:
                # Ignore if module or function not present
                pass

        return PromptInputs(context=base, images=images)

    # ---------- Preference summary ----------
    def prepare_preference_summary_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> PromptInputs:
        variables = problem.agent_variables(agent.agent_id)
        base: Dict[str, Any] = {
            "protocol": "preference_summary",
            "round": 0,
            "stage": "preference_summary",
            "controlled_variables": [spec.name for spec in variables],
        }
        if static_context:
            base.update(dict(static_context))
        base.setdefault(
            "summary_guidance",
            "Summarise: (1) top-3 admissible choices with reasons, (2) hard no-go constraints, "
            "(3) per-partner coordination preference (match/contrast) and strength, (4) willingness to compromise.",
        )
        images = tuple(self.collect_images(agent=agent, static_context=static_context or {}))
        if images:
            base.setdefault("vision_hint", "You may refer to the images while summarising preferences.")
            base.setdefault("image_sources", images)
        return PromptInputs(context=base, images=images)

    # ---------- Decision + message ----------
    def prepare_decision_message_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        round_index: int,
        shared_assignment: Mapping[str, Any],
        protocol_name: str,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> PromptInputs:
        payload = self.prepare_decision_inputs(
            agent=agent,
            problem=problem,
            round_index=round_index,
            shared_assignment=shared_assignment,
            protocol_name=protocol_name,
            static_context=static_context,
        )
        payload.context["stage"] = "decision_with_message"
        return payload

    # ---------- Image helpers ----------
    def collect_images(self, *, agent: CollaborativeAgent, static_context: Mapping[str, Any]) -> Sequence[str]:
        images: List[str] = []
        # Common keys
        for key in ("image", "image_url", "image_path"):
            value = static_context.get(key)
            images.extend(self._normalise_image_field(value))
        for key in ("images", "image_urls", "image_paths"):
            value = static_context.get(key)
            images.extend(self._normalise_image_field(value))
        # Support nested media blocks
        media = static_context.get("media")
        if isinstance(media, Mapping):
            for key in ("image", "images", "image_url", "image_urls", "image_path", "image_paths"):
                value = media.get(key)  # type: ignore[index]
                images.extend(self._normalise_image_field(value))
        # Optional per-agent mapping
        image_map = static_context.get("image_map")
        if isinstance(image_map, Mapping):
            ref = image_map.get(agent.agent_id)
            images.extend(self._normalise_image_field(ref))
        # Deduplicate preserving order
        seen = set()
        unique: List[str] = []
        for item in images:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    @staticmethod
    def _normalise_image_field(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            v = value.strip()
            return [v] if v else []
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            result: List[str] = []
            for item in value:
                result.extend(PromptBuilder._normalise_image_field(item))
            return result
        if isinstance(value, Mapping):
            result: List[str] = []
            for key in ("url", "path", "image", "image_url", "image_path"):
                if key in value:
                    result.extend(PromptBuilder._normalise_image_field(value[key]))
            return result
        return []


class DefaultPromptBuilder(PromptBuilder):
    pass
