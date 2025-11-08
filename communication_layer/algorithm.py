from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from agent_layer import CollaborativeAgent
from problem_layer import ProblemDefinition
from .prompt_builder import DefaultPromptBuilder, PromptInputs


class DecisionAlgorithm:
    """
    Minimal decision algorithm interface used by ICC-style protocols.
    Responsible for preparing per-turn context and optional image inputs.
    """

    def prepare_agent_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        round_index: int,
        shared_assignment: Mapping[str, Any],
        protocol_name: str,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> PromptInputs:
        # Delegate to centralised prompt builder for consistency
        builder = DefaultPromptBuilder()
        return builder.prepare_decision_inputs(
            agent=agent,
            problem=problem,
            round_index=round_index,
            shared_assignment=shared_assignment,
            protocol_name=protocol_name,
            static_context=static_context,
        )

    def collect_image_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        static_context: Mapping[str, Any],
    ) -> Sequence[str]:
        # Keep subclassing compatibility: call into the builder by default.
        builder = DefaultPromptBuilder()
        return builder.collect_images(agent=agent, static_context=static_context)

    def on_agent_decision(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        decision: Mapping[str, Any],
        round_index: int,
        shared_assignment: Mapping[str, Any],
    ) -> None:  # pragma: no cover - placeholder hook
        """Hook invoked after an agent produces a decision (no-op by default)."""
        return

    # Normalisation helpers are now owned by PromptBuilder


class ICCDecisionOnlyAlgorithm(DecisionAlgorithm):
    """Default algorithm for ICC decision-only style protocols."""

    def prepare_agent_inputs(
        self,
        *,
        agent: CollaborativeAgent,
        problem: ProblemDefinition,
        round_index: int,
        shared_assignment: Mapping[str, Any],
        protocol_name: str,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> PromptInputs:
        inputs = super().prepare_agent_inputs(
            agent=agent,
            problem=problem,
            round_index=round_index,
            shared_assignment=shared_assignment,
            protocol_name=protocol_name,
            static_context=static_context,
        )
        # Strengthen action guidance
        inputs.context["algorithm_instruction"] = (
            "Decide values for your controlled variables only. "
            "Output strictly valid JSON matching your schema; do not include extra keys."
        )
        return inputs
