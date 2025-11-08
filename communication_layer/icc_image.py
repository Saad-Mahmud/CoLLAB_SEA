from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .algorithm import DecisionAlgorithm, ICCDecisionOnlyAlgorithm
from agent_layer.agent_thinking import ThinkingAgent
from problem_layer import ProblemDefinition

from .base import CommunicationProtocol, ProtocolResult, ProtocolTurn


class ICCImageProtocol(CommunicationProtocol):
    """
    ICC variant for vision-aware planning:
    1) Synchronous free-text step: each agent writes a brief image description
       (if images are present) and a concrete action plan for next decision.
    2) Sequential JSON decisions using the agent's plan as prior reasoning.

    Images are surfaced from the static_context via PromptBuilder (image_map,
    vision_hints, etc.). Planning prompts are batched when supported.
    """

    name = "icc_image"

    def __init__(
        self,
        *,
        algorithm: Optional[DecisionAlgorithm] = None,
        max_rounds: int = 1,
        # Planning (free text)
        planning_max_tokens: int = 512,
        planning_temperature: float = 0.3,
        batch_planning: bool = True,
        # Decision (JSON)
        decision_max_tokens: int = 512,
        decision_temperature: float = 0.0,
        # Prompt visibility
        show_neighbour_assignments: bool = True,
        show_factor_information: bool = False,
    ) -> None:
        self.algorithm = algorithm or ICCDecisionOnlyAlgorithm()
        self.max_rounds = int(max_rounds)
        self.planning_max_tokens = int(planning_max_tokens)
        self.planning_temperature = float(planning_temperature)
        self.batch_planning = bool(batch_planning)
        self.decision_max_tokens = int(decision_max_tokens)
        self.decision_temperature = float(decision_temperature)
        self.show_neighbour_assignments = bool(show_neighbour_assignments)
        self.show_factor_information = bool(show_factor_information)

    def run(
        self,
        problem: ProblemDefinition,
        agents: Sequence[ThinkingAgent],
        *,
        rounds: Optional[int] = None,
        initial_assignment: Optional[Mapping[str, Any]] = None,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> ProtocolResult:
        if not agents:
            raise ValueError("At least one agent is required to run the protocol.")

        total_rounds = int(rounds if rounds is not None else self.max_rounds)
        if total_rounds < 1:
            raise ValueError("Rounds must be at least 1.")

        shared_assignment: Dict[str, Any] = dict(initial_assignment or {})
        turns: List[ProtocolTurn] = []
        static_context_payload = dict(static_context or {})
        static_context_payload.setdefault("total_rounds", total_rounds)
        static_context_payload.setdefault("show_neighbour_assignments", self.show_neighbour_assignments)
        static_context_payload.setdefault("show_factor_information", self.show_factor_information)

        # Build neighbour graph from factor scopes
        var_owner = {name: spec.owner for name, spec in problem.variables.items()}
        adjacency: Dict[str, set] = {agent.agent_id: set() for agent in agents}
        for factor in problem.list_factors():
            owners = {var_owner.get(var) for var in factor.scope if var in var_owner}
            owners.discard(None)  # type: ignore[arg-type]
            owners = set(owners)  # type: ignore[assignment]
            if len(owners) <= 1:
                continue
            for a in owners:
                for b in owners:
                    if a == b:
                        continue
                    adjacency[a].add(b)  # type: ignore[index]
        for agent in agents:
            agent.set_neighbours(sorted(adjacency.get(agent.agent_id, set())))

        for round_index in range(1, total_rounds + 1):
            # Phase 1: synchronous image description + plan
            snapshot = dict(shared_assignment)
            plan_prompts: List[str] = []
            per_agent_inputs: Dict[str, Any] = {}
            per_agent_images: Dict[str, Sequence[str]] = {}
            for agent in agents:
                inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=snapshot,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                # Build the base thinking prompt (decision context without schema)
                base_thinking = agent.prepare_thinking_prompt(
                    problem,
                    inputs.context,
                    inputs.images,
                )
                prompt = self._compose_image_plan_prompt(base_thinking)
                plan_prompts.append(prompt)
                per_agent_inputs[agent.agent_id] = inputs
                per_agent_images[agent.agent_id] = tuple(inputs.images)

            plans_by_agent: Dict[str, str] = {agent.agent_id: "" for agent in agents}
            if self.batch_planning and hasattr(agents[0].api, "generate_text_batch"):
                try:
                    # Align one image (if any) per prompt
                    batch_images: List[Optional[str]] = []
                    for agent in agents:
                        imgs = per_agent_images.get(agent.agent_id, ())
                        batch_images.append(imgs[0] if imgs else None)
                    outputs = agents[0].api.generate_text_batch(
                        plan_prompts,
                        max_tokens=self.planning_max_tokens,
                        temperature=self.planning_temperature,
                        images=batch_images,
                    )
                    for agent, plan in zip(agents, outputs):
                        plans_by_agent[agent.agent_id] = str(plan or "").strip()
                except NotImplementedError:
                    # Fallback to sequential per-agent planning
                    for agent, prompt in zip(agents, plan_prompts):
                        imgs = per_agent_images.get(agent.agent_id, ())
                        img_ref = imgs[0] if imgs else None
                        msg = agent.api.generate_text(
                            prompt,
                            max_tokens=self.planning_max_tokens,
                            temperature=self.planning_temperature,
                            image=img_ref,
                        )
                        plans_by_agent[agent.agent_id] = str(msg or "").strip()
            else:
                for agent, prompt in zip(agents, plan_prompts):
                    imgs = per_agent_images.get(agent.agent_id, ())
                    img_ref = imgs[0] if imgs else None
                    msg = agent.api.generate_text(
                        prompt,
                        max_tokens=self.planning_max_tokens,
                        temperature=self.planning_temperature,
                        image=img_ref,
                    )
                    plans_by_agent[agent.agent_id] = str(msg or "").strip()

            # Phase 2: sequential JSON decisions using plan as prior reasoning
            for agent in agents:
                inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=shared_assignment,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                plan_text = plans_by_agent.get(agent.agent_id, "")
                # Do NOT attach images in the JSON phase; rely solely on the prior reasoning
                # Also strip any vision hints or image source references from the decision prompt context
                ctx = dict(inputs.context)
                ctx.pop("vision_hint", None)
                ctx.pop("image_sources", None)
                decision = agent.decide_from_thought(
                    problem,
                    ctx,
                    plan_text,
                    None,
                    max_tokens=self.decision_max_tokens,
                    temperature=self.decision_temperature,
                )
                shared_assignment.update(decision)

                self.algorithm.on_agent_decision(
                    agent=agent,
                    problem=problem,
                    decision=decision,
                    round_index=round_index,
                    shared_assignment=shared_assignment,
                )

                for neighbour in agents:
                    if neighbour.agent_id == agent.agent_id:
                        continue
                    neighbour.notify_neighbour_decision(agent.agent_id, decision, round_index)

                turn_evaluation = problem.eval(shared_assignment)
                turns.append(
                    ProtocolTurn(
                        round_index=round_index,
                        agent_id=agent.agent_id,
                        decision=dict(decision),
                        prompt=agent.last_prompt,
                        joint_assignment=dict(shared_assignment),
                        evaluation=turn_evaluation,
                    )
                )

        evaluation = problem.eval(shared_assignment)
        return ProtocolResult(
            protocol=self.name,
            turns=turns,
            final_assignment=shared_assignment,
            evaluation=evaluation if evaluation.get("valid", False) else evaluation,
        )

    @staticmethod
    def _compose_image_plan_prompt(base_prompt: str) -> str:
        """
        Extend a generic thinking prompt with image description + action plan guidance.
        The base_prompt is the agent's decision context without the JSON schema.
        """
        sections: List[str] = [base_prompt.rstrip(), ""]
        sections.extend(
            [
                "Image understanding & planning:",
                "- If images are referenced above, briefly describe what they show, focusing on identifiers and attributes relevant to your variables (e.g., IDs, titles, time windows, loads in kW, colours).",
                "- Then propose a concrete plan for the next JSON decision: list your likely choices per variable (1–2 candidates) with a short justification.",
                "Do not output JSON or bullet lists; write one concise paragraph of natural language.",
            ]
        )
        return "\n".join(sections)
