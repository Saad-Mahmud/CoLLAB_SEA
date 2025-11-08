from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .algorithm import DecisionAlgorithm, ICCDecisionOnlyAlgorithm
from agent_layer import CollaborativeAgent
from problem_layer import ProblemDefinition

from .base import CommunicationProtocol, ProtocolResult, ProtocolTurn
from .prompt_builder import DefaultPromptBuilder


class ICCNaturalCommProtocol(CommunicationProtocol):
    """
    ICC variant with a natural-language broadcast before sequential JSON decisions.

    Flow per round:
      1) Synchronous broadcast: each agent emits a short, free-text message
         for neighbours describing likely next choices with clear identifiers.
      2) Sequential JSON decisions (Gauss–Seidel) using json_only mode,
         augmented with neighbour messages in the decision prompt context.
    """

    name = "ICC_NM"

    def __init__(
        self,
        *,
        algorithm: Optional[DecisionAlgorithm] = None,
        max_rounds: int = 1,
        # Message generation controls
        message_max_tokens: int = 128,
        message_temperature: float = 0.4,
        batch_messages: bool = True,
        # Decision controls
        decision_max_tokens: int = 512,
        decision_temperature: float = 0.0,
        # Prompt visibility toggles for decisions
        show_neighbour_assignments: bool = True,
        show_factor_information: bool = False,
    ) -> None:
        self.algorithm = algorithm or ICCDecisionOnlyAlgorithm()
        self.max_rounds = max_rounds
        self.message_max_tokens = message_max_tokens
        self.message_temperature = message_temperature
        self.batch_messages = bool(batch_messages)
        self.decision_max_tokens = decision_max_tokens
        self.decision_temperature = decision_temperature
        self.show_neighbour_assignments = bool(show_neighbour_assignments)
        self.show_factor_information = bool(show_factor_information)

    def run(
        self,
        problem: ProblemDefinition,
        agents: Sequence[CollaborativeAgent],
        *,
        rounds: Optional[int] = None,
        initial_assignment: Optional[Mapping[str, Any]] = None,
        static_context: Optional[Mapping[str, Any]] = None,
    ) -> ProtocolResult:
        if not agents:
            raise ValueError("At least one agent is required to run the protocol.")

        total_rounds = rounds if rounds is not None else self.max_rounds
        if total_rounds < 1:
            raise ValueError("Rounds must be at least 1.")

        shared_assignment: Dict[str, Any] = dict(initial_assignment or {})
        turns: List[ProtocolTurn] = []
        static_context_payload = dict(static_context or {})
        static_context_payload.setdefault("total_rounds", total_rounds)
        static_context_payload.setdefault("show_neighbour_assignments", self.show_neighbour_assignments)
        static_context_payload.setdefault("show_factor_information", self.show_factor_information)

        # Build neighbour graph from factor scopes (same as other ICC variants)
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

        builder = DefaultPromptBuilder()

        for round_index in range(1, total_rounds + 1):
            # Phase 1: synchronous natural-communication broadcasts
            snapshot = dict(shared_assignment)
            msg_prompts: List[str] = []
            per_agent_inputs: Dict[str, Mapping[str, Any]] = {}
            per_agent_images: Dict[str, Sequence[str]] = {}
            msg_prompt_by_agent: Dict[str, str] = {}
            for agent in agents:
                # Start from decision inputs then strip decision-specific fields
                inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=snapshot,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                # Remove decision/action-specific bits to produce a pure message context
                context = dict(inputs.context)
                context.pop("algorithm_instruction", None)
                # Build controlled variables summary and variable hints
                self._augment_with_variable_context(problem, agent.agent_id, context)
                # Compose the generic broadcast prompt
                prompt = self._compose_broadcast_prompt(context)

                per_agent_inputs[agent.agent_id] = context
                per_agent_images[agent.agent_id] = tuple(inputs.images)
                msg_prompts.append(prompt)
                msg_prompt_by_agent[agent.agent_id] = prompt

            messages_by_agent: Dict[str, str] = {agent.agent_id: "" for agent in agents}
            # Batch if supported
            if self.batch_messages and hasattr(agents[0].api, "generate_text_batch"):
                try:
                    outputs = agents[0].api.generate_text_batch(
                        msg_prompts,
                        max_tokens=self.message_max_tokens,
                        temperature=self.message_temperature,
                    )
                    for agent, text in zip(agents, outputs):
                        messages_by_agent[agent.agent_id] = str(text or "").strip()
                except NotImplementedError:
                    # Fallback to sequential
                    for agent, prompt in zip(agents, msg_prompts):
                        messages_by_agent[agent.agent_id] = agent.api.generate_text(
                            prompt,
                            max_tokens=self.message_max_tokens,
                            temperature=self.message_temperature,
                            image=None,
                        ).strip()
            else:
                for agent, prompt in zip(agents, msg_prompts):
                    messages_by_agent[agent.agent_id] = agent.api.generate_text(
                        prompt,
                        max_tokens=self.message_max_tokens,
                        temperature=self.message_temperature,
                        image=None,
                    ).strip()

            # Phase 2: sequential JSON decisions with messages injected
            for agent in agents:
                # Prepare inputs afresh against the latest shared assignment
                decision_inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=shared_assignment,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                # Inject neighbour messages into context under message_from::<id>::naturalcomm keys
                ctx = dict(decision_inputs.context)
                for sender_id, text in messages_by_agent.items():
                    if not text:
                        continue
                    if sender_id == agent.agent_id:
                        # Surface own message as well (optional, helpful for consistency)
                        ctx[f"your_message::naturalcomm"] = text
                        continue
                    if sender_id not in agent.neighbours:
                        continue
                    ctx[f"message_from::{sender_id}::naturalcomm"] = text

                decision = agent.decide(
                    problem,
                    ctx,
                    images=decision_inputs.images,
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
                        broadcast_prompt=msg_prompt_by_agent.get(agent.agent_id),
                        broadcast_message=messages_by_agent.get(agent.agent_id, ""),
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

    # ---------- Helpers ----------
    @staticmethod
    def _compose_broadcast_prompt(context: Mapping[str, Any]) -> str:
        """Problem-independent message prompt, using decision-like context minus action text."""
        round_idx = context.get("round")
        total_rounds = context.get("total_rounds")
        neighbour_render = context.get("neighbour_assignment_render")
        factor_info = context.get("factor_information")
        controlled = context.get("controlled_variables")
        var_hints = context.get("variable_hints")

        lines: List[str] = []
        lines.append(
            "Compose a concise message that will be seen by all your neighbours for the upcoming round."
        )
        lines.append(
            "Write one paragraph of plain text only (no JSON, no bullet lists, no headings)."
        )
        lines.append(
            "Include the top one or two candidate assignments you are currently considering for each variable you control,"
            " and make each choice unambiguous by naming the variable identifier and any useful attributes (e.g., IDs, time windows, loads in kW, colours)."
        )
        lines.append("")
        lines.append(f"Round: {round_idx}/{total_rounds}")
        if controlled:
            lines.append(f"Variables you control: {controlled}")
        if var_hints:
            lines.append(f"Variable details: {var_hints}")
        if isinstance(neighbour_render, Mapping) and neighbour_render:
            lines.append("Current neighbour assignments:")
            # Render a compact one-line preview
            preview = ", ".join(f"{k}={v}" for k, v in sorted(neighbour_render.items()))
            lines.append(preview)
        if isinstance(factor_info, Mapping):
            info_lines = factor_info.get("lines")
            if isinstance(info_lines, Sequence) and info_lines:
                lines.append("Factor information:")
                lines.append(" ".join(str(x) for x in info_lines[:2]))  # keep concise
        lines.append("")
        lines.append("Write the message now as one concise paragraph.")
        return "\n".join(lines)

    @staticmethod
    def _augment_with_variable_context(
        problem: ProblemDefinition,
        agent_id: str,
        context: Dict[str, Any],
    ) -> None:
        """Add controlled variable summaries and generic per-variable hints, when available."""
        # Controlled variables summary with brief domain hint
        controlled: List[str] = []
        hints: List[str] = []
        for spec in problem.agent_variables(agent_id):
            preview_vals = list(spec.domain)[:4]
            preview = ", ".join(str(v) for v in preview_vals)
            controlled.append(f"{spec.name}: [{preview}]")
        context["controlled_variables"] = "; ".join(controlled)

        # Problem-aware hints via instance back-refs (best-effort; silent if unavailable)
        try:
            # Meeting Scheduling
            ms_inst = getattr(problem, "_meeting_scheduling_instance", None)
            if ms_inst is not None:
                # Map meeting_id -> (title, window)
                meta = {m.meeting_id: (m.title, m.start, m.end) for m in ms_inst.meetings}
                for spec in problem.agent_variables(agent_id):
                    # Var format: Agent__mXXX
                    parts = str(spec.name).split("__", 1)
                    meeting_id = parts[1] if len(parts) == 2 else None
                    if meeting_id and meeting_id in meta:
                        title, start, end = meta[meeting_id]
                        hints.append(f"{spec.name} -> {meeting_id} '{title}' [{start},{end})")
        except Exception:
            pass
        try:
            # Smart Grid
            sg_inst = getattr(problem, "_smart_grid_instance", None)
            if sg_inst is not None:
                for spec in problem.agent_variables(agent_id):
                    m = sg_inst.machines.get(spec.name)
                    if m is not None:
                        load = sg_inst.machine_powers.get(spec.name)
                        if load is None and m.end > m.start:
                            try:
                                load = float(m.energy) / max(1, (m.end - m.start))
                            except Exception:
                                load = None
                        if load is not None:
                            hints.append(
                                f"{spec.name} -> {getattr(m,'label',spec.name)} load={load:.1f}kW window=[{m.start},{m.end})"
                            )
                        else:
                            hints.append(f"{spec.name} -> {getattr(m,'label',spec.name)} window=[{m.start},{m.end})")
        except Exception:
            pass
        try:
            # Personal Assistant
            pa_inst = getattr(problem, "_personal_assistant_instance", None)
            if pa_inst is not None:
                # Spec name typically: "{agent}'s Outfit"
                outfits = pa_inst.wardrobe.get(agent_id) or []
                # List first few colours with indices
                colour_preview: List[str] = []
                for idx, outfit in enumerate(outfits[:4], start=1):
                    colour = getattr(outfit, "color", None) or getattr(outfit, "colour", None) or "?"
                    colour_preview.append(f"{idx}:{colour}")
                if colour_preview:
                    for spec in problem.agent_variables(agent_id):
                        hints.append(f"{spec.name} -> colours {', '.join(colour_preview)}")
        except Exception:
            pass

        if hints:
            context["variable_hints"] = "; ".join(hints)
