from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .algorithm import DecisionAlgorithm, ICCDecisionOnlyAlgorithm
from agent_layer.agent_thinking import ThinkingAgent
from problem_layer import ProblemDefinition

from .base import CommunicationProtocol, ProtocolResult, ProtocolTurn


class ICCThinkingProtocol(CommunicationProtocol):
    """
    ICC variant with two phases per round:
    1) Synchronous "thinking" (free text) for all agents from the same snapshot — batchable.
    2) Sequential JSON decisions (Gauss–Seidel), each using its prior thought and the latest state.
    """

    name = "ICC_CoT"

    def __init__(
        self,
        *,
        algorithm: Optional[DecisionAlgorithm] = None,
        max_rounds: int = 1,
        thinking_max_tokens: int = 512,
        thinking_temperature: float = 0.3,
        decision_max_tokens: int = 512,
        decision_temperature: float = 0.0,
        show_neighbour_assignments: bool = True,
        show_factor_information: bool = False,
        batch_thinking: bool = True,
    ) -> None:
        self.algorithm = algorithm or ICCDecisionOnlyAlgorithm()
        self.max_rounds = max_rounds
        self.thinking_max_tokens = thinking_max_tokens
        self.thinking_temperature = thinking_temperature
        self.decision_max_tokens = decision_max_tokens
        self.decision_temperature = decision_temperature
        self.show_neighbour_assignments = bool(show_neighbour_assignments)
        self.show_factor_information = bool(show_factor_information)
        self.batch_thinking = bool(batch_thinking)

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

        total_rounds = rounds if rounds is not None else self.max_rounds
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
            # Phase 1: synchronous thinking from last round's snapshot
            snapshot = dict(shared_assignment)
            prompts: List[str] = []
            inputs_cache: Dict[str, Any] = {}
            for agent in agents:
                turn_inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=snapshot,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                thinking_prompt = agent.prepare_thinking_prompt(
                    problem,
                    turn_inputs.context,
                    turn_inputs.images,
                )
                inputs_cache[agent.agent_id] = turn_inputs  # reuse image refs for decision
                prompts.append(thinking_prompt)

            thoughts_by_agent: Dict[str, str] = {agent.agent_id: "" for agent in agents}
            if self.batch_thinking and hasattr(agents[0].api, "generate_text_batch"):
                try:
                    outputs = agents[0].api.generate_text_batch(
                        prompts,
                        max_tokens=self.thinking_max_tokens,
                        temperature=self.thinking_temperature,
                    )
                    for agent, thought in zip(agents, outputs):
                        thoughts_by_agent[agent.agent_id] = str(thought or "").strip()
                except NotImplementedError as exc:
                    print(
                        "[ICCThinkingProtocol] Batch thinking is not supported by the configured LLM API/server.\n"
                        "  - Required: REST /generate endpoint that accepts a list for 'prompt'.\n"
                        "  - Hint: start vLLM via 'python -m vllm.entrypoints.api_server' (REST).\n"
                        f"  - Details: {exc}"
                    )
                    # Do not silently fall back; surface the problem to the caller
                    raise
            else:
                print(
                    "[ICCThinkingProtocol] Batch thinking disabled or API lacks 'generate_text_batch'.\n"
                    "  - To enable batching, pass batch_thinking=True and use a REST vLLM server that exposes /generate."
                )
                raise RuntimeError("Batch thinking not available; refusing to fall back to sequential to avoid masking configuration issues.")

            # Phase 2: sequential JSON decisions using each agent's thought
            for agent in agents:
                turn_inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=shared_assignment,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                thought = thoughts_by_agent.get(agent.agent_id, "")
                decision = agent.decide_from_thought(
                    problem,
                    turn_inputs.context,
                    thought,
                    turn_inputs.images,
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
