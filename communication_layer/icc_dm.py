from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from .algorithm import DecisionAlgorithm, ICCDecisionOnlyAlgorithm
from agent_layer import CollaborativeAgent
from problem_layer import ProblemDefinition

from .base import CommunicationProtocol, ProtocolResult, ProtocolTurn


class ICCDecisionOnlyProtocol(CommunicationProtocol):
    """
    Iterative Constraint Communication (decision-only variant):
    - Each round, agents broadcast a single JSON decision that adheres to their schema.
    - The shared assignment is updated in-place after every agent turn.
    - Neighbours receive the latest decision to inform subsequent turns.
    """

    name = "ICC_DM"

    def __init__(
        self,
        *,
        algorithm: Optional[DecisionAlgorithm] = None,
        max_rounds: int = 1,
        decision_max_tokens: int = 512,
        decision_temperature: float = 0.0,
        show_neighbour_assignments: bool = True,
        show_factor_information: bool = False,
    ) -> None:
        self.algorithm = algorithm or ICCDecisionOnlyAlgorithm()
        self.max_rounds = max_rounds
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

        # Set protocol neighbours from the factor graph: two agents are
        # neighbours iff they share at least one factor (i.e., a factor whose
        # scope contains variables owned by both agents).
        # Build variable -> owner map
        var_owner = {name: spec.owner for name, spec in problem.variables.items()}
        # Initialise adjacency
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
        # Apply computed neighbour sets
        for agent in agents:
            agent.set_neighbours(sorted(adjacency.get(agent.agent_id, set())))

        for round_index in range(1, total_rounds + 1):
            for agent in agents:
                turn_inputs = self.algorithm.prepare_agent_inputs(
                    agent=agent,
                    problem=problem,
                    round_index=round_index,
                    shared_assignment=shared_assignment,
                    protocol_name=self.name,
                    static_context=static_context_payload,
                )
                context = turn_inputs.context
                decision = agent.decide(
                    problem,
                    context,
                    images=turn_inputs.images,
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
