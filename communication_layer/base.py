from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from agent_layer import CollaborativeAgent
from problem_layer import ProblemDefinition


@dataclass
class ProtocolTurn:
    round_index: int
    agent_id: str
    decision: Dict[str, Any]
    prompt: Optional[str]
    # Optional natural-communication artifacts captured for dumps
    broadcast_prompt: Optional[str] = None
    broadcast_message: Optional[str] = None
    joint_assignment: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None


@dataclass
class ProtocolResult:
    protocol: str
    turns: List[ProtocolTurn]
    final_assignment: Dict[str, Any]
    evaluation: Optional[Dict[str, Any]]


class CommunicationProtocol:
    """
    Base communication protocol that coordinates agents and accumulates their decisions.
    """

    name = "base_protocol"

    def run(
        self,
        problem: ProblemDefinition,
        agents: Sequence[CollaborativeAgent],
        *,
        rounds: int = 1,
        initial_assignment: Optional[Mapping[str, Any]] = None,
    ) -> ProtocolResult:
        raise NotImplementedError
