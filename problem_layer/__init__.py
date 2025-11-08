from .base import AgentSpec, FactorSpec, ProblemDefinition, ProblemRegistry, VariableSpec
from . import meeting_scheduling, personal_assistant, smart_grid

__all__ = [
    "AgentSpec",
    "FactorSpec",
    "ProblemDefinition",
    "ProblemRegistry",
    "VariableSpec",
    "MEETING_SCHEDULING",
    "PERSONAL_ASSISTANT",
    "SMART_GRID",
    "load_default_registry",
]

MEETING_SCHEDULING = meeting_scheduling.PROBLEM
PERSONAL_ASSISTANT = personal_assistant.PROBLEM
SMART_GRID = smart_grid.PROBLEM


def load_default_registry() -> ProblemRegistry:
    registry = ProblemRegistry()
    registry.register(MEETING_SCHEDULING)
    registry.register(PERSONAL_ASSISTANT)
    registry.register(SMART_GRID)
    return registry

