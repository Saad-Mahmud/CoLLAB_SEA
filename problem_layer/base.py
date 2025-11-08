from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence


UtilityFn = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True)
class VariableSpec:
    """Specification for a single decision variable in the DCOP problem."""

    name: str
    domain: Sequence[Any]
    owner: str
    description: str = ""

    def validate(self, value: Any) -> Optional[str]:
        if value in self.domain:
            return None
        if all(isinstance(v, (int, float)) for v in self.domain) and isinstance(value, (int, float)):
            minimum = min(self.domain)
            maximum = max(self.domain)
            if minimum <= value <= maximum:
                return None
        return f"Value {value!r} is not permitted for variable {self.name!r}."


@dataclass(frozen=True)
class AgentSpec:
    """Metadata describing a collaborating agent."""

    agent_id: str
    name: str
    instruction: str


@dataclass
class FactorSpec:
    """
    Factor specification including a natural-language description and a utility function.
    """

    name: str
    scope: Sequence[str]
    description: str
    utility_fn: UtilityFn
    factor_type: str = "generic"
    weight: float = 1.0

    def evaluate(self, joint_assignment: Mapping[str, Any]) -> float:
        local_assignment: Dict[str, Any] = {}
        for var in self.scope:
            if var not in joint_assignment:
                raise KeyError(f"Missing assignment for variable {var!r} in factor {self.name!r}.")
            local_assignment[var] = joint_assignment[var]
        raw = self.utility_fn(local_assignment)
        return float(raw) * self.weight


class ProblemDefinition:
    """
    Defines a DCOP-style collaborative problem with JSON schemas and evaluators.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agents: Sequence[AgentSpec],
        variables: Sequence[VariableSpec],
        factors: Sequence[FactorSpec],
    ) -> None:
        if not agents:
            raise ValueError("At least one agent must be defined.")
        if not variables:
            raise ValueError("At least one variable must be defined.")
        self.name = name
        self.description = description
        self.agents: Dict[str, AgentSpec] = {agent.agent_id: agent for agent in agents}
        if len(self.agents) != len(agents):
            raise ValueError("Agent IDs must be unique.")
        self.variables: Dict[str, VariableSpec] = {var.name: var for var in variables}
        if len(self.variables) != len(variables):
            raise ValueError("Variable names must be unique.")

        for var in variables:
            if var.owner not in self.agents:
                raise ValueError(f"Variable {var.name!r} references unknown owner {var.owner!r}.")

        self.factors: List[FactorSpec] = list(factors)
        if not self.factors:
            raise ValueError("At least one factor must be defined for the problem.")

        self._agent_variables: Dict[str, List[VariableSpec]] = {agent_id: [] for agent_id in self.agents}
        for var in variables:
            self._agent_variables[var.owner].append(var)

        self._personal_factors: Dict[str, List[FactorSpec]] = self._collect_personal_preference_factors()

    # ---------- Schema helpers ----------
    def agent_schema(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self._agent_variables:
            raise KeyError(f"Unknown agent {agent_id!r}.")
        agent_vars = self._agent_variables[agent_id]
        properties: Dict[str, Any] = {}
        for spec in agent_vars:
            properties[spec.name] = _domain_to_schema(spec.domain, spec.description)
        return {
            "title": f"{self.name}::agent::{agent_id}",
            "type": "object",
            "properties": properties,
            "required": [spec.name for spec in agent_vars],
            "additionalProperties": False,
        }

    def joint_assignment_schema(self) -> Dict[str, Any]:
        properties = {
            name: _domain_to_schema(var.domain, var.description)
            for name, var in self.variables.items()
        }
        return {
            "title": f"{self.name}::joint_assignment",
            "type": "object",
            "properties": properties,
            "required": list(self.variables.keys()),
            "additionalProperties": False,
        }

    # ---------- Evaluation ----------
    def eval(self, joint_assignment: Mapping[str, Any]) -> Dict[str, Any]:
        validation_errors = self._validate_assignment(joint_assignment)
        if validation_errors:
            return {
                "problem": self.name,
                "valid": False,
                "violations": validation_errors,
                "factor_details": {},
                "total_utility": None,
                "joint_assignment": dict(joint_assignment),
            }

        factor_details: Dict[str, Dict[str, Any]] = {}
        violations: List[str] = []
        total_utility = 0.0

        for factor in self.factors:
            try:
                utility = factor.evaluate(joint_assignment)
            except Exception as exc:  # pylint: disable=broad-except
                violations.append(f"Factor {factor.name}: {exc}")
                continue
            factor_details[factor.name] = {
                "scope": list(factor.scope),
                "description": factor.description,
                "type": factor.factor_type,
                "utility": utility,
            }
            total_utility += utility

        return {
            "problem": self.name,
            "valid": not violations,
            "violations": violations,
            "factor_details": factor_details,
            "total_utility": total_utility if not violations else None,
            "joint_assignment": dict(joint_assignment),
        }

    # ---------- Introspection ----------
    def agent_instruction(self, agent_id: str) -> str:
        if agent_id not in self.agents:
            raise KeyError(f"Unknown agent {agent_id!r}.")
        return self.agents[agent_id].instruction

    def agent_variables(self, agent_id: str) -> Sequence[VariableSpec]:
        if agent_id not in self._agent_variables:
            raise KeyError(f"Unknown agent {agent_id!r}.")
        return tuple(self._agent_variables[agent_id])

    def personal_preference_factors(self, agent_id: str) -> Sequence[FactorSpec]:
        if agent_id not in self._personal_factors:
            raise KeyError(f"Unknown agent {agent_id!r}.")
        return tuple(self._personal_factors[agent_id])

    def list_factors(self) -> Sequence[FactorSpec]:
        return tuple(self.factors)

    # ---------- Internal helpers ----------
    def _validate_assignment(self, joint_assignment: Mapping[str, Any]) -> List[str]:
        errors: List[str] = []
        for name, spec in self.variables.items():
            if name not in joint_assignment:
                errors.append(f"Missing value for variable {name!r}.")
                continue
            value = joint_assignment[name]
            validation_error = spec.validate(value)
            if validation_error:
                errors.append(validation_error)
        for name in joint_assignment:
            if name not in self.variables:
                errors.append(f"Unexpected variable {name!r} in assignment.")
        return errors

    def _collect_personal_preference_factors(self) -> Dict[str, List[FactorSpec]]:
        per_agent: Dict[str, List[FactorSpec]] = {agent_id: [] for agent_id in self.agents}
        for factor in self.factors:
            if factor.factor_type != "personal_preference":
                continue
            if len(factor.scope) != 1:
                raise ValueError(
                    f"Personal preference factor {factor.name!r} must be unary."
                )
            variable_name = factor.scope[0]
            if variable_name not in self.variables:
                raise ValueError(
                    f"Factor {factor.name!r} references unknown variable {variable_name!r}."
                )
            owner = self.variables[variable_name].owner
            per_agent[owner].append(factor)

        missing: List[str] = [
            agent_id for agent_id, facs in per_agent.items() if not facs
        ]
        if missing:
            raise ValueError(
                f"Agents without personal preference factors: {', '.join(missing)}"
            )
        return per_agent


def _domain_to_schema(domain: Sequence[Any], description: str = "") -> Dict[str, Any]:
    if not domain:
        raise ValueError("Domain must contain at least one value.")
    unique_values = list(dict.fromkeys(domain))
    if all(isinstance(v, bool) for v in unique_values):
        schema: Dict[str, Any] = {"type": "boolean"}
    elif all(isinstance(v, int) for v in unique_values):
        schema = {"type": "integer"}
        if _is_contiguous(unique_values):
            schema["minimum"] = min(unique_values)
            schema["maximum"] = max(unique_values)
        else:
            schema["enum"] = unique_values
    elif all(isinstance(v, (int, float)) for v in unique_values):
        schema = {
            "type": "number",
            "minimum": min(unique_values),
            "maximum": max(unique_values),
        }
    elif all(isinstance(v, str) for v in unique_values):
        schema = {"type": "string", "enum": unique_values}
    else:
        schema = {"enum": unique_values}
    if description:
        schema["description"] = description
    return schema


def _is_contiguous(values: Sequence[int]) -> bool:
    sorted_values = sorted(set(values))
    return sorted_values == list(range(sorted_values[0], sorted_values[-1] + 1))


class ProblemRegistry:
    """Lightweight registry to access problems by name."""

    def __init__(self) -> None:
        self._registry: Dict[str, ProblemDefinition] = {}

    def register(self, problem: ProblemDefinition) -> None:
        if problem.name in self._registry:
            raise ValueError(f"Problem {problem.name!r} already registered.")
        self._registry[problem.name] = problem

    def get(self, name: str) -> ProblemDefinition:
        if name not in self._registry:
            raise KeyError(f"Problem {name!r} is not registered.")
        return self._registry[name]

    def names(self) -> Sequence[str]:
        return tuple(self._registry.keys())

    def problems(self) -> Sequence[ProblemDefinition]:
        return tuple(self._registry.values())
