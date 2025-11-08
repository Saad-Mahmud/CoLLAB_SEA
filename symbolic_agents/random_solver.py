from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the random sampling solver."""

    tries: int = 128
    rng_seed: Optional[int] = None


@dataclass
class SolveResult:
    """Result of running the random sampling solver on an instance."""

    instance_dir: Path
    best_assignment: Dict[str, Any]
    total_utility: float
    min_utility: Optional[float]
    max_utility: Optional[float]
    evaluation: Dict[str, Any]


def _load_pickled_instance(instance_dir: Path) -> Any:
    pkls = sorted(instance_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No pickled instance found in {instance_dir}")
    with open(pkls[0], "rb") as pf:
        return pickle.load(pf)


def _extract_domains(problem) -> Dict[str, Sequence[Any]]:
    return {name: tuple(spec.domain) for name, spec in problem.variables.items()}


def _sample_assignment(domains: Mapping[str, Sequence[Any]], rng: random.Random) -> Dict[str, Any]:
    return {name: rng.choice(domain) for name, domain in domains.items()}


def _score(problem, assignment: Mapping[str, Any]) -> float:
    result = problem.eval(assignment)
    total = result.get("total_utility")
    if total is None:
        return float("-inf"), result
    return float(total), result


def _estimate_max_utility(instance: Any, problem) -> Optional[float]:
    for attr in ("max_utility", "MAX_UTILITY", "maxUtility"):
        value = getattr(instance, attr, None)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    if getattr(problem, "name", "") == "personal_assistant":
        total = 0.0
        for factor in problem.list_factors():
            total += 1.0 if len(factor.scope) == 1 else 2.0
        return float(total)
    return None


def solve_instance(instance_dir: Path, config: SolverConfig) -> SolveResult:
    """
    Solve a DCOP instance by sampling random assignments and keeping the best one.
    Works across personal assistant, meeting scheduling, and smart grid problems.
    """
    instance_dir = instance_dir.resolve()
    instance = _load_pickled_instance(instance_dir)
    problem = instance.problem
    domains = _extract_domains(problem)
    if not domains:
        raise ValueError("Problem has no variables to assign.")

    rng_seed = config.rng_seed if config.rng_seed is not None else hash((instance_dir, config.tries))
    rng = random.Random(rng_seed)

    best_assignment: Optional[Dict[str, Any]] = None
    best_score = float("-inf")
    best_evaluation: Optional[Dict[str, Any]] = None

    for _ in range(config.tries):
        candidate = _sample_assignment(domains, rng)
        score, evaluation = _score(problem, candidate)
        if best_assignment is None or score > best_score:
            best_score = score
            best_assignment = dict(candidate)
            best_evaluation = evaluation

    if best_assignment is None or best_evaluation is None:
        raise RuntimeError("Random solver failed to evaluate any assignment.")

    total_utility = float(best_evaluation.get("total_utility", 0.0) or 0.0)
    min_utility_val = best_evaluation.get("min_utility")
    if min_utility_val is None:
        min_utility_val = getattr(instance, "min_utility", 0.0)
    min_utility = float(min_utility_val) if min_utility_val is not None else None
    max_utility_val = best_evaluation.get("max_utility")
    if max_utility_val is None:
        max_utility_val = _estimate_max_utility(instance, problem)
    max_utility = float(max_utility_val) if max_utility_val is not None else None

    return SolveResult(
        instance_dir=instance_dir,
        best_assignment=best_assignment,
        total_utility=total_utility,
        min_utility=min_utility,
        max_utility=max_utility,
        evaluation=best_evaluation,
    )

