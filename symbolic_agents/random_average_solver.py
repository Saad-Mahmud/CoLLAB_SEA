from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the random-average baseline solver."""

    tries: int = 128
    rng_seed: Optional[int] = None


@dataclass
class SolveResult:
    """Result produced by the random-average solver."""

    instance_dir: Path
    total_utility: float
    min_utility: Optional[float]
    max_utility: Optional[float]
    samples: int
    evaluation: Dict[str, Any]
    best_assignment: Optional[Dict[str, Any]] = None


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
    Sample random assignments and report the average utility instead of the best utility.
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

    sum_total = 0.0
    valid_samples = 0
    min_utility: Optional[float] = None
    max_utility: Optional[float] = None

    for _ in range(config.tries):
        candidate = _sample_assignment(domains, rng)
        evaluation = problem.eval(candidate)
        total_val = evaluation.get("total_utility")
        if total_val is None:
            continue
        total_float = float(total_val)
        sum_total += total_float
        valid_samples += 1

        if min_utility is None and evaluation.get("min_utility") is not None:
            min_utility = float(evaluation["min_utility"])
        if max_utility is None and evaluation.get("max_utility") is not None:
            max_utility = float(evaluation["max_utility"])

    if valid_samples == 0:
        raise RuntimeError("Random-average solver failed to obtain any valid assignment.")

    if min_utility is None:
        min_attr = getattr(instance, "min_utility", None)
        min_utility = float(min_attr) if min_attr is not None else 0.0
    if max_utility is None:
        estimated_max = _estimate_max_utility(instance, problem)
        max_utility = float(estimated_max) if estimated_max is not None else None

    average_total = sum_total / float(valid_samples)
    evaluation_payload: Dict[str, Any] = {
        "samples": valid_samples,
        "average_total_utility": average_total,
        "min_utility": min_utility,
        "max_utility": max_utility,
    }

    return SolveResult(
        instance_dir=instance_dir,
        total_utility=average_total,
        min_utility=min_utility,
        max_utility=max_utility,
        samples=valid_samples,
        evaluation=evaluation_payload,
    )

