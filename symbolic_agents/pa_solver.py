from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class SolverConfig:
    tries: int = 100
    steps_per_try: Optional[int] = None
    temperature: float = 1.5
    cooling: float = 0.95
    rng_seed: Optional[int] = None


@dataclass
class SolveResult:
    instance_dir: Path
    best_assignment: Dict[str, int]
    total_utility: float
    min_utility: float
    max_utility: float
    evaluation: Dict[str, Any]


def _load_pickled_instance(instance_dir: Path) -> Any:
    pkls = sorted(instance_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No pickle instance found in {instance_dir}")
    with open(pkls[0], "rb") as pf:
        return pickle.load(pf)


def _max_possible_utility(problem) -> float:
    total = 0.0
    for factor in problem.list_factors():
        total += 1.0 if len(factor.scope) == 1 else 2.0
    return total


def _random_assignment(domains: Mapping[str, Iterable[int]], rng: random.Random) -> Dict[str, int]:
    return {name: rng.choice(domain) for name, domain in domains.items()}


def _score(problem, assignment: Mapping[str, int]) -> float:
    result = problem.eval(assignment)
    total = result.get("total_utility")
    if total is None:
        return float("-inf")
    return float(total)


def solve_instance(instance_dir: Path, config: SolverConfig) -> SolveResult:
    instance_dir = instance_dir.resolve()
    pa_instance = _load_pickled_instance(instance_dir)
    problem = pa_instance.problem
    domains = {name: tuple(spec.domain) for name, spec in problem.variables.items()}
    rng = random.Random(config.rng_seed if config.rng_seed is not None else hash((instance_dir, config.tries)))

    steps_per_try = config.steps_per_try or max(50, len(domains) * 10)
    best_assignment: Optional[Dict[str, int]] = None
    best_score = float("-inf")

    for _ in range(config.tries):
        current = _random_assignment(domains, rng)
        current_score = _score(problem, current)
        temperature = config.temperature

        for _step in range(steps_per_try):
            var = rng.choice(list(domains.keys()))
            domain = domains[var]
            new_value = rng.choice(domain)
            if new_value == current[var]:
                continue
            old_value = current[var]
            current[var] = new_value
            new_score = _score(problem, current)
            delta = new_score - current_score
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-6)):
                current_score = new_score
            else:
                current[var] = old_value
            temperature *= config.cooling

            if current_score > best_score:
                best_score = current_score
                best_assignment = dict(current)

        if current_score > best_score:
            best_score = current_score
            best_assignment = dict(current)

    if best_assignment is None:
        raise RuntimeError("Solver was unable to produce any assignment.")

    evaluation = problem.eval(best_assignment)
    total_utility = float(evaluation.get("total_utility", 0.0) or 0.0)
    min_utility = float(evaluation.get("min_utility", 0.0) or 0.0)
    max_utility_val = evaluation.get("max_utility")
    if max_utility_val is None:
        max_utility_val = getattr(pa_instance, "max_utility", None)
    if max_utility_val is None:
        max_utility_val = _max_possible_utility(problem)
    max_utility = float(max_utility_val)

    return SolveResult(
        instance_dir=instance_dir,
        best_assignment=best_assignment,
        total_utility=total_utility,
        min_utility=min_utility,
        max_utility=max_utility,
        evaluation=evaluation,
    )

