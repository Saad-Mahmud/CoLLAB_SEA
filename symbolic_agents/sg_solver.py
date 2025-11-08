from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SolverConfig:
    tries: int = 30
    steps_per_try: Optional[int] = None
    temperature: float = 1.0
    cooling: float = 0.98
    rng_seed: Optional[int] = None


@dataclass
class SolveResult:
    instance_dir: Path
    best_assignment: Dict[str, str]
    total_utility: float
    min_utility: float
    max_utility: float
    evaluation: Dict[str, float]


def _load_instance(instance_dir: Path):
    pkls = sorted(instance_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(f"No smart-grid instance found in {instance_dir}")
    with open(pkls[0], "rb") as pf:
        return pickle.load(pf)


def _random_assignment(domains: Mapping[str, Sequence[str]], rng: random.Random) -> Dict[str, str]:
    return {name: rng.choice(list(domain)) for name, domain in domains.items()}


def _score(problem, assignment: Mapping[str, str]) -> float:
    result = problem.eval(assignment)
    total = result.get("total_utility")
    if total is None:
        return float("-inf")
    return float(total)


def solve_instance(instance_dir: Path, config: SolverConfig) -> SolveResult:
    instance_dir = instance_dir.resolve()
    sg_instance = _load_instance(instance_dir)
    problem = sg_instance.problem
    domains = {name: tuple(spec.domain) for name, spec in problem.variables.items()}
    variables = list(domains.keys())
    rng = random.Random(config.rng_seed if config.rng_seed is not None else hash((instance_dir, config.tries)))

    steps_per_try = config.steps_per_try or max(80, len(variables) * 12)
    best_assignment: Optional[Dict[str, str]] = None
    best_score = float("-inf")

    for _ in range(config.tries):
        current = {name: rng.choice(domains[name]) for name in variables}
        current_score = _score(problem, current)
        temperature = config.temperature

        for _step in range(steps_per_try):
            var = rng.choice(variables)
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
        raise RuntimeError("Solver failed to find an assignment")

    evaluation = problem.eval(best_assignment)
    total_utility = float(evaluation.get("total_utility", 0.0) or 0.0)
    min_utility = float(evaluation.get("min_utility", sg_instance.min_utility))
    max_utility = float(evaluation.get("max_utility", sg_instance.max_utility))

    return SolveResult(
        instance_dir=instance_dir,
        best_assignment=best_assignment,
        total_utility=total_utility,
        min_utility=min_utility,
        max_utility=max_utility,
        evaluation=evaluation,
    )

