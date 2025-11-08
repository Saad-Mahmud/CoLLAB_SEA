#!/usr/bin/env python3
from __future__ import annotations

"""
Example 01 — Run multiple symbolic baselines on a saved instance

What it does
- Reads a previously generated instance directory (from example00_generate.py)
- Detects problem type and runs three symbolic solvers:
  1) problem-specific solver (pa/ms/sg)
  2) random solver
  3) random-average solver
- Prints per-solver totals, bounds, ratio and runtime, plus basic instance stats.

No LLM server or API is needed for this example.
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _ratio(total: float, min_u: Optional[float], max_u: Optional[float]) -> Optional[float]:
    if max_u is None:
        return None
    mv = 0.0 if min_u is None else float(min_u)
    denom = float(max_u) - mv
    if abs(denom) < 1e-9:
        return None
    return max(0.0, min(1.0, (float(total) - mv) / denom))


def run_symbolic_all(problem: str, instance_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Run problem-specific, random, and random-average solvers and collect stats."""
    results: Dict[str, Dict[str, Any]] = {}

    # 1) Problem-specific
    if problem == "personal_assistant":
        from symbolic_agents import pa_solver as solver
        cfg = solver.SolverConfig(tries=100, steps_per_try=None, temperature=1.5, cooling=0.95, rng_seed=123)
        label = "problem_specific"
    elif problem == "meeting_scheduling":
        from symbolic_agents import ms_solver as solver
        cfg = solver.SolverConfig(tries=30, steps_per_try=300, temperature=1.0, cooling=0.97, rng_seed=321)
        label = "problem_specific"
    elif problem == "smart_grid":
        from symbolic_agents import sg_solver as solver
        cfg = solver.SolverConfig(tries=40, steps_per_try=400, temperature=0.9, cooling=0.97, rng_seed=123)
        label = "problem_specific"
    else:
        raise SystemExit(f"Unknown problem: {problem}")
    t0 = time.perf_counter()
    res = solver.solve_instance(instance_dir, cfg)
    t1 = time.perf_counter()
    results[label] = {
        "total": float(res.total_utility),
        "min": float(getattr(res, "min_utility", 0.0) or 0.0),
        "max": float(getattr(res, "max_utility", 0.0) or 0.0),
        "time_s": t1 - t0,
    }

    # 2) Random solver
    from symbolic_agents import random_solver as rsolver
    rcfg = rsolver.SolverConfig(tries=200, rng_seed=42)
    t0 = time.perf_counter()
    rres = rsolver.solve_instance(instance_dir, rcfg)
    t1 = time.perf_counter()
    results["random"] = {
        "total": float(rres.total_utility),
        "min": float(getattr(rres, "min_utility", 0.0) or 0.0),
        "max": float(getattr(rres, "max_utility", 0.0) or 0.0),
        "time_s": t1 - t0,
    }

    # 3) Random-average solver
    from symbolic_agents import random_average_solver as rasolver
    racfg = rasolver.SolverConfig(tries=300, rng_seed=99)
    t0 = time.perf_counter()
    rares = rasolver.solve_instance(instance_dir, racfg)
    t1 = time.perf_counter()
    results["random_average"] = {
        "total": float(rares.total_utility),
        "min": float(getattr(rares, "min_utility", 0.0) or 0.0),
        "max": float(getattr(rares, "max_utility", 0.0) or 0.0),
        "time_s": t1 - t0,
    }

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Example 01: Run multiple symbolic baselines on a saved instance.")
    ap.add_argument("instance_dir", type=Path, help="Directory containing the saved instance JSON/PKL (from example00_generate.py).")
    args = ap.parse_args()
    inst_dir: Path = args.instance_dir
    if not inst_dir.exists():
        raise SystemExit(f"[Error] instance-dir not found: {inst_dir}")

    # Detect problem by reading the pickle
    pkls = sorted(inst_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"[Error] No pickle found under {inst_dir}")
    with open(pkls[0], "rb") as pf:
        obj = pickle.load(pf)
    problem_name = getattr(getattr(obj, "problem", None), "name", None) or ""
    if problem_name not in ("personal_assistant", "meeting_scheduling", "smart_grid"):
        raise SystemExit(f"[Error] Unable to detect problem type from pickle: {pkls[0]}")

    prob = getattr(obj, "problem", None)
    try:
        n_agents = len(getattr(prob, "agents", {}))
        n_vars = len(getattr(prob, "variables", {}))
        n_factors = len(list(prob.list_factors())) if prob else 0
    except Exception:
        n_agents = n_vars = n_factors = 0

    print(f"[Example01] Symbolic baselines | problem={problem_name} | instance_dir={inst_dir}")
    print(f"  - stats: agents={n_agents} | variables={n_vars} | factors={n_factors}")
    all_out = run_symbolic_all(problem_name, inst_dir)
    for solver_name, metrics in all_out.items():
        r = _ratio(metrics.get("total"), metrics.get("min"), metrics.get("max"))
        ratio_txt = f"{r:.3f}" if r is not None else "N/A"
        print(
            f"  -> solver={solver_name:>16} | total={metrics['total']:.2f} | min={metrics['min']:.2f} | max={metrics['max']:.2f} | ratio={ratio_txt} | time={metrics['time_s']:.2f}s"
        )


if __name__ == "__main__":
    main()
