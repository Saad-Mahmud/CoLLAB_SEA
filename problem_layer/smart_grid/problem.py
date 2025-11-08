from __future__ import annotations

import json
import pickle
import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Pillow is optional but preferred for chart generation
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

from ..base import AgentSpec, FactorSpec, ProblemDefinition, VariableSpec

"""
Core domain and configuration
-----------------------------
This module defines the Smart Grid coordination problem used in experiments:
- Renewable sources are global shared resources (e.g., one Solar, one Hydro),
  each with an hourly capacity profile across a 24-hour timeline and a list of
  client agents that can draw from it.
- Each agent owns machines that must run within fixed time windows and draw a
  uniform load while running.
- The decision is: for every machine, select which connected shared source
  powers it.
- Utility rewards assignments that avoid exceeding source capacity; any
  overflow on a shared source is penalised by a coordination factor spanning
  all machines that can use that source.

The code is organised as follows (top to bottom):
1) Core constants, dataclasses, and config
2) Random helpers and sampling (names, capacity profiles)
3) Instance generation (global sources, agent connectivity, machines, demand shaping)
4) Visualisation and instruction composition
5) Utility factors and ProblemDefinition assembly (shared-source overflow)
6) Instance summary and persistence
7) Public instance construction helpers and module-level defaults
8) eval() helper that augments ProblemDefinition.eval with overflow analysis
"""


SUSTAINABLE_TYPES = ["Solar", "Wind", "Hydro", "Geothermal", "Biomass"]
MACHINE_TYPES = [
    "HVAC unit",
    "Water heater",
    "EV charger",
    "CNC mill",
    "Refrigeration chain",
    "Lighting bank",
    "Server rack",
    "Assembly line",
    "Kiln",
    "Irrigation pump",
]


class InstructionMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    kind: str
    capacity: Tuple[float, ...]
    clients: Tuple[str, ...]


@dataclass(frozen=True)
class MachineSpec:
    machine_id: str
    owner: str
    label: str
    energy: float
    start: int
    end: int

    @property
    def duration(self) -> int:
        return max(1, self.end - self.start)


@dataclass(frozen=True)
class SmartGridConfig:
    num_agents: int = 6
    min_sources_per_agent: int = 2
    max_sources_per_agent: int = 3
    min_machines_per_agent: int = 3
    max_machines_per_agent: int = 6
    timeline_length: int = 24
    instruction_mode: InstructionMode = InstructionMode.TEXT
    include_charts: bool = False
    rng_seed: int = 900
    output_stem: str = "smart_grid_instance"

    def validate(self) -> None:
        """Basic sanity checks for configuration values."""
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")
        if self.timeline_length < 1:
            raise ValueError("timeline_length must be >= 1")


@dataclass
class SmartGridInstance:
    config: SmartGridConfig
    problem: ProblemDefinition
    timeline_length: int
    sources: Dict[str, SourceSpec]
    machines: Dict[str, MachineSpec]
    agent_sources: Dict[str, List[str]]
    machine_powers: Dict[str, float]
    instructions: Dict[str, str]
    charts: Dict[str, Optional[str]]
    min_utility: float
    max_utility: float
    json_path: Optional[Path] = None
    pickle_path: Optional[Path] = None


def _rng(seed: int):
    """Deterministic RNG helper (separate import to avoid global state)."""
    import random

    return random.Random(seed)


def _pick_names(count: int) -> List[str]:
    """Return themed site names, cycling with suffixes if needed."""
    base = [
        "Aurora Industries",
        "Beacon Works",
        "Cascade Labs",
        "Delta Homes",
        "Evergreen Coop",
        "Frontier Foods",
        "Gridwise Corp",
        "Harbor House",
        "Innova Plant",
        "Juniper Estates",
        "Keystone Foundry",
        "Lumen Tech",
        "Metro Lofts",
        "Northfield Farm",
        "Orchid Villas",
        "Pacific Metal",
        "Quartz Systems",
        "Riverline Plastics",
        "Summit Towers",
        "Tidal Labs",
        "Urban Makers",
        "Valley Utility",
        "Willow Workshop",
        "Zenith Motors",
    ]
    names = []
    for i in range(count):
        base_name = base[i % len(base)]
        suffix = "" if i < len(base) else f"_{i // len(base) + 1}"
        names.append(base_name + suffix)
    return names


def _sample_capacity_profile(kind: str, timeline_length: int, rng) -> Tuple[float, ...]:
    """
    Sample an hourly capacity profile for a given source kind.
    Each kind follows a simple generative pattern with noise to add variety.
    """
    if timeline_length <= 0:
        return (0.0,)

    profile: List[float] = []
    hours = range(timeline_length)

    if kind == "Solar":
        sunrise = rng.randint(5, 7)
        sunset = rng.randint(17, 19)
        daylight = max(1, sunset - sunrise)
        peak = rng.uniform(5.0, 9.0)

        def pattern(h: int) -> float:
            if h < sunrise or h >= sunset:
                return 0.0
            phase = (h - sunrise) / daylight
            return peak * math.sin(math.pi * phase)

    elif kind == "Wind":
        base = rng.uniform(4.5, 7.5)
        nocturnal_bias = rng.uniform(0.3, 0.6)
        phase = rng.uniform(0, 2 * math.pi)

        def pattern(h: int) -> float:
            norm = (h % timeline_length) / timeline_length
            diurnal = 0.5 + nocturnal_bias * (1 - math.cos(2 * math.pi * norm))
            seasonal = 0.4 + 0.6 * abs(math.sin(2 * math.pi * norm + phase))
            return base * diurnal * seasonal

    elif kind == "Hydro":
        base = rng.uniform(5.5, 8.5)
        tidal_period = rng.uniform(10.0, 14.0)
        phase = rng.uniform(0, 2 * math.pi)
        runoff = rng.uniform(0.1, 0.3)

        def pattern(h: int) -> float:
            tidal = 0.6 + 0.3 * math.sin((h / tidal_period) * 2 * math.pi + phase)
            seasonal = 0.8 + runoff * math.sin((h / max(1, timeline_length)) * 2 * math.pi)
            return base * tidal * seasonal

    elif kind == "Geothermal":
        base = rng.uniform(4.0, 6.0)

        def pattern(_: int) -> float:
            return base

    elif kind == "Biomass":
        base = rng.uniform(3.5, 6.5)
        peak_center = rng.uniform(0.65, 0.85)
        spread = rng.uniform(0.08, 0.15)

        def pattern(h: int) -> float:
            norm = (h % timeline_length) / timeline_length
            evening_peak = math.exp(-((norm - peak_center) ** 2) / (2 * spread**2))
            baseline = 0.6 + 0.4 * evening_peak
            return base * baseline

    else:
        base = rng.uniform(4.0, 7.0)

        def pattern(_: int) -> float:
            return base

    for hour in hours:
        value = pattern(hour)
        noise = rng.uniform(-0.5, 0.5)
        profile.append(max(0.0, value + noise))
    return tuple(profile)


def _generate_sources(
    config: SmartGridConfig,
    agents: Sequence[str],
    rng,
) -> Tuple[Dict[str, SourceSpec], Dict[str, List[str]]]:
    """
    Create a set of global shared sources (one per kind) and a sparse
    connectivity mapping from agents to the sources they can use.

    Returns:
      - sources: mapping source_id -> SourceSpec with clients listing connected agents
      - agent_sources: mapping agent_id -> list of connected source_ids
    """
    # Create one shared source per sustainable kind with a base capacity profile
    source_ids: List[str] = []
    base_sources: Dict[str, SourceSpec] = {}
    for kind in SUSTAINABLE_TYPES:
        source_id = f"src_{kind}"
        capacity_profile = _sample_capacity_profile(kind, config.timeline_length, rng)
        base_sources[source_id] = SourceSpec(
            source_id=source_id,
            kind=kind,
            capacity=tuple(capacity_profile),
            clients=tuple(),
        )
        source_ids.append(source_id)

    # Connect each agent to a subset of sources
    agent_sources: Dict[str, List[str]] = {agent: [] for agent in agents}
    clients: Dict[str, List[str]] = {sid: [] for sid in source_ids}

    for agent in agents:
        k_min = max(1, min(config.min_sources_per_agent, len(source_ids)))
        k_max = max(k_min, min(config.max_sources_per_agent, len(source_ids)))
        k = rng.randint(k_min, k_max)
        connected = rng.sample(source_ids, k)
        agent_sources[agent] = list(connected)
        for sid in connected:
            clients[sid].append(agent)

    # Ensure every source has at least one client by assigning a random agent if needed
    for sid in source_ids:
        if not clients[sid] and agents:
            agent = rng.choice(list(agents))
            clients[sid].append(agent)
            if sid not in agent_sources[agent]:
                agent_sources[agent].append(sid)

    # Finalise sources with client lists
    sources: Dict[str, SourceSpec] = {}
    for sid, spec in base_sources.items():
        sources[sid] = SourceSpec(
            source_id=spec.source_id,
            kind=spec.kind,
            capacity=spec.capacity,
            clients=tuple(clients[sid]),
        )

    return sources, agent_sources


def _generate_machines(config: SmartGridConfig, agents: Sequence[str], agent_sources: Mapping[str, List[str]], rng) -> Dict[str, MachineSpec]:
    """
    Create machines per agent with random windows and total energy.
    Ensures every agent owns at least one machine.
    """
    machines: Dict[str, MachineSpec] = {}
    machine_count = 0
    for agent in agents:
        count = rng.randint(config.min_machines_per_agent, config.max_machines_per_agent)
        for _ in range(count):
            machine_count += 1
            machine_id = f"machine_{machine_count:03d}"
            label = rng.choice(MACHINE_TYPES)
            duration = rng.randint(2, 6)
            start = rng.randint(0, max(0, config.timeline_length - duration))
            end = start + duration
            energy = rng.uniform(4.0, 12.0)
            machines[machine_id] = MachineSpec(
                machine_id=machine_id,
                owner=agent,
                label=label,
                energy=energy,
                start=start,
                end=end,
            )
    # ensure every agent owns at least one machine
    for agent in agents:
        if not any(m.owner == agent for m in machines.values()):
            machine_count += 1
            start = rng.randint(0, config.timeline_length - 3)
            end = start + 3
            machines[f"machine_{machine_count:03d}"] = MachineSpec(
                machine_id=f"machine_{machine_count:03d}",
                owner=agent,
                label=rng.choice(MACHINE_TYPES),
                energy=rng.uniform(4.0, 8.0),
                start=start,
                end=end,
            )
    return machines


def _machine_power(machine: MachineSpec) -> float:
    """Uniform hourly load (kW) implied by total energy and duration."""
    return machine.energy / machine.duration


def _compute_agent_hourly_demand(
    agents: Sequence[str],
    machines: Mapping[str, MachineSpec],
    machine_powers: Mapping[str, float],
    timeline_length: int,
) -> Dict[str, List[float]]:
    """
    Sum per-agent hourly demand induced by their machines running at uniform load.
    Used to shape shortages so sources do not trivially cover all demand.
    """
    demand = {agent: [0.0 for _ in range(timeline_length)] for agent in agents}
    for machine in machines.values():
        load = machine_powers[machine.machine_id]
        for t in range(machine.start, machine.end):
            if 0 <= t < timeline_length:
                demand[machine.owner][t] += load
    return demand


def _apply_capacity_shortage(
    sources: Mapping[str, SourceSpec],
    agent_sources: Mapping[str, List[str]],
    agent_hourly_demand: Mapping[str, Sequence[float]],
    rng,
    *,
    min_coverage: float = 0.88,
    max_coverage: float = 0.94,
) -> Dict[str, SourceSpec]:
    """
    Shape capacities for shared sources by scaling per-hour totals across all sources.
    Steps per hour t:
      - Compute base capacity for each source as its profile value scaled by the number of clients.
      - Compute total base capacity across all sources and total demand across all agents.
      - Target desired_total = coverage_ratio * total_demand.
      - If total_base > desired_total, downscale all sources proportionally; otherwise keep base.
    This preserves relative capacities and avoids making any single source trivially infeasible.
    """
    if not sources:
        return dict(sources)
    timeline_length = len(next(iter(sources.values())).capacity)
    coverage_ratio = rng.uniform(min_coverage, max_coverage)

    # Precompute base-scaled capacities per source
    base_scaled: Dict[str, List[float]] = {}
    for sid, src in sources.items():
        clients = max(1, len(src.clients))
        base = list(src.capacity)
        base_scaled[sid] = [
            (base[t] if t < len(base) else base[-1]) * clients for t in range(timeline_length)
        ]

    # Compute per-hour total demand across all agents
    total_demand: List[float] = [0.0 for _ in range(timeline_length)]
    for agent, seq in agent_hourly_demand.items():
        for t in range(min(len(seq), timeline_length)):
            total_demand[t] += seq[t]

    # Scale per hour if needed
    adjusted_caps: Dict[str, List[float]] = {sid: [0.0] * timeline_length for sid in sources.keys()}
    for t in range(timeline_length):
        base_total = sum(base_scaled[sid][t] for sid in sources.keys())
        desired_total = coverage_ratio * total_demand[t]
        if base_total > 0 and desired_total < base_total:
            scale = desired_total / max(base_total, 1e-9)
        else:
            scale = 1.0
        for sid in sources.keys():
            adjusted_caps[sid][t] = base_scaled[sid][t] * scale

    adjusted: Dict[str, SourceSpec] = {}
    for sid, src in sources.items():
        adjusted[sid] = SourceSpec(
            source_id=src.source_id,
            kind=src.kind,
            capacity=tuple(adjusted_caps[sid]),
            clients=tuple(src.clients),
        )
    return adjusted


def _generate_capacity_chart(
    agent_id: str,
    source_ids: Sequence[str],
    sources: Mapping[str, SourceSpec],
    output_dir: Path,
    timeline_length: int,
) -> Optional[Path]:
    """
    Draw a simple multi-line chart of hourly capacities for an agent's sources.
    Returns the saved image path or None if PIL is unavailable.
    """
    if Image is None or ImageDraw is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    legend_width = 160 if source_ids else 0
    height = 360
    margin = 50
    base_width = max(640, timeline_length * 24 + 2 * margin)
    width = base_width + legend_width
    chart_area_width = width - 2 * margin - legend_width
    if chart_area_width <= 0:
        chart_area_width = max(120, timeline_length * 10)
        width = chart_area_width + 2 * margin + legend_width
    chart_area_height = height - 2 * margin

    max_capacity = 0.0
    for source_id in source_ids:
        capacity = sources[source_id].capacity
        if capacity:
            max_capacity = max(max_capacity, max(capacity))
    if max_capacity <= 0:
        max_capacity = 1.0

    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)

    # Axes
    x0 = margin
    y0 = height - margin
    x1 = margin + chart_area_width
    y1 = margin
    draw.line((x0, y0, x1, y0), fill="black", width=2)
    draw.line((x0, y0, x0, y1), fill="black", width=2)

    # Ticks and labels
    step = chart_area_width / max(1, timeline_length - 1)
    font = None
    if ImageFont is not None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    for hour in range(timeline_length):
        x = x0 + hour * step
        draw.line((x, y0, x, y0 + 5), fill="black")
        label = str(hour)
        if font is not None:
            draw.text((x - 5, y0 + 8), label, fill="black", font=font)

    # Capacity lines
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    for idx, source_id in enumerate(source_ids):
        capacity = sources[source_id].capacity
        if not capacity:
            continue
        color = palette[idx % len(palette)]
        points: List[Tuple[float, float]] = []
        for hour in range(timeline_length):
            value = capacity[hour] if hour < len(capacity) else capacity[-1]
            x = x0 + hour * step
            y = y0 - (value / max_capacity) * chart_area_height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)

    # Legend
    legend_x = x1 + 20
    legend_y = margin
    for idx, source_id in enumerate(source_ids):
        color = palette[idx % len(palette)]
        draw.rectangle((legend_x, legend_y + idx * 18, legend_x + 12, legend_y + idx * 18 + 12), fill=color)
        if font is not None:
            draw.text((legend_x + 16, legend_y + idx * 18), source_id, fill="black", font=font)

    safe_name = agent_id.replace(" ", "_")
    chart_path = output_dir / f"{safe_name}_capacity.png"
    try:
        image.save(chart_path)
    except Exception:
        return None
    return chart_path


def _source_factor_utility(
    assignment: Mapping[str, str],
    *,
    source_id: str,
    machine_ids: Sequence[str],
    machines: Mapping[str, MachineSpec],
    machine_powers: Mapping[str, float],
    capacity: Sequence[float],
) -> float:
    """
    Coordination factor: negative overflow accumulated for a source.
    For each hour, sum machine loads assigned to the source; overflow over the
    capacity is penalised and aggregated across the timeline.
    """
    timeline = len(capacity)
    total_overflow = 0.0
    for t in range(timeline):
        demand = 0.0
        for machine_id in machine_ids:
            if assignment.get(machine_id) != source_id:
                continue
            machine = machines[machine_id]
            if machine.start <= t < machine.end:
                demand += machine_powers[machine_id]
        overflow = max(0.0, demand - capacity[t])
        total_overflow += overflow
    return -total_overflow


def _anchor_utility(assignment: Mapping[str, str], *, machine_id: str) -> float:
    """Neutral unary factor to ensure every machine appears in at least one factor."""
    assignment.get(machine_id, "")  # touch for safety
    return 0.0


def _build_problem(
    meetings: Dict[str, MachineSpec],
    sources: Dict[str, SourceSpec],
    agent_sources: Mapping[str, List[str]],
) -> Tuple[ProblemDefinition, Dict[str, List[str]]]:
    """
    Assemble variables and factors into a ProblemDefinition.
    - Variables: one categorical choice per machine over the agent's sources.
    - Factors: unary anchors + per-source overflow penalties (multi-scope).
    Returns the problem and a mapping agent_id -> list of their machine ids.
    """
    variables: List[VariableSpec] = []
    agent_machines: Dict[str, List[str]] = defaultdict(list)
    for machine in meetings.values():
        domain = agent_sources[machine.owner]
        variables.append(
            VariableSpec(
                name=machine.machine_id,
                domain=list(domain),
                owner=machine.owner,
                description=f"Assign {machine.label} to a power source",
            )
        )
        agent_machines[machine.owner].append(machine.machine_id)

    factors: List[FactorSpec] = []
    for machine in meetings.values():
        factors.append(
            FactorSpec(
                name=f"presence_{machine.machine_id}",
                scope=[machine.machine_id],
                description=f"Anchor factor for {machine.label}",
                utility_fn=partial(_anchor_utility, machine_id=machine.machine_id),
                factor_type="personal_preference",
            )
        )

    for source in sources.values():
        related_machines = [m_id for m_id, m in meetings.items() if m.owner in source.clients]
        factors.append(
            FactorSpec(
                name=f"overflow_{source.source_id}",
                scope=related_machines,
                description=f"Overflow penalty for shared {source.kind} source {source.source_id}",
                utility_fn=partial(
                    _source_factor_utility,
                    source_id=source.source_id,
                    machine_ids=tuple(related_machines),
                    machines=meetings,
                    machine_powers={mid: _machine_power(meetings[mid]) for mid in related_machines},
                    capacity=source.capacity,
                ),
                factor_type="coordination",
            )
        )

    agent_specs = [
        AgentSpec(
            agent_id=agent,
            name=agent,
            instruction="Allocate machines to sustainable sources to minimise main-grid draw.",
        )
        for agent in agent_sources.keys()
    ]

    problem = ProblemDefinition(
        name="smart_grid",
        description="Assign machines to shared renewable sources across 24h to minimise main-grid usage.",
        agents=agent_specs,
        variables=variables,
        factors=factors,
    )
    return problem, agent_machines


def _compose_instruction(
    agent: str,
    sources: Sequence[SourceSpec],
    machines: Sequence[MachineSpec],
    machine_powers: Mapping[str, float],
    timeline_length: int,
    chart_path: Optional[str],
) -> str:
    """
    Render natural-language instructions for an agent, listing sources, machine
    windows and loads, and highlighting the overflow objective. When charts are
    enabled, references the attached capacity image instead of inlining numbers.
    """
    lines = [
        "You manage energy assignments for this site across a 24-hour horizon.",
        "Sources are shared across multiple sites; other sites will also draw from these sources this round.",
        "Overflow pulls from the main grid and should be minimised.",
        "",
        "Available shared sources you can use:",
    ]
    if chart_path:
        for source in sources:
            client_count = len(source.clients)
            lines.append(
                f"- {source.source_id} ({source.kind}) — clients: {client_count} (including you). "
                f"Review the attached capacity chart for hourly limits."
            )
        lines.append("")
        lines.append("Capacity chart image attached; rely on it to read hourly source limits.")
    else:
        for source in sources:
            hourly = ", ".join(f"{value:.1f}" for value in source.capacity[:timeline_length])
            client_count = len(source.clients)
            lines.append(
                f"- {source.source_id} ({source.kind}) — clients: {client_count} (including you). "
                f"Hourly capacity (kW): [{hourly}] (shared)"
            )
    lines.append("")
    lines.append("Machines under your control:")
    for machine in machines:
        power = machine_powers[machine.machine_id]
        lines.append(
            f"- {machine.label} ({machine.machine_id}): window [{machine.start}, {machine.end}), "
            f"uniform load {power:.1f} kW"
        )
    lines.append("")
    lines.append("Overflow is attributed proportionally each hour on shared pools; choose sources to stay within capacity.")
    return "\n".join(lines)


def _instance_summary(instance: SmartGridInstance, agent_machines: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    """Build a JSON-serialisable snapshot of the instance for export/debugging."""
    problem = instance.problem
    agents_block = []
    for agent_id, spec in problem.agents.items():
        agents_block.append(
            {
                "id": agent_id,
                "name": spec.name,
                "instruction": instance.instructions[agent_id],
                "variables": list(agent_machines.get(agent_id, [])),
                "assets": {"image": instance.charts.get(agent_id)},
            }
        )

    variables_block = []
    for name, spec in problem.variables.items():
        variables_block.append(
            {
                "name": name,
                "owner": spec.owner,
                "domain": list(spec.domain),
                "description": spec.description,
            }
        )

    factors_block = []
    for factor in problem.list_factors():
        factors_block.append(
            {
                "name": factor.name,
                "scope": list(factor.scope),
                "description": factor.description,
                "type": factor.factor_type,
                "assets": {"image": None},
            }
        )

    sources_block = [
        {
            "id": source.source_id,
            "kind": source.kind,
            "capacity": list(source.capacity),
            "clients": list(source.clients),
        }
        for source in instance.sources.values()
    ]

    machines_block = [
        {
            "id": machine.machine_id,
            "owner": machine.owner,
            "label": machine.label,
            "start": machine.start,
            "end": machine.end,
            "energy": machine.energy,
        }
        for machine in instance.machines.values()
    ]

    return {
        "problem_name": problem.name,
        "instance_id": instance.config.output_stem,
        "instruction_mode": instance.config.instruction_mode.value,
        "metadata": {
            "num_agents": len(problem.agents),
            "num_variables": len(problem.variables),
            "num_factors": len(problem.list_factors()),
            "timeline_length": instance.timeline_length,
            "max_utility": instance.max_utility,
            "min_utility": instance.min_utility,
        },
        "agent_variable_map": {k: list(v) for k, v in agent_machines.items()},
        "agents": agents_block,
        "variables": variables_block,
        "factors": factors_block,
        "sources": sources_block,
        "machines": machines_block,
        "schemas": {
            "joint_assignment": problem.joint_assignment_schema(),
            "agents": {agent_id: problem.agent_schema(agent_id) for agent_id in problem.agents.keys()},
        },
    }


def _analyze_assignment(instance: SmartGridInstance, assignment: Mapping[str, str]) -> Dict[str, Any]:
    timeline = instance.timeline_length
    per_source_overflow: Dict[str, float] = defaultdict(float)
    per_agent_overflow: Dict[str, float] = defaultdict(float)

    for source in instance.sources.values():
        for t in range(timeline):
            demand = 0.0
            demand_by_agent: DefaultDict[str, float] = defaultdict(float)
            for machine_id, machine in instance.machines.items():
                if assignment.get(machine_id) != source.source_id:
                    continue
                if machine.start <= t < machine.end:
                    load = instance.machine_powers[machine_id]
                    demand += load
                    demand_by_agent[machine.owner] += load
            overflow = max(0.0, demand - source.capacity[t])
            if overflow > 0:
                per_source_overflow[source.source_id] += overflow
                if demand > 0:
                    for agent, load in demand_by_agent.items():
                        share = overflow * (load / demand)
                        per_agent_overflow[agent] += share

    total_overflow = sum(per_source_overflow.values())
    return {
        "total_overflow": total_overflow,
        "per_source_overflow": dict(per_source_overflow),
        "per_agent_overflow": dict(per_agent_overflow),
    }


def _build_instance(config: SmartGridConfig, *, save_dir: Optional[Path]) -> SmartGridInstance:
    """
    Generate sources and machines, shape capacities, compose instructions, and
    optionally persist the instance (JSON + pickle). Returns the in-memory
    SmartGridInstance with bounds and asset references.
    """
    config.validate()
    rng = _rng(config.rng_seed)
    agents = _pick_names(config.num_agents)
    sources, agent_sources = _generate_sources(config, agents, rng)
    machines = _generate_machines(config, agents, agent_sources, rng)
    machine_powers = {machine_id: _machine_power(machine) for machine_id, machine in machines.items()}
    agent_hourly_demand = _compute_agent_hourly_demand(agents, machines, machine_powers, config.timeline_length)
    sources = _apply_capacity_shortage(sources, agent_sources, agent_hourly_demand, rng)

    problem, agent_machines = _build_problem(machines, sources, agent_sources)

    chart_rel_paths: Dict[str, Optional[str]] = {agent: None for agent in agents}
    if config.instruction_mode is InstructionMode.IMAGE and config.include_charts and save_dir is not None:
        chart_dir = save_dir / "charts"
        for agent in agents:
            chart_path = _generate_capacity_chart(
                agent,
                agent_sources[agent],
                sources,
                chart_dir,
                config.timeline_length,
            )
            if chart_path is not None:
                chart_rel_paths[agent] = str(Path("charts") / chart_path.name)

    instructions = {}
    for agent in agents:
        instructions[agent] = _compose_instruction(
            agent,
            [sources[s_id] for s_id in agent_sources[agent]],
            [machines[m_id] for m_id in agent_machines.get(agent, [])],
            machine_powers,
            config.timeline_length,
            chart_rel_paths.get(agent),
        )
    problem = ProblemDefinition(
        name=problem.name,
        description=problem.description,
        agents=[
            AgentSpec(
                agent_id=spec.agent_id,
                name=spec.name,
                instruction=instructions[spec.agent_id],
            )
            for spec in problem.agents.values()
        ],
        variables=list(problem.variables.values()),
        factors=list(problem.list_factors()),
    )

    total_energy = sum(machine.energy for machine in machines.values())
    max_utility = 0.0
    min_utility = -total_energy

    instance = SmartGridInstance(
        config=config,
        problem=problem,
        timeline_length=config.timeline_length,
        sources=sources,
        machines=machines,
        agent_sources={k: list(v) for k, v in agent_sources.items()},
        machine_powers=machine_powers,
        instructions=instructions,
        charts=chart_rel_paths,
        min_utility=min_utility,
        max_utility=max_utility,
    )

    # Attach back-reference so the problem can provide factor summaries at runtime
    try:
        setattr(problem, "_smart_grid_instance", instance)
    except Exception:
        pass

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = _instance_summary(instance, agent_machines)
        json_path = save_dir / f"{config.output_stem}.json"
        pickle_path = save_dir / f"{config.output_stem}.pkl"
        json_path.write_text(json.dumps(summary, indent=2))
        with open(pickle_path, "wb") as pf:
            pickle.dump(instance, pf)
        instance.json_path = json_path
        instance.pickle_path = pickle_path

    return instance


def generate_instance(config: SmartGridConfig, output_dir: Path) -> SmartGridInstance:
    """Public entry-point to create and save a SmartGrid instance to a folder."""
    return _build_instance(config, save_dir=output_dir)


DEFAULT_CONFIG = SmartGridConfig()
DEFAULT_INSTANCE = _build_instance(DEFAULT_CONFIG, save_dir=None)

PROBLEM = DEFAULT_INSTANCE.problem
AGENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    agent_id: PROBLEM.agent_schema(agent_id) for agent_id in PROBLEM.agents.keys()
}
JOINT_ASSIGNMENT_SCHEMA: Dict[str, Any] = PROBLEM.joint_assignment_schema()
MAX_UTILITY = DEFAULT_INSTANCE.max_utility
MIN_UTILITY = DEFAULT_INSTANCE.min_utility


def eval(joint_assignment: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a joint assignment against the default problem and enrich the
    result with overflow analysis and explicit min/max utility bounds.
    """
    raw = PROBLEM.eval(joint_assignment)
    result = dict(raw)
    result["min_utility"] = MIN_UTILITY
    result["max_utility"] = MAX_UTILITY
    analysis = _analyze_assignment(DEFAULT_INSTANCE, joint_assignment)
    result["total_overflow"] = analysis["total_overflow"]
    result["per_source_overflow"] = analysis["per_source_overflow"]
    result["per_agent_overflow"] = analysis["per_agent_overflow"]
    if result.get("total_utility") is not None:
        result["total_utility"] = float(result["total_utility"])
    return result


def _compute_source_hourly_loads(instance: SmartGridInstance, assignment: Mapping[str, Any]) -> Dict[str, List[float]]:
    """Compute current hourly load per shared source based on a (partial) assignment."""
    timeline = instance.timeline_length
    loads: Dict[str, List[float]] = {sid: [0.0 for _ in range(timeline)] for sid in instance.sources.keys()}
    for machine_id, choice in assignment.items():
        if choice not in loads:
            continue
        m = instance.machines.get(machine_id)
        if not m:
            continue
        power = instance.machine_powers.get(machine_id, 0.0)
        for t in range(max(0, m.start), min(timeline, m.end)):
            loads[choice][t] += power
    return loads


def format_factor_information(
    *,
    problem: ProblemDefinition,
    agent_id: str,
    shared_assignment: Mapping[str, Any],
) -> Mapping[str, Any]:
    """
    Smart Grid factor information for prompts: per-source hourly current loads (shared demand),
    restricted to sources the agent can use.
    Returns a mapping with a title and list of preformatted lines for the agent prompt.
    """
    instance: Optional[SmartGridInstance] = getattr(problem, "_smart_grid_instance", None)  # type: ignore[assignment]
    if instance is None:
        return {}
    agent_sources = instance.agent_sources.get(agent_id, [])
    if not agent_sources:
        return {}
    loads = _compute_source_hourly_loads(instance, shared_assignment)
    lines: List[str] = []
    for sid in agent_sources:
        src = instance.sources.get(sid)
        if not src:
            continue
        series = loads.get(sid, [0.0] * instance.timeline_length)
        hourly = ", ".join(f"{v:.1f}" for v in series[: instance.timeline_length])
        lines.append(f"- {sid} ({src.kind}) — Hourly current load (kW): [{hourly}]")
    return {
        "title": "Current shared loads by source:",
        "lines": lines,
    }


def format_neighbour_assignments(*, problem: ProblemDefinition, shared_assignment: Mapping[str, Any]) -> Mapping[str, str]:
    """
    Render neighbour assignments as human-readable source labels, e.g., "src_Solar (Solar)".
    Returns var_name -> rendered value.
    """
    instance: Optional[SmartGridInstance] = getattr(problem, "_smart_grid_instance", None)  # type: ignore[assignment]
    rendered: Dict[str, str] = {}
    for var_name, value in shared_assignment.items():
        try:
            val = str(value)
        except Exception:
            continue
        label = val
        if instance is not None and val in instance.sources:
            src = instance.sources[val]
            label = f"{src.source_id} ({src.kind})"
        rendered[str(var_name)] = label
    return rendered


def format_problem_guidance(*, problem: ProblemDefinition, agent_id: str) -> Mapping[str, Any]:
    """Smart Grid problem-specific guidance rendered as its own block."""
    return {
        "title": "Energy Allocation Rules & Strategy:",
        "lines": [
            "Use the shared load arrays to spread concurrent machines across your available sources and hours with headroom.",
            "Prefer daylight Solar capacity during high-output hours; use Biomass/Geothermal at night to avoid overflow.",
            "Avoid hours where current load is near/exceeds capacity; reassign machines to reduce overloaded hours.",
            "When adjusting, prioritise global overflow reduction over keeping all machines on a single source.",
        ],
    }
