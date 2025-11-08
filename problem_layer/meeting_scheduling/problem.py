from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from itertools import combinations

try:  # Pillow is optional but preferred for timeline charts
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore

from ..base import AgentSpec, FactorSpec, ProblemDefinition, VariableSpec


# ---------- Core domain -----------------------------------------------------


SOFT_MEETING_TYPES = [
    "Gaming session",
    "Swimming session",
    "Gym play session",
    "Friends’ hangout",
    "Study group",
    "Music jamming session",
    "Lunch break in lounge",
    "Board game night",
    "Volleyball free play",
    "Collaborative art workshop",
]

STRICT_MEETING_TYPES = [
    "University class",
    "Doctor’s appointment",
    "Club administration meeting",
    "Faculty advising session",
    "Scholarship interview",
    "Job interview",
    "Dentist appointment",
    "Mandatory training",
    "One-on-one supervisor meeting",
    "Scheduled exam",
]


# How instructions are presented to agents
class InstructionMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"


# A single scheduled item on the shared timeline
@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    meeting_type: str  # "soft" or "strict"
    title: str
    start: int
    end: int
    participants: Sequence[str]


# Controls instance size, randomness, and presentation
@dataclass(frozen=True)
class MeetingSchedulingConfig:
    num_agents: int = 8
    num_meetings: int = 6
    timeline_length: int = 12
    min_participants: int = 2
    max_participants: int = 4
    soft_meeting_ratio: float = 0.6
    instruction_mode: InstructionMode = InstructionMode.TEXT
    include_timelines: bool = False
    rng_seed: int = 123
    output_stem: str = "meeting_scheduling_instance"

    def validate(self) -> None:
        # Basic sanity checks for configuration values
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.num_meetings < 1:
            raise ValueError("num_meetings must be at least 1")
        if not (0 <= self.soft_meeting_ratio <= 1):
            raise ValueError("soft_meeting_ratio must be between 0 and 1")


# Produced instance with everything needed to run/serialize
@dataclass
class MeetingSchedulingInstance:
    config: MeetingSchedulingConfig
    problem: ProblemDefinition
    meetings: Sequence[Meeting]
    explanations: Dict[str, str]
    timeline_length: int
    max_utility: float
    schedule_images: Dict[str, Optional[str]]
    json_path: Optional[Path] = None
    pickle_path: Optional[Path] = None


"""Random helpers and sampling"""


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def _pick_names(n: int) -> List[str]:
    # Prefer the richer neutral names list from Personal Assistant dataset; fall back to a small built-in list.
    base: List[str]
    try:
        pa_names_path = (Path(__file__).resolve().parent.parent / "personal_assistant" / "data" / "names.json")
        data = json.loads(pa_names_path.read_text(encoding="utf-8"))
        names_field = data.get("names", [])
        base = [str(x) for x in names_field if isinstance(x, (str, int, float))]
        if not base:
            raise ValueError("empty names")
    except Exception:
        base = [
            "Alex",
            "Bailey",
            "Casey",
            "Dakota",
            "Emerson",
            "Harper",
            "Jordan",
            "Logan",
            "Morgan",
            "Parker",
            "Quinn",
            "Riley",
            "Sawyer",
            "Taylor",
            "Rowan",
            "Skyler",
        ]
    names = []
    for i in range(n):
        base_name = base[i % len(base)]
        suffix = "" if i < len(base) else f"_{i // len(base) + 1}"
        names.append(base_name + suffix)
    return names


def _generate_meetings(config: MeetingSchedulingConfig, agents: Sequence[str], rng) -> List[Meeting]:
    # Sample meetings with time windows, titles, and participant subsets
    meetings: List[Meeting] = []
    for idx in range(config.num_meetings):
        meeting_id = f"m{idx+1:03d}"
        is_soft = rng.random() < config.soft_meeting_ratio
        meeting_type = "soft" if is_soft else "strict"
        title = rng.choice(SOFT_MEETING_TYPES if is_soft else STRICT_MEETING_TYPES)

        if meeting_type == "strict":
            start = rng.randrange(0, config.timeline_length - 1)
            end = start + 1
        else:
            start = rng.randrange(0, max(1, config.timeline_length - 3))
            duration = rng.randint(2, min(4, config.timeline_length - start))
            end = start + duration

        size = rng.randint(config.min_participants, min(config.max_participants, len(agents)))
        participants = rng.sample(list(agents), size)
        meetings.append(Meeting(meeting_id, meeting_type, title, start, end, participants))

    # Ensure at least one soft meeting exists
    soft_indices = [i for i, m in enumerate(meetings) if m.meeting_type == "soft"]
    if not soft_indices and meetings:
        # Convert a random meeting to a soft group activity
        j = rng.randrange(len(meetings))
        m = meetings[j]
        start = m.start
        end = m.end
        # For a soft activity, prefer duration >= 2 if possible
        if end - start < 2 and config.timeline_length >= 2:
            start = rng.randrange(0, max(1, config.timeline_length - 3))
            duration = rng.randint(2, min(4, config.timeline_length - start))
            end = start + duration
        meetings[j] = Meeting(
            meeting_id=m.meeting_id,
            meeting_type="soft",
            title=rng.choice(SOFT_MEETING_TYPES),
            start=start,
            end=end,
            participants=m.participants,
        )
        soft_indices = [j]

    # Guarantee: every agent participates in at least one SOFT meeting
    if soft_indices:
        # Track soft coverage
        covered = {a for i in soft_indices for a in meetings[i].participants}
        agents_list = list(agents)
        for agent in agents_list:
            if agent in covered:
                continue
            # Try to add agent to an existing soft meeting that has room
            target_idx: Optional[int] = None
            for i in soft_indices:
                if len(meetings[i].participants) < config.max_participants:
                    target_idx = i
                    break
            if target_idx is not None:
                m = meetings[target_idx]
                participants = list(m.participants)
                if agent not in participants:
                    participants.append(agent)
                meetings[target_idx] = Meeting(
                    meeting_id=m.meeting_id,
                    meeting_type=m.meeting_type,
                    title=m.title,
                    start=m.start,
                    end=m.end,
                    participants=tuple(participants),
                )
                covered.add(agent)
                continue

            # Otherwise, create a new soft meeting that includes this agent
            new_id = f"m{len(meetings) + 1:03d}"
            start = rng.randrange(0, max(1, config.timeline_length - 3))
            duration = rng.randint(2, min(4, config.timeline_length - start))
            end = start + duration
            # Choose size and participants ensuring inclusion of the agent
            size = rng.randint(max(2, config.min_participants), min(config.max_participants, len(agents)))
            pool = [a for a in agents_list if a != agent]
            others = rng.sample(pool, k=max(0, size - 1))
            participants = [agent] + others
            meetings.append(
                Meeting(
                    meeting_id=new_id,
                    meeting_type="soft",
                    title=rng.choice(SOFT_MEETING_TYPES),
                    start=start,
                    end=end,
                    participants=tuple(participants),
                )
            )
            soft_indices.append(len(meetings) - 1)
            covered.add(agent)

    return meetings


# No longer needed: all agents are guaranteed a soft meeting in _generate_meetings


def _encode_interval(join: int, leave: int) -> str:
    # Canonical textual encoding for an attendance interval
    return f"{join}-{leave}"


def _parse_interval(value: str, start: int, end: int) -> Optional[Tuple[int, int]]:
    # Parse an encoded interval and validate it against a meeting window
    if value == "skip":
        return None
    start_str, end_str = value.split("-")
    join = int(start_str)
    leave = int(end_str)
    if join < start or leave > end or join >= leave:
        return None
    return (join, leave)


def _overlap(int_a: Optional[Tuple[int, int]], int_b: Optional[Tuple[int, int]]) -> int:
    # Length of overlap between two half-open intervals
    if not int_a or not int_b:
        return 0
    return max(0, min(int_a[1], int_b[1]) - max(int_a[0], int_b[0]))


def _build_variables(meetings: Sequence[Meeting], agents: Sequence[str]) -> Tuple[List[VariableSpec], Dict[str, List[str]]]:
    # Create per-agent, per-meeting variables with interval domains
    variables: List[VariableSpec] = []
    engagements: Dict[str, List[str]] = {agent: [] for agent in agents}
    for meeting in meetings:
        for agent in meeting.participants:
            domain: List[str] = ["skip"]
            for join in range(meeting.start, meeting.end):
                for leave in range(join + 1, meeting.end + 1):
                    domain.append(_encode_interval(join, leave))
                if meeting.meeting_type == "strict":
                    break
            var_name = f"{agent}__{meeting.meeting_id}"
            variables.append(
                VariableSpec(
                    name=var_name,
                    domain=domain,
                    owner=agent,
                    description=f"Attendance choice for {meeting.meeting_id}",
                )
            )
            engagements[agent].append(meeting.meeting_id)
    return variables, engagements


def _soft_utility(assign: Mapping[str, Any], *, agent_vars: Mapping[str, str], target_agent: str, start: int, end: int) -> float:
    # Reward overlaps: target agent gains utility per overlapping attendee
    def interval_for(agent: str) -> Optional[Tuple[int, int]]:
        value = assign.get(agent_vars[agent], "skip")
        return _parse_interval(value, start, end)

    target_interval = interval_for(target_agent)
    if not target_interval:
        return 0.0
    overlaps = 0
    for other in agent_vars:
        if other == target_agent:
            continue
        if _overlap(target_interval, interval_for(other)) > 0:
            overlaps += 1
    return float(overlaps)


def _strict_utility(assign: Mapping[str, Any], *, var_name: str, start: int, end: int) -> float:
    # Reward only if the exact full window is attended
    interval = _parse_interval(assign.get(var_name, "skip"), start, end)
    if not interval:
        return 0.0
    return 1.0 if interval == (start, end) else 0.0


def _presence_utility(assign: Mapping[str, Any], *, var_name: str) -> float:
    # Neutral tracker (kept for symmetry; contributes 0)
    assign.get(var_name, "skip")  # touch to avoid unused warnings
    return 0.0


def _overlap_penalty(
    assign: Mapping[str, Any],
    *,
    var_a: str,
    start_a: int,
    end_a: int,
    var_b: str,
    start_b: int,
    end_b: int,
) -> float:
    # Penalize overlapping attendance across two meetings for the same agent
    interval_a = _parse_interval(assign.get(var_a, "skip"), start_a, end_a)
    interval_b = _parse_interval(assign.get(var_b, "skip"), start_b, end_b)
    if not interval_a or not interval_b:
        return 0.0
    overlap = _overlap(interval_a, interval_b)
    return -float(overlap)


def _make_soft_factor(meeting: Meeting, target_agent: str) -> FactorSpec:
    # Factor: soft meeting utility for target agent depends on others' overlaps
    agent_vars = {agent: f"{agent}__{meeting.meeting_id}" for agent in meeting.participants}
    return FactorSpec(
        name=f"utility_{meeting.meeting_id}_{target_agent}",
        scope=list(agent_vars.values()),
        description=f"Utility for {target_agent} at {meeting.title}",
        utility_fn=partial(
            _soft_utility,
            agent_vars=agent_vars,
            target_agent=target_agent,
            start=meeting.start,
            end=meeting.end,
        ),
        factor_type="coordination",
    )


def _make_strict_factor(meeting: Meeting, target_agent: str) -> FactorSpec:
    # Factor: strict meeting utility rewards full-window attendance only
    var_name = f"{target_agent}__{meeting.meeting_id}"
    return FactorSpec(
        name=f"utility_{meeting.meeting_id}_{target_agent}",
        scope=[var_name],
        description=f"Full attendance reward for {target_agent} at {meeting.title}",
        utility_fn=partial(
            _strict_utility,
            var_name=var_name,
            start=meeting.start,
            end=meeting.end,
        ),
        factor_type="personal_preference",
    )


def _make_presence_factor(meeting: Meeting, agent: str) -> FactorSpec:
    # Factor: neutral presence (keeps variables engaged; contributes 0)
    var_name = f"{agent}__{meeting.meeting_id}"
    return FactorSpec(
        name=f"presence_{meeting.meeting_id}_{agent}",
        scope=[var_name],
        description=f"Neutral presence tracker for {agent} at {meeting.title}",
        utility_fn=partial(_presence_utility, var_name=var_name),
        factor_type="personal_preference",
    )


def _make_overlap_penalty_factor(meeting_a: Meeting, meeting_b: Meeting, agent: str) -> FactorSpec:
    # Factor: within-agent penalty for overlapping attendance across two meetings
    var_a = f"{agent}__{meeting_a.meeting_id}"
    var_b = f"{agent}__{meeting_b.meeting_id}"
    return FactorSpec(
        name=f"overlap_penalty_{meeting_a.meeting_id}_{meeting_b.meeting_id}_{agent}",
        scope=[var_a, var_b],
        description=f"Penalty for {agent} attending overlapping portions of {meeting_a.title} and {meeting_b.title}",
        utility_fn=partial(
            _overlap_penalty,
            var_a=var_a,
            start_a=meeting_a.start,
            end_a=meeting_a.end,
            var_b=var_b,
            start_b=meeting_b.start,
            end_b=meeting_b.end,
        ),
        factor_type="coordination",
    )


def _build_problem(meetings: Sequence[Meeting], agents: Sequence[str]) -> Tuple[ProblemDefinition, Dict[str, List[str]]]:
    # Assemble variables and factors into a ProblemDefinition
    variables, engagements = _build_variables(meetings, agents)
    factors: List[FactorSpec] = []

    for meeting in meetings:
        for agent in meeting.participants:
            factors.append(_make_presence_factor(meeting, agent))
            if meeting.meeting_type == "soft":
                factors.append(_make_soft_factor(meeting, agent))
            else:
                factors.append(_make_strict_factor(meeting, agent))

    for agent in agents:
        agent_meetings = [meeting for meeting in meetings if agent in meeting.participants]
        for meeting_a, meeting_b in combinations(agent_meetings, 2):
            factors.append(_make_overlap_penalty_factor(meeting_a, meeting_b, agent))

    agent_specs = [
        AgentSpec(
            agent_id=agent,
            name=agent,
            instruction="Coordinate your attendance across assigned meetings, balancing overlaps and full coverage where needed.",
        )
        for agent in agents
    ]

    problem = ProblemDefinition(
        name="meeting_scheduling",
        description="Decide attendance intervals for meetings on a shared timeline.",
        agents=agent_specs,
        variables=variables,
        factors=factors,
    )
    return problem, engagements


def _compute_max_utility(meetings: Sequence[Meeting]) -> float:
    # Upper bound used for ratio reporting (soft: O(n^2), strict: O(n))
    total = 0.0
    for meeting in meetings:
        participant_count = len(meeting.participants)
        if meeting.meeting_type == "soft":
            total += float((participant_count - 1) * participant_count)
        else:
            total += float(participant_count)
    return total


def format_factor_information(*, problem: ProblemDefinition, agent_id: str, shared_assignment: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Meeting Scheduling does not provide a factor information block by default.
    Returning an empty mapping allows the prompt builder to warn or skip.
    """
    return {}


def format_problem_guidance(*, problem: ProblemDefinition, agent_id: str) -> Mapping[str, Any]:
    """Meeting Scheduling problem-specific guidance rendered as its own block."""
    return {
        "title": "Scheduling Rules & Strategy:",
        "lines": [
            "For soft group activities: choose an interval that overlaps with the majority of participants; longer overlaps score better.",
            "For strict mandatory sessions: choose exactly the full window of the meeting.",
            "If two meetings overlap: prioritise one rather than splitting across both.",
            "If strict vs soft overlap: fully attend the strict window, then attend any remaining non-overlapping portion of the soft window.",
            "If soft vs soft overlap: prefer the meeting with more participants or with a consistent interval chosen by others; reduce/skip the other.",
            "When unsure and no neighbour choices are visible: for soft choose the longest feasible interval; for strict choose the exact window.",
            "If multiple teammates chose the same interval, matching that exact interval is preferred.",
            "Avoid 'skip' unless conflicts or lack of benefit make attendance impractical.",
        ],
    }


def format_neighbour_assignments(*, problem: ProblemDefinition, shared_assignment: Mapping[str, Any]) -> Mapping[str, str]:
    """
    Render neighbour assignments as human-readable meeting intervals including titles, e.g.,
    "5-7 (Gaming session)". Returns var_name -> rendered value.
    """
    instance = getattr(problem, "_meeting_scheduling_instance", DEFAULT_INSTANCE)
    # Map meeting_id to title for quick lookup
    title_by_id = {m.meeting_id: m.title for m in instance.meetings}
    rendered: Dict[str, str] = {}
    for var_name, value in shared_assignment.items():
        # var name pattern: Agent__meeting_id
        parts = str(var_name).split("__", 1)
        meeting_id = parts[1] if len(parts) == 2 else None
        title = title_by_id.get(meeting_id, None) if meeting_id else None
        try:
            if title:
                rendered[var_name] = f"{value} ({title})"
            else:
                rendered[var_name] = str(value)
        except Exception:
            rendered[var_name] = str(value)
    return rendered


def _wrap_text(text: str, *, max_chars: int = 42) -> List[str]:
    # Lightweight text wrapper for chart labels
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    current: List[str] = []
    length = 0
    for word in words:
        extra = len(word) if not current else len(word) + 1
        if length + extra > max_chars and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines


"""Visualization and instruction composition"""


def _generate_schedule_chart(
    agent_id: str,
    meetings: Sequence[Meeting],
    timeline_length: int,
    output_dir: Path,
) -> Optional[Path]:
    # Draw a simple per-agent timeline chart (if PIL is available)
    if not meetings:
        return None
    if Image is None or ImageDraw is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    row_height = 80
    margin = 60
    legend_width = 160
    scale = 40
    chart_width = max(timeline_length * scale, 360)
    width = chart_width + 2 * margin + legend_width
    height = max(320, margin * 2 + row_height * len(meetings))
    chart_x0 = margin
    chart_x1 = margin + chart_width
    chart_y0 = margin
    chart_y1 = height - margin

    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    font = None
    small_font = None
    if ImageFont is not None:
        try:
            font = ImageFont.load_default()
            small_font = font
        except Exception:
            font = None
            small_font = None

    # Axes
    draw.line((chart_x0, chart_y1, chart_x1, chart_y1), fill="black", width=2)
    draw.line((chart_x0, chart_y1, chart_x0, chart_y0), fill="black", width=2)

    # Timeline ticks
    for hour in range(timeline_length + 1):
        x = chart_x0 + hour * (chart_width / max(1, timeline_length))
        draw.line((x, chart_y1, x, chart_y1 + 6), fill="black")
        if font is not None:
            draw.text((x - 4, chart_y1 + 10), str(hour), fill="black", font=font)

    mandatory_color = "#1f77b4"
    soft_color = "#ff7f0e"

    for idx, meeting in enumerate(sorted(meetings, key=lambda m: (m.start, m.end))):
        row_top = chart_y0 + idx * row_height
        row_bottom = row_top + 28
        start_x = chart_x0 + meeting.start * (chart_width / max(1, timeline_length))
        end_x = chart_x0 + meeting.end * (chart_width / max(1, timeline_length))
        color = soft_color if meeting.meeting_type == "soft" else mandatory_color
        draw.rectangle((start_x, row_top, end_x, row_bottom), fill=color, outline="black")

        title = meeting.title
        title_lines = _wrap_text(title, max_chars=28)
        text_y = row_top - 18
        for line in title_lines:
            if font is not None:
                draw.text((start_x, text_y), line, fill="black", font=font)
            text_y += 14

        participant_text = "Participants: " + ", ".join(meeting.participants)
        for line_idx, line in enumerate(_wrap_text(participant_text, max_chars=50)):
            if small_font is not None:
                draw.text((start_x, row_bottom + 6 + line_idx * 12), line, fill="black", font=small_font)

        window_text = f"[{meeting.start}, {meeting.end})"
        if font is not None:
            draw.text((end_x + 4, row_top + 6), window_text, fill="black", font=font)

    # Legend
    legend_x = chart_x1 + 20
    legend_y = chart_y0
    legend_entries = [
        (mandatory_color, "Mandatory session"),
        (soft_color, "Group activity"),
    ]
    for idx, (color, label) in enumerate(legend_entries):
        y = legend_y + idx * 24
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=color, outline="black")
        if font is not None:
            draw.text((legend_x + 20, y + 1), label, fill="black", font=font)

    safe_name = agent_id.replace(" ", "_")
    chart_path = output_dir / f"{safe_name}_schedule.png"
    try:
        image.save(chart_path)
    except Exception:
        return None
    return chart_path


def _compose_instruction(
    agent: str,
    meetings: Sequence[Meeting],
    timeline_length: int,
    schedule_path: Optional[str] = None,
) -> str:
    # Render final instruction text per agent (TEXT lists items; IMAGE references chart)
    if not meetings:
        return (
            "You are not assigned to any meetings on this schedule. Observe the timeline and be ready if new events arise."
        )
    lines = [
        "You are the personal scheduling assistant for this attendee.",
        "Review each meeting and choose whether to attend or skip.",
        "Timeline ticks are numbered slots; attendance intervals must lie within each meeting window.",
    ]
    if schedule_path:
        lines.extend(
            [
                "",
                "Meeting schedule image attached; use it to review the windows and participants before deciding.",
            ]
        )
    else:
        lines.append("")
        lines.append("Meetings you are expected to consider:")
        for meeting in sorted(meetings, key=lambda m: m.start):
            kind = "group activity" if meeting.meeting_type == "soft" else "mandatory session"
            window = f"[{meeting.start}, {meeting.end})"
            lines.append(
                f"- {meeting.title} ({kind}), window {window}, participants: {', '.join(meeting.participants)}"
            )
        lines.append("")
    lines.append(
        "Soft activities reward you for overlapping with peers; strict sessions require covering the full window."
    )

    # Detailed rules are provided via problem-specific guidance block in prompts.
    return "\n".join(lines)


"""Instance summary and persistence"""


def _instance_summary(instance: MeetingSchedulingInstance, engagements: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    # Build a JSON-serializable snapshot of the instance (agents, variables, factors, schemas)
    problem = instance.problem
    agents_block = []
    for agent in problem.agents.values():
        agents_block.append(
            {
                "id": agent.agent_id,
                "name": agent.name,
                "instruction": instance.explanations[agent.agent_id],
                "variables": engagements.get(agent.agent_id, []),
                "assets": {"image": instance.schedule_images.get(agent.agent_id)},
            }
        )

    variables_block = []
    for var_name, var_spec in problem.variables.items():
        variables_block.append(
            {
                "name": var_name,
                "owner": var_spec.owner,
                "domain": list(var_spec.domain),
                "description": var_spec.description,
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

    meetings_block = [
        {
            "id": meeting.meeting_id,
            "title": meeting.title,
            "type": meeting.meeting_type,
            "window": [meeting.start, meeting.end],
            "participants": list(meeting.participants),
        }
        for meeting in instance.meetings
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
        },
        "agent_variable_map": engagements,
        "agents": agents_block,
        "variables": variables_block,
        "factors": factors_block,
        "meetings": meetings_block,
        "schemas": {
            "joint_assignment": problem.joint_assignment_schema(),
            "agents": {agent_id: problem.agent_schema(agent_id) for agent_id in problem.agents.keys()},
        },
    }


"""Instance construction and module-level defaults"""


def _build_instance(config: MeetingSchedulingConfig, *, save_dir: Optional[Path]) -> MeetingSchedulingInstance:
    # Generate meetings, build problem, compose instructions, and optionally save
    config.validate()
    rng = _rng(config.rng_seed)
    agents = _pick_names(config.num_agents)
    meetings = _generate_meetings(config, agents, rng)
    problem, engagements = _build_problem(meetings, agents)

    schedule_rel_paths: Dict[str, Optional[str]] = {agent: None for agent in agents}
    if config.instruction_mode is InstructionMode.IMAGE and config.include_timelines and save_dir is not None:
        schedule_dir = save_dir / "schedules"
        for agent in agents:
            agent_meetings = [meeting for meeting in meetings if agent in meeting.participants]
            chart_path = _generate_schedule_chart(agent, agent_meetings, config.timeline_length, schedule_dir)
            if chart_path is not None:
                schedule_rel_paths[agent] = str(Path("schedules") / chart_path.name)

    explanations: Dict[str, str] = {}
    for agent in agents:
        agent_meetings = [meeting for meeting in meetings if agent in meeting.participants]
        explanations[agent] = _compose_instruction(
            agent,
            agent_meetings,
            config.timeline_length,
            schedule_rel_paths.get(agent),
        )

    problem = ProblemDefinition(
        name=problem.name,
        description=problem.description,
        agents=[
            AgentSpec(
                agent_id=spec.agent_id,
                name=spec.name,
                instruction=explanations[spec.agent_id],
            )
            for spec in problem.agents.values()
        ],
        variables=list(problem.variables.values()),
        factors=list(problem.list_factors()),
    )

    max_utility = _compute_max_utility(meetings)

    instance = MeetingSchedulingInstance(
        config=config,
        problem=problem,
        meetings=meetings,
        explanations=explanations,
        timeline_length=config.timeline_length,
        max_utility=max_utility,
        schedule_images=schedule_rel_paths,
    )

    # Attach back-reference for neighbour rendering and guidance
    try:
        setattr(problem, "_meeting_scheduling_instance", instance)
    except Exception:
        pass

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = _instance_summary(instance, {k: list(v) for k, v in engagements.items()})
        json_path = save_dir / f"{config.output_stem}.json"
        pickle_path = save_dir / f"{config.output_stem}.pkl"
        json_path.write_text(json.dumps(summary, indent=2))
        with open(pickle_path, "wb") as pf:
            pickle.dump(instance, pf)
        instance.json_path = json_path
        instance.pickle_path = pickle_path

    return instance


def generate_instance(config: MeetingSchedulingConfig, output_dir: Path) -> MeetingSchedulingInstance:
    # Public entry point for generating and saving an instance
    return _build_instance(config, save_dir=output_dir)


DEFAULT_CONFIG = MeetingSchedulingConfig()
DEFAULT_INSTANCE = _build_instance(DEFAULT_CONFIG, save_dir=None)

PROBLEM = DEFAULT_INSTANCE.problem
AGENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    agent_id: PROBLEM.agent_schema(agent_id) for agent_id in PROBLEM.agents.keys()
}
JOINT_ASSIGNMENT_SCHEMA: Dict[str, Any] = PROBLEM.joint_assignment_schema()
MAX_UTILITY = DEFAULT_INSTANCE.max_utility


def eval(joint_assignment: Mapping[str, Any]) -> Dict[str, Any]:
    # Evaluate a joint assignment against the default problem
    result = PROBLEM.eval(joint_assignment)
    out = dict(result)
    out["min_utility"] = 0.0
    out["max_utility"] = MAX_UTILITY
    if out.get("valid") and out.get("total_utility") is not None:
        out["total_utility"] = float(out["total_utility"])
    return out
