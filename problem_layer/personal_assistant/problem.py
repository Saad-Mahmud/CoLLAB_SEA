from __future__ import annotations

import json
import pickle
import random
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..base import AgentSpec, FactorSpec, ProblemDefinition, VariableSpec


# ---------- Dataset loading --------------------------------------------------


@dataclass(frozen=True)
class PersonalAssistantDataset:
    # Small, static dataset for sampling instances
    names: Sequence[str]
    colors: Sequence[str]
    articles: Sequence[str]
    image_index: Dict[str, Dict[str, str]]  # article -> color -> relative path (e.g. "figs/blue_shirt.png")


def _load_dataset(data_dir: Path) -> PersonalAssistantDataset:
    # Load JSON files and build a quick lookup from (article, color) -> image path
    def read_json(filename: str) -> Any:
        with open(data_dir / filename, "r", encoding="utf-8") as f:
            return json.load(f)

    names = read_json("names.json")['names']
    colors = read_json("colors.json")['colors']
    articles = read_json("articles.json")['articles']
    images = read_json("images.json")['images']

    image_index: Dict[str, Dict[str, str]] = {}
    for record in images:
        article = record['article']
        color = record['color']
        filename = record['filename']
        image_index.setdefault(article, {})[color] = f"figs/{filename}"

    return PersonalAssistantDataset(
        names=tuple(names),
        colors=tuple(colors),
        articles=tuple(articles),
        image_index=image_index,
    )


# ---------- Core types ------------------------------------------------------


@dataclass(frozen=True)
class Outfit:
    # A single wardrobe item
    article: str
    color: str
    image: Optional[str] = None  # Relative path within problem data directory


DATA_DIR = Path(__file__).parent / "data"
DATASET = _load_dataset(DATA_DIR)


# ---------- Factor utilities -------------------------------------------------


def _choice_to_outfit(wardrobe: Mapping[str, Sequence[Outfit]], agent_id: str, choice: Any) -> Outfit:
    # Convert a 1-based menu choice into an Outfit for the given agent
    options = wardrobe.get(agent_id)
    if not options:
        raise KeyError(f"Unknown agent {agent_id!r}.")
    idx = int(choice) - 1
    if idx < 0 or idx >= len(options):
        raise ValueError(f"Choice {choice!r} out of bounds for agent {agent_id!r}.")
    return options[idx]


def _pref_color_utility(
    assign: Mapping[str, Any],
    *,
    wardrobe: Mapping[str, Sequence[Outfit]],
    agent_id: str,
    var_name: str,
    color: str,
    reward: float,
) -> float:
    # Reward when chosen outfit matches the preferred color
    outfit = _choice_to_outfit(wardrobe, agent_id, assign[var_name])
    return reward if outfit.color == color else 0.0


def _avoid_color_utility(
    assign: Mapping[str, Any],
    *,
    wardrobe: Mapping[str, Sequence[Outfit]],
    agent_id: str,
    var_name: str,
    color: str,
    reward: float,
) -> float:
    # Reward when chosen outfit avoids a specific color
    outfit = _choice_to_outfit(wardrobe, agent_id, assign[var_name])
    return reward if outfit.color != color else 0.0


def _paired_preference_utility(
    assign: Mapping[str, Any],
    *,
    wardrobe: Mapping[str, Sequence[Outfit]],
    agent_a: str,
    agent_b: str,
    var_a: str,
    var_b: str,
    attribute: str,
    pref_a: str,
    pref_b: str,
) -> float:
    # Pairwise score: match/contrast on an attribute (color/article) per each agent's preference
    outfit_a = _choice_to_outfit(wardrobe, agent_a, assign[var_a])
    outfit_b = _choice_to_outfit(wardrobe, agent_b, assign[var_b])
    value_a = getattr(outfit_a, attribute)
    value_b = getattr(outfit_b, attribute)
    match = value_a == value_b
    score = 0.0
    if pref_a == "match" and match:
        score += 1.0
    if pref_a == "contrast" and not match:
        score += 1.0
    if pref_b == "match" and match:
        score += 1.0
    if pref_b == "contrast" and not match:
        score += 1.0
    return score


def _pref_colors_utility(
    assign: Mapping[str, Any],
    *,
    wardrobe: Mapping[str, Sequence[Outfit]],
    agent_id: str,
    var_name: str,
    colors: Sequence[str],
    reward: float,
) -> float:
    # Reward when chosen outfit matches any preferred color in the set
    outfit = _choice_to_outfit(wardrobe, agent_id, assign[var_name])
    return reward if outfit.color in set(colors) else 0.0


def _avoid_colors_utility(
    assign: Mapping[str, Any],
    *,
    wardrobe: Mapping[str, Sequence[Outfit]],
    agent_id: str,
    var_name: str,
    colors: Sequence[str],
    reward: float,
) -> float:
    # Reward when chosen outfit avoids all colors in the given set
    outfit = _choice_to_outfit(wardrobe, agent_id, assign[var_name])
    return reward if outfit.color not in set(colors) else 0.0


def format_factor_information(*, problem: ProblemDefinition, agent_id: str, shared_assignment: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Personal Assistant does not expose a factor information block; return empty.
    """
    return {}


def format_problem_guidance(*, problem: ProblemDefinition, agent_id: str) -> Mapping[str, Any]:
    """Personal Assistant problem-specific guidance rendered as its own block."""
    return {
        "title": "Wardrobe Rules & Strategy:",
        "lines": [
            "Prefer outfits that match your preferred colours; avoid listed no-go colours.",
            "Respect pairwise constraints (match/contrast on colour or article) with teammates when it increases overall utility.",
            "When ties occur, choose items that coordinate better with prevalent teammate choices.",
            "If no strong constraints apply, pick the highest-preference option from your wardrobe list.",
        ],
    }


# ---------- Config + instance types -----------------------------------------


class InstructionMode(str, Enum):
    TEXT = "text"
    IMAGE = "image"


@dataclass(frozen=True)
class PersonalAssistantConfig:
    # Controls instance size, randomness, and presentation
    num_agents: int = 3
    density: float = 0.3  # edge density in [0,1] for coordination graph
    min_outfits_per_agent: int = 4
    max_outfits_per_agent: int = 6
    rng_seed: int = 7
    instruction_mode: InstructionMode = InstructionMode.TEXT
    include_collages: bool = True
    collage_cols: int = 2
    collage_thumb_size: Tuple[int, int] = (256, 256)
    collage_padding: int = 12
    output_stem: str = "personal_assistant_instance"

    def validate(self) -> None:
        # Sanity-checks for configuration values
        if self.num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        if self.min_outfits_per_agent < 1 or self.max_outfits_per_agent < self.min_outfits_per_agent:
            raise ValueError("Invalid outfit count range")
        if not (0.0 <= float(self.density) <= 1.0):
            raise ValueError("density must be between 0 and 1")


@dataclass
class PersonalAssistantInstance:
    # Produced instance with everything needed to run/serialize
    config: PersonalAssistantConfig
    problem: ProblemDefinition
    wardrobe: Dict[str, Sequence[Outfit]]
    instructions: Dict[str, str]
    collages: Dict[str, Optional[str]]
    # Bounds for ratio reporting
    max_utility: float = 0.0
    json_path: Optional[Path] = None
    pickle_path: Optional[Path] = None


# ---------- Problem construction --------------------------------------------


def _pick_name(index: int, names: Sequence[str]) -> str:
    # Deterministically cycle through names and append suffix when wrapping
    base = names[index % len(names)]
    suffix_idx = index // len(names)
    if suffix_idx == 0:
        return base
    return f"{base}_{suffix_idx + 1}"


def _random_wardrobe(rng: random.Random, dataset: PersonalAssistantDataset, config: PersonalAssistantConfig) -> Dict[str, Sequence[Outfit]]:
    # Sample per-agent wardrobes with unique (article, color) combos and optional image paths
    wardrobe: Dict[str, Sequence[Outfit]] = {}
    for agent_idx in range(config.num_agents):
        agent_id = _pick_name(agent_idx, dataset.names)
        k = rng.randint(config.min_outfits_per_agent, config.max_outfits_per_agent)
        outfits: List[Outfit] = []
        seen: set[Tuple[str, str]] = set()
        attempts = 0
        while len(outfits) < k and attempts < k * 50:
            attempts += 1
            article = rng.choice(dataset.articles)
            color = rng.choice(dataset.colors)
            if (article, color) in seen:
                continue
            seen.add((article, color))
            rel_path = dataset.image_index.get(article, {}).get(color)
            image_path = str(Path("data") / rel_path) if rel_path else None
            outfits.append(Outfit(article=article, color=color, image=image_path))
        if len(outfits) < k:
            raise RuntimeError(f"Unable to sample sufficient outfit variety for agent {agent_id}.")
        if len(outfits) == 1:
            warnings.warn(
                f"Agent {agent_id} sampled a wardrobe of size 1; consider increasing min_outfits_per_agent.")
        wardrobe[agent_id] = tuple(outfits)
    return wardrobe


def _build_agents_and_factors(
    rng: random.Random,
    dataset: PersonalAssistantDataset,
    config: PersonalAssistantConfig,
    wardrobe: Mapping[str, Sequence[Outfit]],
) -> Tuple[List[AgentSpec], List[VariableSpec], List[FactorSpec], Dict[str, Dict[str, Any]]]:
    # Create Agent/Variable specs, unary preference factors, and sparse pairwise coordination factors
    metadata: Dict[str, Dict[str, Any]] = {}
    agents: List[AgentSpec] = []
    variables: List[VariableSpec] = []
    factors: List[FactorSpec] = []

    agent_ids = list(wardrobe.keys())
    agent_variable_map: Dict[str, str] = {}

    for agent_id in agent_ids:
        variable_name = f"{agent_id}'s Outfit"
        metadata[agent_id] = {
            "prefer": [],
            "avoid": [],
            "constraints": [],
        }

        # General instruction; specifics are added in composed instruction text
        instruction = (
            "Focus on outfits that highlight your preferred colors and steer clear of your avoid colors, "
            "while staying coordinated with the rest of the team."
        )

        agents.append(
            AgentSpec(
                agent_id=agent_id,
                name=f"{agent_id.title()}",
                instruction=instruction,
            )
        )

        variables.append(
            VariableSpec(
                name=variable_name,
                domain=list(range(1, len(wardrobe[agent_id]) + 1)),
                owner=agent_id,
                description="Select the numbered outfit from your wardrobe list.",
            )
        )
        agent_variable_map[agent_id] = variable_name

        # Determine preference and avoidance color sets
        wardrobe_colors = sorted({o.color for o in wardrobe[agent_id]})

        # Pick at least two preferred colors from the agent's wardrobe when possible
        if len(wardrobe_colors) >= 2:
            prefer_colors = rng.sample(wardrobe_colors, k=2)
        else:
            prefer_colors = wardrobe_colors  # fall back to whatever is available (possibly 1)

        # Pick at least two avoid colors (preferably outside preferred colors)
        avoid_pool = [c for c in dataset.colors if c not in set(prefer_colors)]
        if len(avoid_pool) >= 2:
            avoid_colors = rng.sample(avoid_pool, k=2)
        else:
            # Not enough outside preferred; allow overlap as a fallback
            base = list(dataset.colors)
            # Ensure we still try to pick distinct colors
            if len(base) >= 2:
                # Remove duplicates if necessary
                first = rng.choice(base)
                second_choices = [c for c in base if c != first]
                second = rng.choice(second_choices) if second_choices else first
                avoid_colors = [first, second]
            else:
                avoid_colors = base  # degenerate dataset case

        # Save metadata for instruction composition
        metadata[agent_id]["prefer"] = list(prefer_colors)
        metadata[agent_id]["avoid"] = list(avoid_colors)

        # Preference factor: reward if chosen color is in preferred set
        pref_colors_name = "_".join(prefer_colors) if prefer_colors else "none"
        factors.append(
            FactorSpec(
                name=f"{agent_id}_pref_set_{pref_colors_name}",
                scope=[variable_name],
                description=f"Reward for matching {agent_id}'s preferred colors ({', '.join(prefer_colors)}).",
                utility_fn=partial(
                    _pref_colors_utility,
                    wardrobe=wardrobe,
                    agent_id=agent_id,
                    var_name=variable_name,
                    colors=tuple(prefer_colors),
                    reward=1.0,
                ),
                factor_type="personal_preference",
            )
        )

        # Avoidance factor: reward if chosen color is not in avoided set
        avoid_colors_name = "_".join(avoid_colors) if avoid_colors else "none"
        factors.append(
            FactorSpec(
                name=f"{agent_id}_avoid_set_{avoid_colors_name}",
                scope=[variable_name],
                description=f"Reward for avoiding {agent_id}'s dislike colors ({', '.join(avoid_colors)}).",
                utility_fn=partial(
                    _avoid_colors_utility,
                    wardrobe=wardrobe,
                    agent_id=agent_id,
                    var_name=variable_name,
                    colors=tuple(avoid_colors),
                    reward=1.0,
                ),
                factor_type="personal_preference",
            )
        )

    # Build connected pairwise interactions based on edge density
    if config.num_agents > 1 and config.density > 0:
        for a, b in _coordination_edges_by_density(agent_ids, float(config.density), rng):
            attribute = rng.choice(["color", "article"])
            pref_a = rng.choice(["match", "contrast"])
            pref_b = rng.choice(["match", "contrast"])
            desc_attr = "color palette" if attribute == "color" else "article style"
            factor = FactorSpec(
                name=f"{a}_{b}_{attribute}_coord",
                scope=[agent_variable_map[a], agent_variable_map[b]],
                description=(
                    f"{a} prefers to {pref_a} on {desc_attr}, while {b} prefers to {pref_b} on {desc_attr}."
                ),
                utility_fn=partial(
                    _paired_preference_utility,
                    wardrobe=wardrobe,
                    agent_a=a,
                    agent_b=b,
                    var_a=agent_variable_map[a],
                    var_b=agent_variable_map[b],
                    attribute=attribute,
                    pref_a=pref_a,
                    pref_b=pref_b,
                ),
                factor_type="coordination",
            )
            factors.append(factor)
            metadata[a]["constraints"].append({"partner": b, "type": pref_a, "attribute": attribute})
            metadata[b]["constraints"].append({"partner": a, "type": pref_b, "attribute": attribute})

    return agents, variables, factors, metadata


def _coordination_edges_by_density(
    agent_ids: Sequence[str], density: float, rng: random.Random
) -> List[Tuple[str, str]]:
    """Build a random connected undirected graph with approximately the given edge density.

    Strategy (networkx-free):
      - Create a random spanning path to guarantee connectivity (n-1 edges).
      - Add random extra edges among the remaining pairs until reaching m = round(density * n(n-1)/2).
    """
    n = len(agent_ids)
    if n <= 1 or density <= 0:
        return []
    max_edges = n * (n - 1) // 2
    target = int(round(float(density) * max_edges))
    target = max(target, n - 1)
    target = min(target, max_edges)

    # Random order for spanning path
    order = list(agent_ids)
    rng.shuffle(order)
    # Use sorted tuples to ensure undirected uniqueness
    edges: set[Tuple[str, str]] = set()
    for i in range(n - 1):
        a, b = order[i], order[i + 1]
        if a > b:
            a, b = b, a
        edges.add((a, b))

    # Candidate non-edges
    candidates: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order[i], order[j]
            if a > b:
                a, b = b, a
            if (a, b) not in edges:
                candidates.append((a, b))
    rng.shuffle(candidates)

    for e in candidates:
        if len(edges) >= target:
            break
        edges.add(e)
    return list(edges)


def _generate_collage(
    agent_id: str,
    outfits: Sequence[Outfit],
    output_dir: Path,
    *,
    cols: int,
    thumb_size: Tuple[int, int],
    padding: int,
) -> Optional[str]:
    # Build a simple grid collage of outfit images (if PIL is available)
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    images: List[Path] = []
    for outfit in outfits:
        if outfit.image:
            rel = Path(outfit.image)
            if rel.is_absolute():
                img_path = rel
            elif rel.parts and rel.parts[0] == "data":
                img_path = DATA_DIR / Path(*rel.parts[1:])
            else:
                img_path = DATA_DIR / rel
            if img_path.exists():
                images.append(img_path)

    if not images:
        return None

    rows = (len(images) + cols - 1) // cols
    width, height = thumb_size
    canvas_w = cols * width + (cols + 1) * padding
    canvas_h = rows * height + (rows + 1) * padding
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 0))

    for idx, image_path in enumerate(images):
        try:
            thumb = Image.open(image_path).convert("RGBA").resize((width, height))
        except Exception:
            continue
        row, col = divmod(idx, cols)
        x = padding + col * (width + padding)
        y = padding + row * (height + padding)
        canvas.paste(thumb, (x, y), mask=thumb if thumb.mode == "RGBA" else None)

    output_dir.mkdir(parents=True, exist_ok=True)
    collage_path = output_dir / f"{agent_id}_wardrobe.png"
    canvas.save(collage_path)
    return str(collage_path)


def _compose_instruction(
    agent: AgentSpec,
    wardrobe: Sequence[Outfit],
    metadata: Mapping[str, Any],
    mode: InstructionMode,
    collage_path: Optional[str],
) -> str:
    # Render the final instruction text per agent (TEXT lists items, IMAGE references collage)
    prefer = metadata.get("prefer")
    avoid = metadata.get("avoid")
    constraints = metadata.get("constraints", [])

    # Normalize prefer/avoid for presentation
    def _fmt_colors(val: Any) -> str:
        if isinstance(val, (list, tuple)):
            return ", ".join(str(x) for x in val)
        return str(val)

    lines: List[str] = [
        "Task: You are the personal assistant AI for Sam, coordinating with other personal agents to choose outfits for an upcoming event.",
        agent.instruction,
        "",
        f"Personal colour guideline: prefer {_fmt_colors(prefer)}, avoid {_fmt_colors(avoid)}.",
    ]

    if mode is InstructionMode.TEXT:
        lines.append("")
        lines.append("Wardrobe options (numbered):")
        for idx, outfit in enumerate(wardrobe, start=1):
            lines.append(f"{idx}. {outfit.color.title()} {outfit.article}")
    else:
        lines.append("")
        if collage_path:
            lines.append("Wardrobe reference image attached in prompt.")
            lines.append("Inspect the collage to see each numbered outfit before choosing.")
        else:
            lines.append("")
            lines.append("Wardrobe options (numbered):")
            for idx, outfit in enumerate(wardrobe, start=1):
                lines.append(f"{idx}. {outfit.color.title()} {outfit.article}")

    if constraints:
        # Summarize coordination rules as bullet points
        lines.append("")
        lines.append("Coordination constraints:")
        for constraint in constraints:
            partner = constraint.get("partner")
            ctype = constraint.get("type")
            attribute = constraint.get("attribute", "color")
            attr_phrase = "colours" if attribute == "color" else "articles"
            if ctype == "match":
                detail = f"Match {attr_phrase} with {partner} to stay cohesive."
            else:
                detail = f"Contrast {attr_phrase} with {partner} to balance the palette."
            lines.append(f"- Constraint ({ctype} on {attribute}): {detail}")

        # Clarify constraint semantics (PA-specific)
        lines.append("")
        lines.append("Constraint semantics:")
        lines.append("- Match on color: your outfit color must equal your partner's outfit color.")
        lines.append("- Contrast on color: your outfit color must differ from your partner's outfit color.")
        lines.append("- Match on article: your outfit article (e.g., shirt, dress) must equal your partner's article; color can differ.")
        lines.append("- Contrast on article: your outfit article must differ from your partner's article.")

    return "\n".join(lines)


def _build_problem_instance(
    config: PersonalAssistantConfig,
    dataset: PersonalAssistantDataset,
    *,
    save_dir: Optional[Path] = None,
) -> PersonalAssistantInstance:
    # Orchestrate sampling wardrobes, building specs, instructions, and optional assets/serialization
    config.validate()
    rng = random.Random(config.rng_seed)

    wardrobe = _random_wardrobe(rng, dataset, config)
    agents, variables, factors, agent_metadata = _build_agents_and_factors(rng, dataset, config, wardrobe)

    collages: Dict[str, Optional[str]] = {agent_id: None for agent_id in wardrobe.keys()}
    if config.instruction_mode is InstructionMode.IMAGE and config.include_collages and save_dir is not None:
        collage_dir = save_dir / "collages"
        for agent_id in wardrobe.keys():
            collage_path = _generate_collage(
                agent_id,
                wardrobe[agent_id],
                collage_dir,
                cols=config.collage_cols,
                thumb_size=config.collage_thumb_size,
                padding=config.collage_padding,
            )
            if collage_path:
                collages[agent_id] = str(Path("collages") / Path(collage_path).name)
            else:
                collages[agent_id] = None

    instructions: Dict[str, str] = {}
    for agent in agents:
        instructions[agent.agent_id] = _compose_instruction(
            agent=agent,
            wardrobe=wardrobe[agent.agent_id],
            metadata=agent_metadata[agent.agent_id],
            mode=config.instruction_mode,
            collage_path=collages.get(agent.agent_id),
        )

    problem = ProblemDefinition(
        name="personal_assistant",
        description="Coordinate wardrobe selections for Sam's campaign using numbered outfit choices.",
        agents=[
            AgentSpec(
                agent_id=agent.agent_id,
                name=agent.name,
                instruction=instructions[agent.agent_id],
            )
            for agent in agents
        ],
        variables=variables,
        factors=factors,
    )

    # Compute an instance-specific upper bound consistent with baselines:
    # unary factors contribute 1; pairwise factors contribute 2.
    max_utility = float(sum(1.0 if len(f.scope) == 1 else 2.0 for f in problem.list_factors()))

    instance = PersonalAssistantInstance(
        config=config,
        problem=problem,
        wardrobe=dict(wardrobe),
        instructions=instructions,
        collages=collages,
        max_utility=max_utility,
    )

    # Attach back-reference so the problem can render neighbour assignments and guidance at runtime
    try:
        setattr(problem, "_personal_assistant_instance", instance)
    except Exception:
        pass

    if save_dir is not None:
        # Persist a compact JSON summary and a pickle of the full instance
        summary = _instance_summary(instance)
        save_dir.mkdir(parents=True, exist_ok=True)
        json_path = save_dir / f"{config.output_stem}.json"
        pickle_path = save_dir / f"{config.output_stem}.pkl"
        json_path.write_text(json.dumps(summary, indent=2))
        with open(pickle_path, "wb") as pf:
            pickle.dump(instance, pf)
        instance.json_path = json_path
        instance.pickle_path = pickle_path

    return instance


def generate_instance(
    config: PersonalAssistantConfig,
    output_dir: Path,
) -> PersonalAssistantInstance:
    # Public entry to create and save a new instance using the module dataset
    return _build_problem_instance(config, DATASET, save_dir=output_dir)


# ---------- Instance summary -------------------------------------------------


def _instance_summary(instance: PersonalAssistantInstance) -> Dict[str, Any]:
    # Build a JSON-serializable snapshot of the instance (agents, variables, factors, schemas)
    problem = instance.problem
    agent_to_vars = {
        agent_id: [var.name for var in problem.agent_variables(agent_id)]
        for agent_id in problem.agents.keys()
    }

    agents_block: List[Dict[str, Any]] = []
    for agent_id, spec in problem.agents.items():
        agents_block.append(
            {
                "id": agent_id,
                "name": spec.name,
                "instruction": instance.instructions[agent_id],
                "variables": agent_to_vars[agent_id],
                "assets": {"image": instance.collages.get(agent_id)},
            }
        )

    variables_block: List[Dict[str, Any]] = []
    for var_name, var_spec in problem.variables.items():
        variables_block.append(
            {
                "name": var_name,
                "owner": var_spec.owner,
                "domain": list(var_spec.domain),
                "description": var_spec.description,
            }
        )

    factors_block: List[Dict[str, Any]] = []
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

    return {
        "problem_name": problem.name,
        "instance_id": instance.config.output_stem,
        "max_utility": instance.max_utility,
        "instruction_mode": instance.config.instruction_mode.value,
        "metadata": {
            "num_agents": len(problem.agents),
            "num_variables": len(problem.variables),
            "num_factors": len(problem.list_factors()),
        },
        "agent_variable_map": agent_to_vars,
        "agents": agents_block,
        "variables": variables_block,
        "factors": factors_block,
        "schemas": {
            "joint_assignment": problem.joint_assignment_schema(),
            "agents": {agent_id: problem.agent_schema(agent_id) for agent_id in problem.agents.keys()},
        },
    }


# ---------- Module-level defaults -------------------------------------------


# Handy defaults for quick use/testing at import time
DEFAULT_CONFIG = PersonalAssistantConfig(
    num_agents=3,
    density=0.3,
    min_outfits_per_agent=4,
    max_outfits_per_agent=6,
    rng_seed=7,
    instruction_mode=InstructionMode.TEXT,
    include_collages=False,
    output_stem="personal_assistant_default",
)

DEFAULT_INSTANCE = _build_problem_instance(DEFAULT_CONFIG, DATASET, save_dir=None)

PROBLEM = DEFAULT_INSTANCE.problem
AGENT_SCHEMAS: Dict[str, Dict[str, Any]] = {
    agent_id: PROBLEM.agent_schema(agent_id) for agent_id in PROBLEM.agents.keys()
}
JOINT_ASSIGNMENT_SCHEMA: Dict[str, Any] = PROBLEM.joint_assignment_schema()
MAX_UTILITY = float(
    # Each unary factor contributes 1; pairwise contributes 2 (both sides)
    sum(1 if len(f.scope) == 1 else 2 for f in PROBLEM.list_factors())
)


def eval(joint_assignment: Mapping[str, Any]) -> Dict[str, Any]:
    # Evaluate a joint assignment against the default problem and expand chosen outfits
    raw = PROBLEM.eval(joint_assignment)
    result = dict(raw)
    result["min_utility"] = 0.0
    result["max_utility"] = MAX_UTILITY
    if result.get("valid"):
        expanded: Dict[str, Dict[str, Any]] = {}
        for var_name, choice in joint_assignment.items():
            spec = PROBLEM.variables.get(var_name)
            if spec is None:
                continue
            agent_id = spec.owner
            outfit = _choice_to_outfit(DEFAULT_INSTANCE.wardrobe, agent_id, choice)
            expanded[agent_id] = {
                "variable": var_name,
                "choice": int(choice),
                "outfit": asdict(outfit),
            }
        result["outfits"] = expanded
        if result.get("total_utility") is not None:
            result["total_utility"] = float(result["total_utility"])
    return result


def format_neighbour_assignments(*, problem: ProblemDefinition, shared_assignment: Mapping[str, Any]) -> Mapping[str, str]:
    """
    Render neighbour assignments as human-readable outfit descriptions (e.g., "2 (red suit)").
    Returns a mapping from variable name to rendered string.
    """
    instance = getattr(problem, "_personal_assistant_instance", DEFAULT_INSTANCE)
    render: Dict[str, str] = {}
    for var_name, choice in shared_assignment.items():
        spec = problem.variables.get(var_name)
        if spec is None:
            continue
        agent_id = spec.owner
        try:
            outfit = _choice_to_outfit(instance.wardrobe, agent_id, choice)
            render[var_name] = f"{int(choice)} ({outfit.color} {outfit.article})"
        except Exception:
            try:
                render[var_name] = str(choice)
            except Exception:
                pass
    return render


def wardrobe_for(agent_id: str) -> Sequence[Dict[str, Any]]:
    # Get the default instance wardrobe for a given agent as simple dicts
    outfits = DEFAULT_INSTANCE.wardrobe.get(agent_id)
    if outfits is None:
        raise KeyError(f"Unknown agent {agent_id!r} in default instance.")
    return [asdict(outfit) for outfit in outfits]
