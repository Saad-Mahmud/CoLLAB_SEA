#!/usr/bin/env python3
from __future__ import annotations

"""
Example 00 — Generate and save problem instances (config driven)

What it does
- Reads one JSON config file that specifies parameters for each problem
- Generates all three problem instances and saves JSON + PKL under a fixed root
- Supports optional image-enabled generation if declared in the config

CLI: only the config file path is accepted (no other flags)

No LLM server or API is needed for this example.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    try:
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return default


def _gen_pa(cfg: Mapping[str, Any], out: Path, *, images: bool) -> Any:
    from problem_layer.personal_assistant.problem import PersonalAssistantConfig, InstructionMode, generate_instance
    out.mkdir(parents=True, exist_ok=True)
    num_agents = int(cfg.get("num_agents", 6))
    density = float(cfg.get("density", 0.3))
    min_outfits = int(cfg.get("min_outfits_per_agent", 4))
    max_outfits = int(cfg.get("max_outfits_per_agent", 6))
    rng_seed = int(cfg.get("seed", 7))
    # instruction_mode may be specified explicitly; otherwise infer from images
    mode_txt = str(cfg.get("instruction_mode", "")).strip().lower()
    if mode_txt in ("image", "img", "vision"):
        mode = InstructionMode.IMAGE
    elif mode_txt in ("text", "txt"):
        mode = InstructionMode.TEXT
    else:
        mode = InstructionMode.IMAGE if images else InstructionMode.TEXT
    include_collages = _as_bool(cfg.get("include_collages"), default=images)
    inst_cfg = PersonalAssistantConfig(
        num_agents=num_agents,
        density=density,
        min_outfits_per_agent=min_outfits,
        max_outfits_per_agent=max_outfits,
        instruction_mode=mode,
        include_collages=include_collages,
        output_stem="example_pa",
        rng_seed=rng_seed,
    )
    return generate_instance(inst_cfg, out)


def _gen_ms(cfg: Mapping[str, Any], out: Path, *, images: bool) -> Any:
    from problem_layer.meeting_scheduling.problem import MeetingSchedulingConfig, InstructionMode, generate_instance
    out.mkdir(parents=True, exist_ok=True)
    num_agents = int(cfg.get("num_agents", 6))
    num_meetings = int(cfg.get("num_meetings", 6))
    timeline = int(cfg.get("timeline_length", 12))
    soft_ratio = float(cfg.get("soft_meeting_ratio", 0.6))
    min_part = int(cfg.get("min_participants", 2))
    max_part = int(cfg.get("max_participants", 4))
    rng_seed = int(cfg.get("seed", 123))
    mode_txt = str(cfg.get("instruction_mode", "")).strip().lower()
    if mode_txt in ("image", "img", "vision"):
        mode = InstructionMode.IMAGE
    elif mode_txt in ("text", "txt"):
        mode = InstructionMode.TEXT
    else:
        mode = InstructionMode.IMAGE if images else InstructionMode.TEXT
    include_timelines = _as_bool(cfg.get("include_timelines"), default=images)
    inst_cfg = MeetingSchedulingConfig(
        num_agents=num_agents,
        num_meetings=num_meetings,
        timeline_length=timeline,
        soft_meeting_ratio=soft_ratio,
        min_participants=min_part,
        max_participants=max_part,
        instruction_mode=mode,
        include_timelines=include_timelines,
        output_stem="example_ms",
        rng_seed=rng_seed,
    )
    return generate_instance(inst_cfg, out)


def _gen_sg(cfg: Mapping[str, Any], out: Path, *, images: bool) -> Any:
    from problem_layer.smart_grid.problem import SmartGridConfig, InstructionMode, generate_instance
    out.mkdir(parents=True, exist_ok=True)
    num_agents = int(cfg.get("num_agents", 6))
    timeline = int(cfg.get("timeline_length", 24))
    min_src = int(cfg.get("min_sources_per_agent", 2))
    max_src = int(cfg.get("max_sources_per_agent", 3))
    min_m = int(cfg.get("min_machines_per_agent", 3))
    max_m = int(cfg.get("max_machines_per_agent", 6))
    rng_seed = int(cfg.get("seed", 900))
    mode_txt = str(cfg.get("instruction_mode", "")).strip().lower()
    if mode_txt in ("image", "img", "vision"):
        mode = InstructionMode.IMAGE
    elif mode_txt in ("text", "txt"):
        mode = InstructionMode.TEXT
    else:
        mode = InstructionMode.IMAGE if images else InstructionMode.TEXT
    include_charts = _as_bool(cfg.get("include_charts"), default=images)
    inst_cfg = SmartGridConfig(
        num_agents=num_agents,
        timeline_length=timeline,
        min_sources_per_agent=min_src,
        max_sources_per_agent=max_src,
        min_machines_per_agent=min_m,
        max_machines_per_agent=max_m,
        instruction_mode=mode,
        include_charts=include_charts,
        output_stem="example_sg",
        rng_seed=rng_seed,
    )
    return generate_instance(inst_cfg, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Example 00: Generate and save instances for all problems from a JSON config (single arg).")
    ap.add_argument("config", type=Path, help="Path to problems JSON config (e.g., configs/problems.json)")
    args = ap.parse_args()

    # Load config (one file, three sections for the problems). Optional top-level keys are allowed.
    cfg_root: Mapping[str, Any] = {}
    if not args.config.exists():
        raise SystemExit(f"[Error] Config file not found: {args.config}")
    try:
        cfg_root = json.loads(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[Error] Failed to read config {args.config}: {exc}") from exc

    # Global images toggle (can be overridden in per-problem sections)
    images_global = _as_bool(cfg_root.get("images"), default=False)

    # Fixed output location
    output_root = Path("outputs/examples/01_generate")
    output_root.mkdir(parents=True, exist_ok=True)

    problems = (
        ("personal_assistant", _gen_pa),
        ("meeting_scheduling", _gen_ms),
        ("smart_grid", _gen_sg),
    )
    for name, gen in problems:
        section = cfg_root.get(name, {}) if isinstance(cfg_root, Mapping) else {}
        # Allow per-problem override of images via section["images"]
        images = _as_bool(section.get("images"), default=images_global)
        out_dir = output_root / name
        print(f"[Generate] {name} (images={images}) ...")
        inst = gen(section, out_dir, images=images)
        prob = inst.problem
        print(f"  - instance_id: {inst.config.output_stem}")
        print(f"  - saved JSON:  {inst.json_path}")
        print(f"  - saved PKL:   {inst.pickle_path}")
        print(f"  - agents:      {len(prob.agents)} | variables: {len(prob.variables)} | factors: {len(prob.list_factors())}")


if __name__ == "__main__":
    main()
