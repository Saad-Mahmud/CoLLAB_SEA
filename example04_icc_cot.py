#!/usr/bin/env python3
from __future__ import annotations

"""
Example 03 — Run ICC_CoT (batched thinking + sequential JSON) with vLLM or GPT (root)

What it shows
- How to generate an instance in-memory
- How to run ICC_CoT using ThinkingAgent
- vLLM/GPT selection with clear errors if config is missing

This variant reads provider settings from a JSON config (e.g., configs/vllm_server.json or configs/gpt_api.json)
to keep shell scripts simple.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agent_layer.agent_thinking import ThinkingAgent
from communication_layer import ICCThinkingProtocol


def _choose_block(cfg: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    block = cfg.get(mode)
    if isinstance(block, dict):
        merged = dict(cfg)
        merged.update(block)
        merged.pop("text", None)
        merged.pop("image", None)
        return merged
    return cfg


def build_api(
    provider: str,
    *,
    base_url: Optional[str],
    model: Optional[str],
    api_key: Optional[str],
    provider_config: Optional[Path] = None,
    config_mode: str = "text",
):
    if provider == "vllm":
        if provider_config is not None:
            try:
                cfg = json.loads(provider_config.read_text(encoding="utf-8"))
                section = _choose_block(cfg, config_mode)
                host = section.get("host", "0.0.0.0")
                port = section.get("port", 8000)
                base_url = f"http://{host}:{port}"
                model = section.get("model", model)
            except Exception as exc:
                raise SystemExit(f"[Error] Failed to read provider-config for vLLM: {exc}") from exc
        if not base_url:
            raise SystemExit("[Error] vLLM selected but --base-url is missing.")
        if not model:
            raise SystemExit("[Error] vLLM selected but --model is missing.")
        from api_layer import VLLMServerAPI
        print("[Info] Using vLLM server — no connectivity probe performed (set --base-url correctly).")
        return VLLMServerAPI(base_url=base_url, model=model)
    if provider == "gpt":
        if provider_config is not None:
            try:
                cfg = json.loads(provider_config.read_text(encoding="utf-8"))
                model = cfg.get("model", model)
                api_key = cfg.get("openai_api_key", api_key)
            except Exception as exc:
                raise SystemExit(f"[Error] Failed to read provider-config for GPT: {exc}") from exc
        if not model:
            raise SystemExit("[Error] GPT selected but --model is missing.")
        from api_layer.openai_api import OpenAIChatAPI
        if api_key is None:
            print("[Warning] GPT selected but --api-key not supplied; relying on OPENAI_API_KEY in environment.")
        return OpenAIChatAPI(model=model, api_key=api_key)
    from api_layer import DebugLLMAPI
    print("[Info] Using DebugLLMAPI (no external calls).")
    return DebugLLMAPI()


def load_instance(instance_dir: Path) -> Any:
    pkls = sorted(instance_dir.glob("*.pkl"))
    if not pkls:
        raise SystemExit(f"[Error] No pickle found under {instance_dir}")
    with open(pkls[0], "rb") as pf:
        return pickle.load(pf)


def build_agents(instance: Any, api) -> List[ThinkingAgent]:
    problem = instance.problem
    agents: List[ThinkingAgent] = []
    for spec in problem.agents.values():
        agents.append(
            ThinkingAgent(
                agent_id=spec.agent_id,
                name=spec.name,
                instruction=getattr(instance, "instructions", {}).get(spec.agent_id, spec.instruction),
                api=api,
                decision_schema=problem.agent_schema(spec.agent_id),
                max_retries=2,
            )
        )
    return agents


def main() -> None:
    ap = argparse.ArgumentParser(description="Example 04: Run ICC_CoT (thinking) with vLLM or GPT on a saved instance.")
    ap.add_argument("instance_dir", type=Path, help="Directory containing the saved instance JSON/PKL (from example00_generate.py).")
    ap.add_argument("--provider", choices=("vllm", "gpt", "debug"), default="debug")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--provider-config", type=Path, default=None, help="Path to provider config JSON (vLLM/GPT)")
    ap.add_argument("--config-mode", choices=("text", "image"), default="text", help="Which section of provider-config to use when present (vLLM)")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    instance = load_instance(args.instance_dir)
    api = build_api(
        args.provider,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        provider_config=args.provider_config,
        config_mode=args.config_mode,
    )
    agents = build_agents(instance, api)

    problem_name = getattr(getattr(instance, "problem", None), "name", "")
    show_neigh = problem_name != "smart_grid"
    show_factor = problem_name == "smart_grid"
    protocol = ICCThinkingProtocol(
        max_rounds=args.rounds,
        thinking_max_tokens=384,
        thinking_temperature=0.3,
        decision_max_tokens=256,
        decision_temperature=0.1,
        show_neighbour_assignments=show_neigh,
        show_factor_information=show_factor,
        batch_thinking=True,
    )

    print(f"[Example04] Running ICC_CoT | provider={args.provider} | rounds={args.rounds} | instance_dir={args.instance_dir}")
    result = protocol.run(
        instance.problem,
        agents,
        rounds=args.rounds,
        static_context={"note": "examples/example03_icc_cot"},
    )
    eval_: Dict[str, Any] = result.evaluation or {}
    total = float(eval_.get("total_utility", 0.0) or 0.0)
    # Prefer bounds from problem-layer evaluation payload; otherwise fall back to instance metadata
    min_u = eval_.get("min_utility")
    max_u = eval_.get("max_utility")
    if min_u is None or max_u is None:
        if problem_name == "smart_grid":
            min_u = getattr(instance, "min_utility", None)
            max_u = getattr(instance, "max_utility", None)
        elif problem_name in ("personal_assistant", "meeting_scheduling"):
            min_u = 0.0
            max_u = getattr(instance, "max_utility", None)
    ratio = None
    if max_u is not None:
        mv = 0.0 if min_u is None else float(min_u)
        denom = float(max_u) - mv
        ratio = (total - mv) / denom if abs(denom) > 1e-9 else None
    ratio_txt = f"{ratio:.3f}" if ratio is not None else "N/A"
    print(f"  -> total={total:.2f} | min={min_u} | max={max_u} | ratio={ratio_txt}")


if __name__ == "__main__":
    main()
