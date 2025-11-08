#!/usr/bin/env python3
from __future__ import annotations

"""
Example 02 — Run a local vLLM server from JSON config

Usage
  python example02_run_vllm.py --mode text --config configs/vllm_server.json
  python example02_run_vllm.py --mode image --config configs/vllm_server.json

The config file can be either:
  - flat (model, host, port, tensor_parallel_size, dtype, gpu_memory_utilization, max_model_len), or
  - nested with "text" and/or "image" blocks that override the flat defaults.

This script starts api_layer/vllm_server.py as a subprocess, prints the command,
then waits until interrupted (Ctrl+C). On exit it stops the server.
"""

import argparse
import json
import shlex
import signal
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


def _choose_block(cfg: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    block = cfg.get(mode)
    if isinstance(block, dict):
        merged = dict(cfg)
        merged.update(block)
        # Remove top-level sub-blocks to avoid confusion downstream
        merged.pop("text", None)
        merged.pop("image", None)
        return merged
    # Fall back to flat config
    return cfg


def _build_cmd(section: Mapping[str, Any], *, mode: str) -> list[str]:
    model = section.get("model")
    host = section.get("host", "0.0.0.0")
    port = section.get("port", 8000)
    tp = section.get("tensor_parallel_size", 1)
    dtype = section.get("dtype", "bfloat16")
    gpu_util = section.get("gpu_memory_utilization")
    max_len = section.get("max_model_len")

    if not model:
        raise SystemExit("[Error] 'model' is required in the vLLM config.")

    server_script = "vllm_server_image.py" if mode == "image" else "vllm_server.py"
    cmd = [
        sys.executable,
        str(Path("api_layer") / server_script),
        "--model", str(model),
        "--tensor-parallel-size", str(tp),
        "--dtype", str(dtype),
        "--host", str(host),
        "--port", str(port),
    ]
    if gpu_util is not None and float(gpu_util) != 0.0:
        cmd += ["--gpu-memory-utilization", str(gpu_util)]
    if max_len not in (None, 0, "0"):
        cmd += ["--max-model-len", str(max_len)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Example 02: Run vLLM server from JSON config.")
    ap.add_argument("--mode", choices=("text", "image"), required=True,
                    help="Choose which server profile to use from the config (text or image block).")
    ap.add_argument("--config", type=Path, required=True, help="Path to vLLM server JSON config.")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"[Error] Config file not found: {args.config}")

    try:
        cfg = json.loads(args.config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"[Error] Failed to read JSON config: {exc}") from exc

    section = _choose_block(cfg, args.mode)
    cmd = _build_cmd(section, mode=args.mode)

    print("[vLLM] Launching server:")
    print("  ", " ".join(shlex.quote(c) for c in cmd))
    try:
        # Start server in its own process group so we can signal the whole group
        proc = subprocess.Popen(cmd, start_new_session=True)
    except Exception as exc:
        raise SystemExit(f"[Error] Failed to start vLLM server: {exc}") from exc

    print("[vLLM] PID=", proc.pid)
    print("[vLLM] Press Ctrl+C to stop.")

    def _kill_group(sig: int) -> None:
        try:
            if proc.poll() is None:
                os.killpg(proc.pid, sig)
        except Exception:
            pass

    # Forward SIGINT/SIGTERM to the child process group for graceful shutdown
    def _forward_signal(_signum, _frame):  # noqa: ANN001
        _kill_group(signal.SIGTERM)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[vLLM] Stopping ...")
        _kill_group(signal.SIGTERM)
        # Wait up to 8s, then escalate to SIGKILL for the whole group
        for _ in range(16):
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        if proc.poll() is None:
            print("[vLLM] Forcing SIGKILL to server group ...")
            _kill_group(signal.SIGKILL)
            try:
                proc.wait(timeout=3)
            except Exception:
                pass
    print("[vLLM] Exited.")


if __name__ == "__main__":
    main()
