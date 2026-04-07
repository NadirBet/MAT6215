"""
experiment_log.py - lightweight experiment/performance logging
==============================================================
Appends one JSON record per event to `data/experiment_log.jsonl`.
Designed to capture CPU-first experiment timings and results now so they can
be compared later against WSL/GPU reruns.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from datetime import datetime, timezone
from typing import Any


DEFAULT_LOG_PATH = "data/experiment_log.jsonl"


def _jsonable(value: Any) -> Any:
    """Convert nested values to something JSON serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]

    # numpy / jax scalar-ish objects
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    # arrays: keep only light metadata by default
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        except Exception:
            pass

    return str(value)


def _runtime_context() -> dict[str, Any]:
    ctx = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pid": os.getpid(),
    }
    try:
        import jax

        ctx["jax_version"] = jax.__version__
        ctx["jax_backend"] = jax.default_backend()
        ctx["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        ctx["jax_error"] = str(exc)
    return ctx


def log_event(task: str,
              stage: str,
              config: dict[str, Any] | None = None,
              metrics: dict[str, Any] | None = None,
              timings: dict[str, Any] | None = None,
              notes: str | None = None,
              log_path: str = DEFAULT_LOG_PATH) -> None:
    """
    Append one event to the experiment log.

    Parameters:
        task: task id / script family, e.g. "T3"
        stage: event label, e.g. "epoch_checkpoint" or "variant_complete"
        config: hyperparameters / run settings
        metrics: scalar results for later comparison
        timings: elapsed wall-clock timings in seconds
        notes: optional free-text note
        log_path: jsonl file to append to
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "stage": stage,
        "config": _jsonable(config or {}),
        "metrics": _jsonable(metrics or {}),
        "timings": _jsonable(timings or {}),
        "notes": notes,
        "runtime": _runtime_context(),
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
