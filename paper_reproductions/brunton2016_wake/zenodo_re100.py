from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class FlowMetadata:
    reynolds: float
    reduced_velocity: float
    n_times: int
    n_nodes: int


def _coerce_path(path: str | Path) -> Path:
    return Path(path)


def read_metadata(path: str | Path) -> FlowMetadata:
    path = _coerce_path(path)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        reynolds, reduced_velocity = map(float, f.readline().strip().split())
        f.readline()
        n_times, n_nodes = map(int, f.readline().strip().split())
    return FlowMetadata(
        reynolds=reynolds,
        reduced_velocity=reduced_velocity,
        n_times=n_times,
        n_nodes=n_nodes,
    )


def _skip_header(f) -> FlowMetadata:
    reynolds, reduced_velocity = map(float, f.readline().strip().split())
    f.readline()
    n_times, n_nodes = map(int, f.readline().strip().split())
    f.readline()
    return FlowMetadata(
        reynolds=reynolds,
        reduced_velocity=reduced_velocity,
        n_times=n_times,
        n_nodes=n_nodes,
    )


def _skip_frame_lines(f, n_nodes: int) -> None:
    deque(islice(f, n_nodes), maxlen=0)


def load_snapshots(
    path: str | Path,
    frame_indices: Sequence[int] | None = None,
    dtype: np.dtype | str = np.float32,
    include_pressure: bool = True,
    validate_mesh: bool = True,
) -> dict[str, np.ndarray | FlowMetadata]:
    path = _coerce_path(path)
    dtype = np.dtype(dtype)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        meta = _skip_header(f)

        if frame_indices is None:
            selected = set(range(meta.n_times))
            frame_order = list(range(meta.n_times))
        else:
            frame_order = sorted({int(i) for i in frame_indices})
            for idx in frame_order:
                if idx < 0 or idx >= meta.n_times:
                    raise IndexError(f"frame index out of range: {idx}")
            selected = set(frame_order)

        times: list[float] = []
        us: list[np.ndarray] = []
        vs: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        x_ref: np.ndarray | None = None
        y_ref: np.ndarray | None = None

        for frame_idx in range(meta.n_times):
            time_value = float(f.readline().strip())
            if frame_idx not in selected:
                _skip_frame_lines(f, meta.n_nodes)
                continue

            block = np.loadtxt(islice(f, meta.n_nodes), dtype=dtype)
            if block.shape != (meta.n_nodes, 5):
                raise ValueError(
                    f"unexpected block shape at frame {frame_idx}: {block.shape}"
                )

            x = block[:, 0].copy()
            y = block[:, 1].copy()

            if x_ref is None:
                x_ref = x
                y_ref = y
            elif validate_mesh:
                if not np.allclose(x_ref, x) or not np.allclose(y_ref, y):
                    raise ValueError(
                        f"mesh changed at frame {frame_idx}; expected fixed nodes"
                    )

            times.append(time_value)
            us.append(block[:, 2].copy())
            vs.append(block[:, 3].copy())
            if include_pressure:
                ps.append(block[:, 4].copy())

    result: dict[str, np.ndarray | FlowMetadata] = {
        "meta": meta,
        "times": np.asarray(times, dtype=dtype),
        "x": x_ref,
        "y": y_ref,
        "u": np.stack(us, axis=0),
        "v": np.stack(vs, axis=0),
    }
    if include_pressure:
        result["p"] = np.stack(ps, axis=0)
    return result


def save_compact_npz(
    path: str | Path,
    out_path: str | Path,
    frame_indices: Sequence[int] | None = None,
    dtype: np.dtype | str = np.float32,
    include_pressure: bool = False,
) -> Path:
    data = load_snapshots(
        path,
        frame_indices=frame_indices,
        dtype=dtype,
        include_pressure=include_pressure,
    )
    out_path = _coerce_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = data["meta"]
    savez_kwargs = {
        "reynolds": np.array(meta.reynolds, dtype=np.float32),
        "reduced_velocity": np.array(meta.reduced_velocity, dtype=np.float32),
        "times": data["times"],
        "x": data["x"],
        "y": data["y"],
        "u": data["u"],
        "v": data["v"],
    }
    if include_pressure and "p" in data:
        savez_kwargs["p"] = data["p"]

    np.savez_compressed(out_path, **savez_kwargs)
    return out_path
