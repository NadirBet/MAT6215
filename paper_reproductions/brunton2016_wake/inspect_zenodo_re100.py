from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_reproductions.brunton2016_wake.zenodo_re100 import load_snapshots, read_metadata


RAW_PATH = Path(
    r"c:\Users\nadir\Desktop\MAT6215\paper_reproductions\brunton2016_wake\data\fixed_cylinder_atRe100"
)


def main() -> None:
    meta = read_metadata(RAW_PATH)
    print("Metadata")
    print(f"  Re: {meta.reynolds}")
    print(f"  Ur: {meta.reduced_velocity}")
    print(f"  Nt: {meta.n_times}")
    print(f"  N_nodes: {meta.n_nodes}")

    sample = load_snapshots(
        RAW_PATH,
        frame_indices=[0, 1, meta.n_times - 1],
        dtype=np.float32,
        include_pressure=False,
        validate_mesh=True,
    )
    times = sample["times"]
    print("\nSampled frames")
    print(f"  times: {times.tolist()}")
    if len(times) >= 2:
        print(f"  dt(first): {float(times[1] - times[0]):.6f}")

    x = sample["x"]
    y = sample["y"]
    u = sample["u"]
    v = sample["v"]
    speed0 = np.sqrt(u[0] ** 2 + v[0] ** 2)

    print("\nMesh / field summary")
    print(f"  x-range: [{float(x.min()):.6f}, {float(x.max()):.6f}]")
    print(f"  y-range: [{float(y.min()):.6f}, {float(y.max()):.6f}]")
    print(f"  snapshot shape (u): {u.shape}")
    print(f"  speed range at first sampled frame: [{float(speed0.min()):.6f}, {float(speed0.max()):.6f}]")


if __name__ == "__main__":
    main()
