from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "traj_train.npy"
TEST_PATH = ROOT / "data" / "traj_analysis.npy"
OUT_DIR = ROOT / "paper_reproductions" / "linot2021_ks"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"


def pca_reconstruction_curve(x_train: np.ndarray, x_test: np.ndarray, dims: list[int]) -> dict[str, list[float]]:
    mean = x_train.mean(axis=0)
    x_train_c = x_train - mean
    x_test_c = x_test - mean

    # Right singular vectors are the PCA directions in state space.
    _, _, vt = np.linalg.svd(x_train_c, full_matrices=False)
    mse_values: list[float] = []

    for d in dims:
        basis = vt[:d].T
        z_test = x_test_c @ basis
        x_rec = z_test @ basis.T + mean
        mse = float(np.mean((x_rec - x_test) ** 2))
        mse_values.append(mse)

    return {"dims": dims, "mse": mse_values}


def save_plot(curve: dict[str, list[float]], path: Path) -> None:
    dims = curve["dims"]
    mse = curve["mse"]
    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(dims, mse, "o-", lw=2, ms=6, color="C0")
    plt.xlabel("Latent dimension d")
    plt.ylabel("Test reconstruction MSE")
    plt.title("Figure 2 baseline for L = 22")
    plt.grid(True, alpha=0.3)
    plt.xticks(dims[::2] if len(dims) > 12 else dims)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(TRAIN_PATH)
    x_test = np.load(TEST_PATH)

    dims = list(range(1, 25))
    curve = pca_reconstruction_curve(x_train, x_test, dims)

    save_plot(curve, FIG_DIR / "figure2_l22_basic_d.png")

    summary = {
        "domain_length": 22.0,
        "state_dim": int(x_train.shape[1]),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "curve": curve,
        "best_dim_by_min_mse": int(dims[int(np.argmin(curve["mse"]))]),
    }
    with (RES_DIR / "figure2_l22_basic_d.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(FIG_DIR / "figure2_l22_basic_d.png")
    print(RES_DIR / "figure2_l22_basic_d.json")


if __name__ == "__main__":
    main()
