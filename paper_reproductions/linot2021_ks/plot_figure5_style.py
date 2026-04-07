"""Linot Figure 5 style plot for our existing L=22 latent NODE rollout.

Panels:
  (a) true trajectory u(x, t)
  (b) predicted trajectory (latent NODE)
  (c) |u_pred - u_true|
  (d) ||u_pred - u_true|| vs t, raw and shift-minimized
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PKL = HERE / "results" / "figure3_l22_core.pkl"
OUT = HERE / "figures" / "figure5_style_l22_latent.png"
DT = 0.25
L = 22.0
U_CLIM = 3.0


def shift_min_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """For each time, return min over integer spatial shifts of ||pred(.+s) - true||."""
    n = true.shape[1]
    errs = np.empty(true.shape[0])
    for t in range(true.shape[0]):
        best = np.inf
        for s in range(n):
            d = np.roll(pred[t], s) - true[t]
            v = float(np.linalg.norm(d))
            if v < best:
                best = v
        errs[t] = best
    return errs


def main() -> None:
    with PKL.open("rb") as f:
        r = pickle.load(f)
    true = r["true_traj"]      # (T, 64)
    pred = r["latent_traj"]    # (T, 64)
    T = true.shape[0]
    t = np.arange(T) * DT
    x = np.linspace(-L / 2.0, L / 2.0, true.shape[1], endpoint=False)
    err = pred - true
    abs_err = np.abs(err)

    raw_norm = np.linalg.norm(err, axis=1)
    shifted_norm = shift_min_error(pred, true)

    fig, axes = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 0.9]})

    extent = [t[0], t[-1], x[0], x[-1]]
    im0 = axes[0].imshow(true.T, origin="lower", aspect="auto", cmap="RdBu_r",
                         vmin=-U_CLIM, vmax=U_CLIM, extent=extent, interpolation="nearest")
    axes[0].set_ylabel("x")
    axes[0].set_title("a) true trajectory")
    fig.colorbar(im0, ax=axes[0], pad=0.01).set_label("u")

    im1 = axes[1].imshow(pred.T, origin="lower", aspect="auto", cmap="RdBu_r",
                         vmin=-U_CLIM, vmax=U_CLIM, extent=extent, interpolation="nearest")
    axes[1].set_ylabel("x")
    axes[1].set_title("b) predicted trajectory (latent NODE, d=8)")
    fig.colorbar(im1, ax=axes[1], pad=0.01).set_label("u")

    aclim = float(np.percentile(abs_err, 99))
    im2 = axes[2].imshow(abs_err.T, origin="lower", aspect="auto", cmap="magma",
                         vmin=0.0, vmax=aclim, extent=extent, interpolation="nearest")
    axes[2].set_ylabel("x")
    axes[2].set_title("c) |u_pred - u_true|")
    fig.colorbar(im2, ax=axes[2], pad=0.01).set_label("|err|")

    axes[3].plot(t, raw_norm, label="Unshifted", color="C0")
    axes[3].plot(t, shifted_norm, label="Shifted", color="C1", linestyle="--")
    axes[3].set_xlabel("t")
    axes[3].set_ylabel("||u_pred - u_true||")
    axes[3].set_title("d) error norm vs t (raw and shift-minimized)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Linot Figure 5 style: L=22 latent NODE, 500 steps (dt=0.25)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170)
    plt.close(fig)
    print(f"Saved {OUT}")
    print(f"raw   norm: max={raw_norm.max():.3f}  final={raw_norm[-1]:.3f}")
    print(f"shift norm: max={shifted_norm.max():.3f}  final={shifted_norm[-1]:.3f}")


if __name__ == "__main__":
    main()
