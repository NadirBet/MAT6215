"""Post-processing diagnostics for Linot Figure 3 results."""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PKL_PATH = Path(__file__).resolve().parent / "results" / "figure3_l22_core.pkl"
FIG_DIR = Path(__file__).resolve().parent / "figures"
DT = 0.25
L = 22.0


def main():
    with PKL_PATH.open("rb") as f:
        r = pickle.load(f)

    true = r["true_traj"]          # (500, 64)
    latent = r["latent_traj"]
    physical = r["physical_traj"]
    fourier = r["fourier_traj"]
    T = true.shape[0]
    t = np.arange(T) * DT

    # --- relative error vs time ---
    def rel_err(pred):
        numer = np.linalg.norm(pred - true, axis=1)
        denom = np.linalg.norm(true, axis=1) + 1e-12
        return numer / denom

    re_lat = rel_err(latent)
    re_phy = rel_err(physical)
    re_fou = rel_err(fourier)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t, re_lat, label="Latent NODE")
    ax.semilogy(t, re_phy, label="Full physical")
    ax.semilogy(t, re_fou, label="Full Fourier")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="Error = 1")
    ax.set_xlabel("t")
    ax.set_ylabel("Relative error ||u_pred - u_true|| / ||u_true||")
    ax.set_title("Linot Figure 3 — relative prediction error vs time (L=22)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_relative_error.png", dpi=180)
    plt.close(fig)
    print("Saved figure3_relative_error.png")

    # --- time-averaged power spectrum ---
    q = np.fft.rfftfreq(true.shape[1], d=L / true.shape[1]) * 2 * np.pi

    def power_spectrum(traj):
        return np.mean(np.abs(np.fft.rfft(traj, axis=1)) ** 2, axis=0)

    ps_true = power_spectrum(true)
    ps_lat = power_spectrum(latent)
    ps_phy = power_spectrum(physical)
    ps_fou = power_spectrum(fourier)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(q, ps_true, "k-", linewidth=2, label="True")
    ax.semilogy(q, ps_lat, label="Latent NODE")
    ax.semilogy(q, ps_phy, label="Full physical")
    ax.semilogy(q, ps_fou, label="Full Fourier")
    ax.set_xlabel("Wavenumber q")
    ax.set_ylabel("E(q) = <|û_q|²>")
    ax.set_title("Linot Figure 3 — time-averaged power spectrum (L=22)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure3_power_spectrum.png", dpi=180)
    plt.close(fig)
    print("Saved figure3_power_spectrum.png")

    # --- summary ---
    def divergence_time(re, threshold=1.0):
        idx = np.where(re > threshold)[0]
        return float(t[idx[0]]) if len(idx) > 0 else float("inf")

    print(f"\nDivergence time (relative error > 1):")
    print(f"  Latent NODE : {divergence_time(re_lat):.1f} t-units")
    print(f"  Full physical: {divergence_time(re_phy):.1f} t-units")
    print(f"  Full Fourier : {divergence_time(re_fou):.1f} t-units")


if __name__ == "__main__":
    main()
