from __future__ import annotations

import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from paper_reproductions.linot2021_ks.data_utils import OUT_DIR, load_cached_dataset

jax.config.update("jax_enable_x64", True)


FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
L_VALUE = 22.0
STATE_DIM = 64
LATENT_DIM = 8
DT = 0.25
ROLLOUT_STEPS = 500
U_CLIM = 3.0


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_data(cls, x: np.ndarray, *, floor: float = 1e-6) -> "Standardizer":
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < floor, 1.0, std)
        return cls(mean=mean, std=std)

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self.mean


@dataclass
class ProgressTracker:
    path: Path
    state: dict

    @classmethod
    def create(cls, path: Path, **meta):
        tracker = cls(path=path, state={"meta": meta, "steps": [], "status": "running"})
        tracker.flush()
        return tracker

    def mark(self, step: str, **metrics):
        entry = {"step": step, "time": time.time(), "metrics": metrics}
        self.state["steps"].append(entry)
        self.flush()

    def finish(self, **metrics):
        self.state["status"] = "completed"
        self.state["final"] = metrics
        self.flush()

    def fail(self, **metrics):
        self.state["status"] = "failed"
        self.state["final"] = metrics
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def copy_params(params):
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def init_mlp(key, sizes, scale=0.05):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (n_in, n_out) in zip(keys, zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (n_in, n_out)) * scale / jnp.sqrt(max(n_in, 1))
        b = jnp.zeros((n_out,))
        params.append((w, b))
    return params


def mlp_forward(params, x):
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = jax.nn.sigmoid(x)
    return x


def fit_pca(x_train: np.ndarray):
    mean = x_train.mean(axis=0)
    x_c = x_train - mean
    _, _, vt = np.linalg.svd(x_c, full_matrices=False)
    return mean, vt


def encode_pca(x: np.ndarray, mean: np.ndarray, vt: np.ndarray) -> np.ndarray:
    return (x - mean) @ vt.T


def decode_pca(coeff: np.ndarray, mean: np.ndarray, vt: np.ndarray) -> np.ndarray:
    return coeff @ vt + mean


def fourier_real_encode(x: np.ndarray) -> np.ndarray:
    z = np.fft.rfft(np.asarray(x), axis=-1)
    head = z[..., :1].real
    mid = z[..., 1:-1]
    tail = z[..., -1:].real
    return np.concatenate([head, mid.real, mid.imag, tail], axis=-1)


def fourier_real_decode(v: np.ndarray, n: int = STATE_DIM) -> np.ndarray:
    v = np.asarray(v)
    r0 = v[..., :1]
    m = n // 2 - 1
    real_mid = v[..., 1:1 + m]
    imag_mid = v[..., 1 + m:1 + 2 * m]
    rny = v[..., -1:]
    z = np.concatenate([r0 + 0j, real_mid + 1j * imag_mid, rny + 0j], axis=-1)
    return np.fft.irfft(z, n=n, axis=-1)


def make_schedule(init_lr: float, epochs: int, updates_per_epoch: int):
    total_updates = max(epochs * updates_per_epoch, 1)
    boundaries = {
        max(int((2.0 / 3.0) * total_updates), 1): 0.2,
        max(int((5.0 / 6.0) * total_updates), 1): 0.2,
    }
    return optax.piecewise_constant_schedule(init_value=init_lr, boundaries_and_scales=boundaries)


def train_decoder(
    x_lat_train: np.ndarray,
    y_res_train: np.ndarray,
    x_lat_test: np.ndarray,
    y_res_test: np.ndarray,
    *,
    hidden_layers: tuple[int, ...] = (500,),
    epochs: int = 400,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    seed: int = 0,
    eval_every: int = 25,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    key = jax.random.PRNGKey(seed)
    params = init_mlp(key, [x_lat_train.shape[1], *hidden_layers, y_res_train.shape[1]], scale=0.1)
    n_train = len(x_lat_train)
    updates_per_epoch = max(math.ceil(n_train / batch_size), 1)
    opt = optax.adam(make_schedule(learning_rate, epochs, updates_per_epoch))
    opt_state = opt.init(params)

    x_train_j = jnp.array(x_lat_train)
    y_train_j = jnp.array(y_res_train)
    x_test_j = jnp.array(x_lat_test)
    y_test_j = jnp.array(y_res_test)
    rng = np.random.default_rng(seed)

    @jax.jit
    def loss_fn(p, xb, yb):
        pred = jax.vmap(lambda x: mlp_forward(p, x))(xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def train_step(p, state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, state = opt.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss

    history = {"epoch": [], "train_loss": [], "test_loss": []}
    best_test = float("inf")
    best_epoch = 0
    best_params = copy_params(params)

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            params, opt_state, _ = train_step(params, opt_state, x_train_j[idx], y_train_j[idx])
        if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            snapshot = {
                "epoch": epoch + 1,
                "train_loss": float(loss_fn(params, x_train_j, y_train_j)),
                "test_loss": float(loss_fn(params, x_test_j, y_test_j)),
            }
            if snapshot["test_loss"] < best_test:
                best_test = snapshot["test_loss"]
                best_epoch = snapshot["epoch"]
                best_params = copy_params(params)
                snapshot["improved"] = True
            else:
                snapshot["improved"] = False
            snapshot["best_epoch"] = best_epoch
            snapshot["best_test_loss"] = best_test
            history["epoch"].append(snapshot["epoch"])
            history["train_loss"].append(snapshot["train_loss"])
            history["test_loss"].append(snapshot["test_loss"])
            if progress_callback is not None:
                progress_callback(snapshot)

    history["best_epoch"] = best_epoch
    history["best_test_loss"] = best_test
    return best_params, history


def decoder_reconstruct(
    x_lat_norm: np.ndarray,
    decoder_params,
    latent_stats: Standardizer,
    tail_stats: Standardizer,
    mean: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    lead = latent_stats.decode(np.asarray(x_lat_norm))
    tail_norm = np.asarray(jax.vmap(lambda x: mlp_forward(decoder_params, x))(jnp.array(x_lat_norm)))
    tail = tail_stats.decode(tail_norm)
    coeff = np.concatenate([lead, tail], axis=-1)
    return decode_pca(coeff, mean, vt)


def train_node_map(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    hidden_width: int = 200,
    epochs: int = 1000,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    dt: float = DT,
    seed: int = 0,
    eval_every: int = 50,
    progress_callback: Optional[Callable[[dict], None]] = None,
):
    dim = x_train.shape[1]
    key = jax.random.PRNGKey(seed)
    params = init_mlp(key, [dim, hidden_width, hidden_width, hidden_width, dim], scale=0.05)

    n_train = len(x_train)
    updates_per_epoch = max(math.ceil(n_train / batch_size), 1)
    schedule = make_schedule(learning_rate, epochs, updates_per_epoch)
    opt = optax.adam(schedule)
    opt_state = opt.init(params)
    rng = np.random.default_rng(seed)

    x_train_j = jnp.array(x_train)
    y_train_j = jnp.array(y_train)
    x_test_j = jnp.array(x_test)
    y_test_j = jnp.array(y_test)

    def rhs(p, x):
        return mlp_forward(p, x)

    def step(p, x):
        k1 = rhs(p, x)
        k2 = rhs(p, x + 0.5 * dt * k1)
        k3 = rhs(p, x + 0.5 * dt * k2)
        k4 = rhs(p, x + dt * k3)
        return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    @jax.jit
    def loss_fn(p, xb, yb):
        pred = jax.vmap(lambda x: step(p, x))(xb)
        return jnp.mean((pred - yb) ** 2)

    @jax.jit
    def train_step(p, state, xb, yb):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb)
        updates, state = opt.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss

    history = {"epoch": [], "train_loss": [], "test_loss": []}
    best_test = float("inf")
    best_epoch = 0
    best_params = copy_params(params)

    for epoch in range(epochs):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = order[start:start + batch_size]
            params, opt_state, _ = train_step(params, opt_state, x_train_j[idx], y_train_j[idx])
        if epoch == 0 or (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            snapshot = {
                "epoch": epoch + 1,
                "train_loss": float(loss_fn(params, x_train_j, y_train_j)),
                "test_loss": float(loss_fn(params, x_test_j, y_test_j)),
            }
            if snapshot["test_loss"] < best_test:
                best_test = snapshot["test_loss"]
                best_epoch = snapshot["epoch"]
                best_params = copy_params(params)
                snapshot["improved"] = True
            else:
                snapshot["improved"] = False
            snapshot["best_epoch"] = best_epoch
            snapshot["best_test_loss"] = best_test
            history["epoch"].append(snapshot["epoch"])
            history["train_loss"].append(snapshot["train_loss"])
            history["test_loss"].append(snapshot["test_loss"])
            if progress_callback is not None:
                progress_callback(snapshot)

    history["best_epoch"] = best_epoch
    history["best_test_loss"] = best_test
    return best_params, history, step


def rollout(step_fn, params, x0: np.ndarray, n_steps: int) -> np.ndarray:
    traj = np.zeros((n_steps, x0.shape[-1]), dtype=np.float64)
    x = jnp.array(x0)
    for i in range(n_steps):
        x = step_fn(params, x)
        traj[i] = np.array(x)
    return traj


def save_stage_history(name: str, history: dict) -> None:
    save_json(RES_DIR / f"figure3_l22_core_{name}_history.json", history)


def plot_loss_panel(histories: dict[str, dict], path: Path):
    fig, axes = plt.subplots(1, len(histories), figsize=(4.8 * len(histories), 4.2))
    if len(histories) == 1:
        axes = [axes]
    for ax, (name, hist) in zip(axes, histories.items()):
        ax.semilogy(hist["epoch"], hist["train_loss"], label="Train")
        ax.semilogy(hist["epoch"], hist["test_loss"], label="Test")
        best_epoch = hist.get("best_epoch")
        best_loss = hist.get("best_test_loss")
        if best_epoch is not None and best_loss is not None:
            ax.scatter([best_epoch], [best_loss], color="black", s=20, zorder=3, label="Best test")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Linot Figure 3 core training losses")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_figure3(true_traj, latent_traj, phys_traj, fourier_traj, path: Path, L: float = L_VALUE):
    systems = [
        ("(a) True", true_traj),
        ("(b) Latent NODE", latent_traj),
        ("(c) Full physical", phys_traj),
        ("(d) Full Fourier", fourier_traj),
    ]
    windows = [
        ("0-50", 0, 50),
        ("450-500", 450, 500),
    ]
    x_coords = np.linspace(-L / 2.0, L / 2.0, true_traj.shape[1], endpoint=False)
    fig, axes = plt.subplots(len(systems), len(windows), figsize=(10.2, 8.8), sharey=True)
    mappable = None
    for row, (label, traj) in enumerate(systems):
        for col, (window_label, start, stop) in enumerate(windows):
            ax = axes[row, col]
            panel = traj[start:stop]
            t_axis = np.arange(start, stop)
            mappable = ax.imshow(
                panel.T,
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-U_CLIM,
                vmax=U_CLIM,
                extent=[t_axis[0], t_axis[-1], x_coords[0], x_coords[-1]],
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(window_label)
            if col == 0:
                ax.set_ylabel(f"{label}\nx")
            if row == len(systems) - 1:
                ax.set_xlabel("t")
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("u")
    fig.suptitle("Figure 3 for L = 22: KS field u(x,t)")
    fig.tight_layout(rect=(0, 0, 0.96, 0.97))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_figure4(phys_traj: np.ndarray, fourier_traj: np.ndarray, path: Path, L: float = L_VALUE):
    q = np.fft.rfftfreq(phys_traj.shape[1], d=L / phys_traj.shape[1]) * (2 * np.pi)

    def mode_mag(traj):
        return np.abs(np.fft.rfft(traj, axis=1))

    phys_mag = mode_mag(phys_traj)
    fourier_mag = mode_mag(fourier_traj)
    vmax = np.percentile(np.concatenate([phys_mag.ravel(), fourier_mag.ravel()]), 99)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    mappable = None
    for ax, mag, title in [
        (axes[0], phys_mag, "Full-space in real space"),
        (axes[1], fourier_mag, "Full-space in Fourier space"),
    ]:
        mappable = ax.pcolormesh(
            q,
            np.arange(mag.shape[0]),
            mag,
            shading="auto",
            cmap="magma",
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Wavenumber q")
    axes[0].set_ylabel("Step")
    cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.set_label("|u_q|")
    fig.suptitle("Figure 4 core spectral evolution")
    fig.tight_layout(rect=(0, 0, 0.97, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def stage_callback_factory(tracker: ProgressTracker, stage_name: str, history_key: str):
    partial = {"epoch": [], "train_loss": [], "test_loss": []}

    def callback(snapshot: dict) -> None:
        partial["epoch"].append(snapshot["epoch"])
        partial["train_loss"].append(snapshot["train_loss"])
        partial["test_loss"].append(snapshot["test_loss"])
        partial["best_epoch"] = snapshot["best_epoch"]
        partial["best_test_loss"] = snapshot["best_test_loss"]
        tracker.mark(stage_name, **snapshot)
        save_stage_history(history_key, partial)

    return callback


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker.create(
        RES_DIR / "figure3_l22_core_progress.json",
        domain_length=L_VALUE,
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        dt=DT,
        note="Uses cached Linot L=22 data, best-checkpoint selection, and paper-style u(x,t) plots.",
    )

    try:
        cached = load_cached_dataset(L_VALUE)
        if cached is None:
            raise FileNotFoundError("Missing cached Linot L=22 dataset. Run the dataset prep path first.")
        x_train, x_test, meta = cached

        mean, vt = fit_pca(x_train)
        coeff_train = encode_pca(x_train, mean, vt)
        coeff_test = encode_pca(x_test, mean, vt)
        d = LATENT_DIM

        latent_stats = Standardizer.from_data(coeff_train[:, :d])
        tail_stats = Standardizer.from_data(coeff_train[:, d:])
        phys_stats = Standardizer.from_data(x_train)
        fft_train = fourier_real_encode(x_train)
        fft_test = fourier_real_encode(x_test)
        fft_stats = Standardizer.from_data(fft_train)

        lead_train = latent_stats.encode(coeff_train[:, :d])
        lead_test = latent_stats.encode(coeff_test[:, :d])
        tail_train = tail_stats.encode(coeff_train[:, d:])
        tail_test = tail_stats.encode(coeff_test[:, d:])
        x_train_norm = phys_stats.encode(x_train)
        x_test_norm = phys_stats.encode(x_test)
        fft_train_norm = fft_stats.encode(fft_train)
        fft_test_norm = fft_stats.encode(fft_test)

        tracker.mark(
            "data_prepared",
            train_shape=list(x_train.shape),
            test_shape=list(x_test.shape),
            train_std=float(x_train.std()),
            test_std=float(x_test.std()),
            data_meta=meta,
            latent_std=list(np.round(latent_stats.std, 6)),
        )

        decoder_params, dec_hist = train_decoder(
            lead_train,
            tail_train,
            lead_test,
            tail_test,
            hidden_layers=(500,),
            epochs=500,
            batch_size=1024,
            learning_rate=1e-3,
            seed=1,
            eval_every=25,
            progress_callback=stage_callback_factory(tracker, "decoder_eval", "decoder"),
        )
        save_stage_history("decoder", dec_hist)
        decoder_recon = decoder_reconstruct(lead_test, decoder_params, latent_stats, tail_stats, mean, vt)
        decoder_mse = float(np.mean((decoder_recon - x_test) ** 2))
        tracker.mark(
            "decoder_trained",
            final_train_loss=dec_hist["train_loss"][-1],
            final_test_loss=dec_hist["test_loss"][-1],
            best_epoch=dec_hist["best_epoch"],
            best_test_loss=dec_hist["best_test_loss"],
            reconstruction_mse=decoder_mse,
        )

        latent_params, latent_hist, latent_step = train_node_map(
            lead_train[:-1],
            lead_train[1:],
            lead_test[:-1],
            lead_test[1:],
            hidden_width=200,
            epochs=2000,
            batch_size=1024,
            learning_rate=1e-3,
            dt=DT,
            seed=2,
            eval_every=50,
            progress_callback=stage_callback_factory(tracker, "latent_node_eval", "latent"),
        )
        save_stage_history("latent", latent_hist)
        tracker.mark(
            "latent_node_trained",
            final_train_loss=latent_hist["train_loss"][-1],
            final_test_loss=latent_hist["test_loss"][-1],
            best_epoch=latent_hist["best_epoch"],
            best_test_loss=latent_hist["best_test_loss"],
        )

        phys_params, phys_hist, phys_step = train_node_map(
            x_train_norm[:-1],
            x_train_norm[1:],
            x_test_norm[:-1],
            x_test_norm[1:],
            hidden_width=200,
            epochs=1200,
            batch_size=1024,
            learning_rate=1e-3,
            dt=DT,
            seed=3,
            eval_every=50,
            progress_callback=stage_callback_factory(tracker, "physical_node_eval", "physical"),
        )
        save_stage_history("physical", phys_hist)
        tracker.mark(
            "physical_node_trained",
            final_train_loss=phys_hist["train_loss"][-1],
            final_test_loss=phys_hist["test_loss"][-1],
            best_epoch=phys_hist["best_epoch"],
            best_test_loss=phys_hist["best_test_loss"],
        )

        fourier_params, fourier_hist, fourier_step = train_node_map(
            fft_train_norm[:-1],
            fft_train_norm[1:],
            fft_test_norm[:-1],
            fft_test_norm[1:],
            hidden_width=200,
            epochs=1200,
            batch_size=1024,
            learning_rate=1e-3,
            dt=DT,
            seed=4,
            eval_every=50,
            progress_callback=stage_callback_factory(tracker, "fourier_node_eval", "fourier"),
        )
        save_stage_history("fourier", fourier_hist)
        tracker.mark(
            "fourier_node_trained",
            final_train_loss=fourier_hist["train_loss"][-1],
            final_test_loss=fourier_hist["test_loss"][-1],
            best_epoch=fourier_hist["best_epoch"],
            best_test_loss=fourier_hist["best_test_loss"],
        )

        true_traj = x_test[1:ROLLOUT_STEPS + 1]
        latent_roll = rollout(latent_step, latent_params, lead_test[0], ROLLOUT_STEPS)
        latent_traj = decoder_reconstruct(latent_roll, decoder_params, latent_stats, tail_stats, mean, vt)

        phys_roll = rollout(phys_step, phys_params, x_test_norm[0], ROLLOUT_STEPS)
        phys_traj = phys_stats.decode(phys_roll)

        fft_roll = rollout(fourier_step, fourier_params, fft_test_norm[0], ROLLOUT_STEPS)
        fourier_traj = fourier_real_decode(fft_stats.decode(fft_roll), n=STATE_DIM)

        latent_rollout_mse = float(np.mean((latent_traj - true_traj) ** 2))
        physical_rollout_mse = float(np.mean((phys_traj - true_traj) ** 2))
        fourier_rollout_mse = float(np.mean((fourier_traj - true_traj) ** 2))
        tracker.mark(
            "rollouts_generated",
            decoder_mse=decoder_mse,
            latent_energy=float(np.mean(np.sum(latent_traj ** 2, axis=1))),
            physical_energy=float(np.mean(np.sum(phys_traj ** 2, axis=1))),
            fourier_energy=float(np.mean(np.sum(fourier_traj ** 2, axis=1))),
            latent_rollout_mse=latent_rollout_mse,
            physical_rollout_mse=physical_rollout_mse,
            fourier_rollout_mse=fourier_rollout_mse,
        )

        plot_loss_panel(
            {
                "Decoder": dec_hist,
                "Latent NODE": latent_hist,
                "Physical NODE": phys_hist,
                "Fourier NODE": fourier_hist,
            },
            FIG_DIR / "figure3_l22_core_losses.png",
        )
        plot_figure3(true_traj, latent_traj, phys_traj, fourier_traj, FIG_DIR / "figure3_l22_core.png")
        plot_figure4(phys_traj, fourier_traj, FIG_DIR / "figure4_l22_core.png")

        results = {
            "decoder_history": dec_hist,
            "latent_history": latent_hist,
            "physical_history": phys_hist,
            "fourier_history": fourier_hist,
            "true_traj": true_traj,
            "latent_traj": latent_traj,
            "physical_traj": phys_traj,
            "fourier_traj": fourier_traj,
            "latent_rollout_mse": latent_rollout_mse,
            "physical_rollout_mse": physical_rollout_mse,
            "fourier_rollout_mse": fourier_rollout_mse,
        }
        with (RES_DIR / "figure3_l22_core.pkl").open("wb") as f:
            pickle.dump(results, f)
        save_json(
            RES_DIR / "figure3_l22_core_summary.json",
            {
                "decoder_mse": decoder_mse,
                "latent_rollout_mse": latent_rollout_mse,
                "physical_rollout_mse": physical_rollout_mse,
                "fourier_rollout_mse": fourier_rollout_mse,
                "latent_best_epoch": latent_hist["best_epoch"],
                "physical_best_epoch": phys_hist["best_epoch"],
                "fourier_best_epoch": fourier_hist["best_epoch"],
            },
        )

        tracker.finish(
            decoder_mse=decoder_mse,
            latent_rollout_mse=latent_rollout_mse,
            physical_rollout_mse=physical_rollout_mse,
            fourier_rollout_mse=fourier_rollout_mse,
        )
        print("Saved:")
        print(RES_DIR / "figure3_l22_core_progress.json")
        print(RES_DIR / "figure3_l22_core.pkl")
        print(FIG_DIR / "figure3_l22_core_losses.png")
        print(FIG_DIR / "figure3_l22_core.png")
        print(FIG_DIR / "figure4_l22_core.png")
    except Exception as exc:
        tracker.fail(error=str(exc))
        raise


if __name__ == "__main__":
    main()
