"""
Run the Park 2024 KS reproduction:

- generate modified KS data in 127D physical space
- train one-step map models with MSE and JAC
- create a Figure 8 style solution plot
- compute the Table 5 KS-row Lyapunov comparison

This script now follows the downloaded stacNODE KS setup much more closely,
while keeping a local JAX implementation for reproducibility inside this repo.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Park2024KSConfig
from .modified_ks_fd import ModifiedKSFD
from .model import init_vector_field_mlp, flow_map_forward
from .train import train_flow_map
from .lyapunov_map import compute_map_lyapunov

jax.config.update("jax_enable_x64", True)


@dataclass
class RunTracker:
    path: object
    state: dict

    @classmethod
    def create(cls, path, **meta):
        tracker = cls(path=path, state={"meta": meta, "status": "running", "updates": []})
        tracker.flush()
        return tracker

    def set_stage(self, stage: str, **metrics) -> None:
        self.state["current_stage"] = stage
        self.update(stage=stage, **metrics)

    def update(self, stage: str | None = None, **metrics) -> None:
        entry = {
            "time": time.time(),
            "stage": stage or self.state.get("current_stage"),
            "metrics": metrics,
        }
        self.state["latest"] = entry
        self.state["updates"].append(entry)
        self.flush()

    def finish(self, **metrics) -> None:
        self.state["status"] = "completed"
        self.state["finished_at"] = time.time()
        self.state["final"] = metrics
        self.flush()

    def fail(self, **metrics) -> None:
        self.state["status"] = "failed"
        self.state["finished_at"] = time.time()
        self.state["final"] = metrics
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)


def save_json(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_partial_results(path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def build_dataset(solver: ModifiedKSFD, cfg: Park2024KSConfig):
    if cfg.use_repo_initial_condition:
        u0_full = solver.repo_initial_condition()
    else:
        key = jax.random.PRNGKey(0)
        u0_full = solver.random_initial_condition(key)

    total_steps = solver.time_to_steps(cfg.total_simulation_time)
    traj_full = np.array(solver.integrate_full(u0_full, total_steps))
    traj = traj_full[:, 1:-1]

    required_steps = cfg.train_size + cfg.test_size + 1
    if traj.shape[0] < required_steps:
        raise ValueError(
            f"Trajectory has {traj.shape[0]} interior snapshots, need at least {required_steps}. "
            "Increase total_simulation_time."
        )

    x_train = traj[:cfg.train_size]
    y_train = traj[1:cfg.train_size + 1]
    x_test = traj[cfg.train_size:cfg.train_size + cfg.test_size]
    y_test = traj[cfg.train_size + 1:cfg.train_size + cfg.test_size + 1]
    return {
        "u0_full": np.array(u0_full),
        "u0": np.array(traj[0]),
        "trajectory_full": traj_full,
        "trajectory": traj,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
    }


def compute_map_jacobians_cached(
    solver: ModifiedKSFD,
    x_data: np.ndarray,
    *,
    cache_path,
    meta_path,
    tracker: RunTracker,
    stage_name: str,
    progress_every: int = 25,
) -> np.ndarray:
    n_samples = int(len(x_data))
    state_dim = int(x_data.shape[1])
    expected_shape = [n_samples, state_dim, state_dim]

    if cache_path.exists() and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("shape") != expected_shape:
            cache_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            meta = {
                "status": "running",
                "shape": expected_shape,
                "completed": 0,
                "updated_at": time.time(),
            }
            cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float64, shape=tuple(expected_shape))
            save_json(meta_path, meta)
            completed = 0
        elif meta.get("status") == "completed":
            tracker.update(stage=stage_name, resumed=True, completed=n_samples, total=n_samples, fraction=1.0)
            return np.load(cache_path, mmap_mode="r")
        else:
            completed = int(meta.get("completed", 0))
            cache = np.lib.format.open_memmap(cache_path, mode="r+", dtype=np.float64, shape=tuple(expected_shape))
    else:
        completed = 0
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float64, shape=tuple(expected_shape))
        meta = {
            "status": "running",
            "shape": expected_shape,
            "completed": 0,
            "updated_at": time.time(),
        }
        save_json(meta_path, meta)

    jac_fn = jax.jit(solver.map_jacobian)
    progress_every = max(int(progress_every), 1)
    tracker.update(stage=stage_name, resumed=completed > 0, completed=completed, total=n_samples, fraction=completed / max(n_samples, 1))

    for idx in range(completed, n_samples):
        cache[idx] = np.array(jac_fn(jnp.array(x_data[idx], dtype=jnp.float64)))
        done = idx + 1
        if (done % progress_every == 0) or (done == n_samples):
            cache.flush()
            meta["completed"] = done
            meta["updated_at"] = time.time()
            meta["status"] = "running" if done < n_samples else "completed"
            save_json(meta_path, meta)
            tracker.update(stage=stage_name, completed=done, total=n_samples, fraction=done / max(n_samples, 1))

    return np.load(cache_path, mmap_mode="r")


def rollout_map(model_params: dict, u0: np.ndarray, n_steps: int, *, dt: float) -> np.ndarray:
    step = jax.jit(lambda x: flow_map_forward(model_params, x, dt))

    def scan_fn(x, _):
        x_next = step(x)
        return x_next, x_next

    _, traj = jax.lax.scan(scan_fn, jnp.array(u0, dtype=jnp.float64), None, length=n_steps)
    return np.array(traj)


def make_loss_figure(cfg: Park2024KSConfig, mse_hist: dict, jac_hist: dict, *, output_path=None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(mse_hist["epochs"], mse_hist["train_loss"], label="Train")
    axes[0].plot(mse_hist["epochs"], mse_hist["test_loss"], label="Test")
    axes[0].set_title("MSE")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(jac_hist["epochs"], jac_hist["train_loss"], label="Train")
    axes[1].plot(jac_hist["epochs"], jac_hist["test_loss"], label="Test")
    axes[1].set_title("JAC")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("Park 2024 style training/test loss curves")
    plt.tight_layout()
    plt.savefig(output_path or (cfg.fig_dir / "park2024_ks_loss.png"), dpi=140)
    plt.close()


def make_figure8(
    cfg: Park2024KSConfig,
    solver: ModifiedKSFD,
    true_traj: np.ndarray,
    mse_traj: np.ndarray,
    jac_traj: np.ndarray,
    *,
    output_path=None,
) -> None:
    x_full = np.linspace(0.0, cfg.domain_length, cfg.n_inner + 2)
    t_vals = np.arange(cfg.figure8_steps) * cfg.dt
    systems = [
        ("True", solver.trajectory_to_full(true_traj[:cfg.figure8_steps])),
        ("MSE", solver.trajectory_to_full(mse_traj[:cfg.figure8_steps])),
        ("JAC", solver.trajectory_to_full(jac_traj[:cfg.figure8_steps])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (title, traj) in zip(axes, systems):
        vmax = max(np.percentile(np.abs(traj), 98), 1e-3)
        ax.pcolormesh(x_full, t_vals, traj, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.suptitle("Park 2024 Figure 8 style KS solution plot")
    plt.tight_layout()
    plt.savefig(output_path or (cfg.fig_dir / "park2024_figure8_ks.png"), dpi=140)
    plt.close()


def main(mode: str = "full"):
    cfg = Park2024KSConfig()
    cfg.ensure_dirs()
    progress_path = cfg.data_dir / ("park2024_ks_progress.json" if mode == "full" else "park2024_ks_rerun_mse_progress.json")
    tracker = RunTracker.create(
        progress_path,
        domain_length=cfg.domain_length,
        n_inner=cfg.n_inner,
        dt=cfg.dt,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        epochs=cfg.epochs,
        mode=mode,
    )
    partial_results_path = cfg.data_dir / ("park2024_ks_partial.pkl" if mode == "full" else "park2024_ks_partial_rerun_mse.pkl")
    solver = ModifiedKSFD(
        n_inner=cfg.n_inner,
        domain_length=cfg.domain_length,
        c_param=cfg.c_param,
        dt=cfg.dt,
    )

    try:
        print("Generating Park-style KS dataset...")
        tracker.set_stage("dataset_generation")
        data = build_dataset(solver, cfg)
        dataset_meta = {
            "x_train_shape": list(data["x_train"].shape),
            "x_test_shape": list(data["x_test"].shape),
            "trajectory_shape": list(data["trajectory"].shape),
        }
        save_json(cfg.data_dir / "park2024_ks_dataset_meta.json", dataset_meta)
        tracker.update(stage="dataset_generation", **dataset_meta)
        save_partial_results(partial_results_path, {"dataset_meta": dataset_meta})

        if mode == "rerun_mse":
            print("Loading cached Park results for MSE-only rerun...")
            base_results_path = cfg.data_dir / "park2024_ks_results.pkl"
            if not base_results_path.exists():
                raise FileNotFoundError(f"Missing base results file: {base_results_path}")
            with open(base_results_path, "rb") as f:
                base_results = pickle.load(f)
            tracker.update(
                stage="dataset_generation",
                reused_results_file=str(base_results_path),
                reused_fields=["jac_history", "lyapunov_table5_ks.true", "lyapunov_table5_ks.jac", "figure8_rollouts.true", "figure8_rollouts.jac"],
            )

            print("Training MSE map model (rerun_mse mode)...")
            tracker.set_stage("mse_training")
            key_mse = jax.random.PRNGKey(1)
            mse_params0 = init_vector_field_mlp(
                key_mse,
                cfg.state_dim,
                cfg.hidden_widths,
                init_style=cfg.mse_init_style,
            )

            mse_history_path = cfg.data_dir / "park2024_ks_mse_history_rerun.json"

            def mse_progress(snapshot: dict) -> None:
                tracker.update(stage="mse_training", **snapshot)
                save_json(mse_history_path, snapshot | {"history_epochs_recorded": snapshot["epoch"]})

            mse_params, mse_hist = train_flow_map(
                mse_params0,
                data["x_train"],
                data["y_train"],
                x_test=data["x_test"],
                y_test=data["y_test"],
                loss_mode="mse",
                dt=cfg.dt,
                epochs=cfg.epochs,
                batch_size=cfg.mse_batch_size,
                learning_rate=cfg.mse_learning_rate,
                weight_decay=cfg.mse_weight_decay,
                lr_schedule=cfg.mse_lr_schedule,
                select_best=cfg.mse_select_best,
                seed=1,
                eval_every=cfg.training_eval_every,
                progress_callback=mse_progress,
            )
            save_json(mse_history_path, mse_hist)
            save_partial_results(
                partial_results_path,
                {
                    "dataset_meta": dataset_meta,
                    "mse_history": mse_hist,
                    "reused_results_file": str(base_results_path),
                },
            )

            print("Rolling out Figure 8 MSE trajectory...")
            tracker.set_stage("figure8_rollout")
            u0_fig = data["x_test"][0]
            mse_traj = rollout_map(mse_params, u0_fig, cfg.figure8_steps, dt=cfg.dt)
            true_traj = np.asarray(base_results["figure8_rollouts"]["true"])
            jac_traj = np.asarray(base_results["figure8_rollouts"]["jac"])
            tracker.update(
                stage="figure8_rollout",
                true_shape=list(true_traj.shape),
                mse_shape=list(mse_traj.shape),
                jac_shape=list(jac_traj.shape),
                reused_true_and_jac=True,
            )

            print("Computing MSE Lyapunov spectrum...")
            mse_map = lambda x: flow_map_forward(mse_params, x, cfg.dt)
            x0_lyap = data["x_test"][100]
            tracker.set_stage("lyapunov_mse")
            le_mse = compute_map_lyapunov(
                mse_map,
                x0_lyap,
                n_steps=cfg.lyap_steps,
                n_warmup=cfg.lyap_warmup,
                n_lyap=cfg.n_lyap,
                time_per_step=cfg.dt,
                progress_callback=lambda snap: tracker.update(stage="lyapunov_mse", **snap),
                progress_block_size=cfg.lyapunov_progress_block_size,
            )
            tracker.update(stage="lyapunov_mse", first_exponent=float(le_mse[0]))

            tracker.set_stage("figure_generation")
            rerun_loss_path = cfg.fig_dir / "park2024_ks_loss_rerun_mse.png"
            rerun_fig8_path = cfg.fig_dir / "park2024_figure8_ks_rerun_mse.png"
            make_loss_figure(cfg, mse_hist, base_results["jac_history"], output_path=rerun_loss_path)
            make_figure8(cfg, solver, true_traj, mse_traj, jac_traj, output_path=rerun_fig8_path)

            rerun_results = {
                "config": cfg,
                "base_results_file": str(base_results_path),
                "mse_history": mse_hist,
                "jac_history": base_results["jac_history"],
                "lyapunov_table5_ks": {
                    "true": np.asarray(base_results["lyapunov_table5_ks"]["true"]),
                    "mse": le_mse,
                    "jac": np.asarray(base_results["lyapunov_table5_ks"]["jac"]),
                },
                "figure8_rollouts": {
                    "true": true_traj,
                    "mse": mse_traj,
                    "jac": jac_traj,
                },
                "reused_fields": [
                    "jac_history",
                    "lyapunov_table5_ks.true",
                    "lyapunov_table5_ks.jac",
                    "figure8_rollouts.true",
                    "figure8_rollouts.jac",
                ],
                "recomputed_fields": [
                    "mse_history",
                    "lyapunov_table5_ks.mse",
                    "figure8_rollouts.mse",
                ],
            }
            rerun_results_path = cfg.data_dir / "park2024_ks_results_rerun_mse.pkl"
            with open(rerun_results_path, "wb") as f:
                pickle.dump(rerun_results, f)
            save_partial_results(partial_results_path, rerun_results)
            save_json(
                cfg.data_dir / "park2024_ks_rerun_meta.json",
                {
                    "mode": mode,
                    "base_results_file": str(base_results_path),
                    "results_file": str(rerun_results_path),
                    "progress_file": str(progress_path),
                    "reused_fields": rerun_results["reused_fields"],
                    "recomputed_fields": rerun_results["recomputed_fields"],
                },
            )

            tracker.finish(
                best_mse_epoch=mse_hist["best_epoch"],
                selected_mse_epoch=mse_hist["selected_epoch"],
                true_lambda1=float(rerun_results["lyapunov_table5_ks"]["true"][0]),
                mse_lambda1=float(le_mse[0]),
                jac_lambda1=float(rerun_results["lyapunov_table5_ks"]["jac"][0]),
            )

            print("\nSaved:")
            print(f"  {rerun_results_path}")
            print(f"  {rerun_loss_path}")
            print(f"  {rerun_fig8_path}")
            print(f"  {progress_path}")
            print("\nTable 5 KS row (mixed rerun):")
            print("  True:", np.array2string(rerun_results["lyapunov_table5_ks"]["true"], precision=4, separator=", "))
            print("  MSE :", np.array2string(le_mse, precision=4, separator=", "))
            print("  JAC :", np.array2string(rerun_results["lyapunov_table5_ks"]["jac"], precision=4, separator=", "))
            return

        print("Computing true map Jacobians for train/test data...")
        tracker.set_stage("jacobian_precompute_train")
        jac_train = compute_map_jacobians_cached(
            solver,
            data["x_train"],
            cache_path=cfg.data_dir / "jac_train_cache.npy",
            meta_path=cfg.data_dir / "jac_train_cache_meta.json",
            tracker=tracker,
            stage_name="jacobian_precompute_train",
            progress_every=cfg.jacobian_progress_every,
        )
        tracker.set_stage("jacobian_precompute_test")
        jac_test = compute_map_jacobians_cached(
            solver,
            data["x_test"],
            cache_path=cfg.data_dir / "jac_test_cache.npy",
            meta_path=cfg.data_dir / "jac_test_cache_meta.json",
            tracker=tracker,
            stage_name="jacobian_precompute_test",
            progress_every=cfg.jacobian_progress_every,
        )
        jacobian_shapes = {"train": list(jac_train.shape), "test": list(jac_test.shape)}
        save_partial_results(partial_results_path, {"dataset_meta": dataset_meta, "jacobian_shapes": jacobian_shapes})

        print("Training MSE map model...")
        tracker.set_stage("mse_training")
        key_mse = jax.random.PRNGKey(1)
        mse_params0 = init_vector_field_mlp(key_mse, cfg.state_dim, cfg.hidden_widths)

        def mse_progress(snapshot: dict) -> None:
            tracker.update(stage="mse_training", **snapshot)
            save_json(cfg.data_dir / "park2024_ks_mse_history.json", snapshot | {"history_epochs_recorded": snapshot["epoch"]})

        mse_params, mse_hist = train_flow_map(
            mse_params0,
            data["x_train"],
            data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            loss_mode="mse",
            dt=cfg.dt,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            seed=1,
            eval_every=cfg.training_eval_every,
            progress_callback=mse_progress,
        )
        save_json(cfg.data_dir / "park2024_ks_mse_history.json", mse_hist)
        save_partial_results(
            partial_results_path,
            {
                "dataset_meta": dataset_meta,
                "jacobian_shapes": jacobian_shapes,
                "mse_history": mse_hist,
            },
        )

        print("Training JAC map model...")
        tracker.set_stage("jac_training")
        key_jac = jax.random.PRNGKey(2)
        jac_params0 = init_vector_field_mlp(key_jac, cfg.state_dim, cfg.hidden_widths)

        def jac_progress(snapshot: dict) -> None:
            tracker.update(stage="jac_training", **snapshot)
            save_json(cfg.data_dir / "park2024_ks_jac_history.json", snapshot | {"history_epochs_recorded": snapshot["epoch"]})

        jac_params, jac_hist = train_flow_map(
            jac_params0,
            data["x_train"],
            data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            jac_train=jac_train,
            jac_test=jac_test,
            loss_mode="jac",
            lam=cfg.jac_lambda,
            dt=cfg.dt,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            seed=2,
            eval_every=cfg.training_eval_every,
            progress_callback=jac_progress,
        )
        save_json(cfg.data_dir / "park2024_ks_jac_history.json", jac_hist)
        save_partial_results(
            partial_results_path,
            {
                "dataset_meta": dataset_meta,
                "jacobian_shapes": jacobian_shapes,
                "mse_history": mse_hist,
                "jac_history": jac_hist,
            },
        )

        print("Rolling out Figure 8 trajectories...")
        tracker.set_stage("figure8_rollout")
        u0_fig = data["x_test"][0]
        true_traj = np.array(solver.integrate(jnp.array(u0_fig, dtype=jnp.float64), cfg.figure8_steps, state_kind="interior"))
        mse_traj = rollout_map(mse_params, u0_fig, cfg.figure8_steps, dt=cfg.dt)
        jac_traj = rollout_map(jac_params, u0_fig, cfg.figure8_steps, dt=cfg.dt)
        tracker.update(
            stage="figure8_rollout",
            true_shape=list(true_traj.shape),
            mse_shape=list(mse_traj.shape),
            jac_shape=list(jac_traj.shape),
        )

        print("Computing Lyapunov spectra for Table 5 KS row...")
        true_map = lambda x: solver.step(x)
        mse_map = lambda x: flow_map_forward(mse_params, x, cfg.dt)
        jac_map = lambda x: flow_map_forward(jac_params, x, cfg.dt)
        x0_lyap = data["x_test"][100]

        tracker.set_stage("lyapunov_true")
        le_true = compute_map_lyapunov(
            true_map,
            x0_lyap,
            n_steps=cfg.lyap_steps,
            n_warmup=cfg.lyap_warmup,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_true", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_true", first_exponent=float(le_true[0]))

        tracker.set_stage("lyapunov_mse")
        le_mse = compute_map_lyapunov(
            mse_map,
            x0_lyap,
            n_steps=cfg.lyap_steps,
            n_warmup=cfg.lyap_warmup,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_mse", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_mse", first_exponent=float(le_mse[0]))

        tracker.set_stage("lyapunov_jac")
        le_jac = compute_map_lyapunov(
            jac_map,
            x0_lyap,
            n_steps=cfg.lyap_steps,
            n_warmup=cfg.lyap_warmup,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_jac", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_jac", first_exponent=float(le_jac[0]))

        tracker.set_stage("figure_generation")
        make_loss_figure(cfg, mse_hist, jac_hist)
        make_figure8(cfg, solver, true_traj, mse_traj, jac_traj)

        results = {
            "config": cfg,
            "mse_history": mse_hist,
            "jac_history": jac_hist,
            "lyapunov_table5_ks": {
                "true": le_true,
                "mse": le_mse,
                "jac": le_jac,
            },
            "figure8_rollouts": {
                "true": true_traj,
                "mse": mse_traj,
                "jac": jac_traj,
            },
            "jacobian_shapes": jacobian_shapes,
        }
        with open(cfg.data_dir / "park2024_ks_results.pkl", "wb") as f:
            pickle.dump(results, f)
        save_partial_results(partial_results_path, results)

        tracker.finish(
            best_mse_epoch=mse_hist["best_epoch"],
            best_jac_epoch=jac_hist["best_epoch"],
            true_lambda1=float(le_true[0]),
            mse_lambda1=float(le_mse[0]),
            jac_lambda1=float(le_jac[0]),
        )

        print("\nSaved:")
        print(f"  {cfg.data_dir / 'park2024_ks_results.pkl'}")
        print(f"  {cfg.fig_dir / 'park2024_ks_loss.png'}")
        print(f"  {cfg.fig_dir / 'park2024_figure8_ks.png'}")
        print(f"  {cfg.data_dir / 'park2024_ks_progress.json'}")
        print("\nTable 5 KS row:")
        print("  True:", np.array2string(le_true, precision=4, separator=", "))
        print("  MSE :", np.array2string(le_mse, precision=4, separator=", "))
        print("  JAC :", np.array2string(le_jac, precision=4, separator=", "))
    except Exception as exc:
        tracker.fail(error=str(exc))
        raise


if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "rerun_mse"), default="full")
    args = parser.parse_args()
    main(mode=args.mode)
    print(f"\nDone in {time.time() - t0:.1f}s")
