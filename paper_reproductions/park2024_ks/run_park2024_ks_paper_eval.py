"""
Paper-faithful Park 2024 KS evaluation runner.

This script differs from [run_park2024_ks.py] by matching the released
stacNODE KS evaluation path more closely:

- Figure 8 uses one-step predictions on the true test inputs
- Lyapunov exponents for the learned models are computed from Jacobians
  evaluated along the true KS trajectory, not along autonomous model rollouts

It writes separate progress, history, figure, and result artifacts so the
older autonomous-rollout experiments remain intact.
"""

from __future__ import annotations

import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Park2024KSConfig
from .lyapunov_map import compute_map_lyapunov_along_trajectory
from .model import flow_map_forward, init_vector_field_mlp
from .modified_ks_fd import ModifiedKSFD
from .run_park2024_ks import (
    RunTracker,
    build_dataset,
    compute_map_jacobians_cached,
    make_loss_figure,
    save_json,
    save_partial_results,
)
from .train import train_flow_map

jax.config.update("jax_enable_x64", True)


def tree_to_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def predict_one_step(model_params: dict, x_batch: np.ndarray, *, dt: float) -> np.ndarray:
    x_batch_j = jnp.array(x_batch, dtype=jnp.float64)
    pred = jax.vmap(lambda x: flow_map_forward(model_params, x, dt))(x_batch_j)
    return np.asarray(pred)


def make_figure8_paper_eval(
    cfg: Park2024KSConfig,
    solver: ModifiedKSFD,
    true_next: np.ndarray,
    mse_next: np.ndarray,
    jac_next: np.ndarray,
    *,
    output_path,
) -> None:
    x_full = np.linspace(0.0, cfg.domain_length, cfg.n_inner + 2)
    n_steps = min(cfg.figure8_steps, len(true_next), len(mse_next), len(jac_next))
    t_vals = np.arange(n_steps) * cfg.dt
    systems = [
        ("True", solver.trajectory_to_full(true_next[:n_steps])),
        ("MSE", solver.trajectory_to_full(mse_next[:n_steps])),
        ("JAC", solver.trajectory_to_full(jac_next[:n_steps])),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (title, traj) in zip(axes, systems):
        vmax = max(np.percentile(np.abs(traj), 98), 1e-3)
        ax.pcolormesh(
            x_full,
            t_vals,
            traj,
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.suptitle("Park 2024 Figure 8 style KS solution plot (paper-eval)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main():
    cfg = Park2024KSConfig()
    cfg.ensure_dirs()

    progress_path = cfg.data_dir / "park2024_ks_paper_eval_progress.json"
    partial_results_path = cfg.data_dir / "park2024_ks_partial_paper_eval.pkl"
    results_path = cfg.data_dir / "park2024_ks_results_paper_eval.pkl"
    mse_history_path = cfg.data_dir / "park2024_ks_mse_history_paper_eval.json"
    jac_history_path = cfg.data_dir / "park2024_ks_jac_history_paper_eval.json"
    loss_fig_path = cfg.fig_dir / "park2024_ks_loss_paper_eval.png"
    figure8_path = cfg.fig_dir / "park2024_figure8_ks_paper_eval.png"

    tracker = RunTracker.create(
        progress_path,
        domain_length=cfg.domain_length,
        n_inner=cfg.n_inner,
        dt=cfg.dt,
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        epochs=cfg.epochs,
        evaluation_mode="paper_faithful",
    )

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
        save_json(cfg.data_dir / "park2024_ks_dataset_meta_paper_eval.json", dataset_meta)
        tracker.update(stage="dataset_generation", **dataset_meta)
        save_partial_results(partial_results_path, {"dataset_meta": dataset_meta})

        print("Loading true map Jacobian caches...")
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
        mse_params0 = init_vector_field_mlp(
            key_mse,
            cfg.state_dim,
            cfg.hidden_widths,
            init_style=cfg.mse_init_style,
        )

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

        print("Training JAC map model...")
        tracker.set_stage("jac_training")
        key_jac = jax.random.PRNGKey(2)
        jac_params0 = init_vector_field_mlp(key_jac, cfg.state_dim, cfg.hidden_widths)

        def jac_progress(snapshot: dict) -> None:
            tracker.update(stage="jac_training", **snapshot)
            save_json(jac_history_path, snapshot | {"history_epochs_recorded": snapshot["epoch"]})

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
            lr_schedule="exp_decay",
            select_best=False,
            seed=2,
            eval_every=cfg.training_eval_every,
            progress_callback=jac_progress,
        )
        save_json(jac_history_path, jac_hist)
        save_partial_results(
            partial_results_path,
            {
                "dataset_meta": dataset_meta,
                "jacobian_shapes": jacobian_shapes,
                "mse_history": mse_hist,
                "jac_history": jac_hist,
            },
        )

        print("Computing one-step Figure 8 predictions...")
        tracker.set_stage("figure8_prediction")
        n_fig = min(cfg.figure8_steps, len(data["x_test"]))
        true_next = np.asarray(data["y_test"][:n_fig])
        mse_next = predict_one_step(mse_params, data["x_test"][:n_fig], dt=cfg.dt)
        jac_next = predict_one_step(jac_params, data["x_test"][:n_fig], dt=cfg.dt)
        tracker.update(
            stage="figure8_prediction",
            true_shape=list(true_next.shape),
            mse_shape=list(mse_next.shape),
            jac_shape=list(jac_next.shape),
        )

        print("Computing Lyapunov spectra along the true trajectory...")
        true_map = lambda x: solver.step(x)
        mse_map = lambda x: flow_map_forward(mse_params, x, cfg.dt)
        jac_map = lambda x: flow_map_forward(jac_params, x, cfg.dt)

        forced_steps = min(6000, len(data["trajectory"]))
        true_traj = np.asarray(data["trajectory"][:forced_steps])

        tracker.set_stage("lyapunov_true")
        le_true = compute_map_lyapunov_along_trajectory(
            true_map,
            true_traj,
            n_steps=forced_steps,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_true", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_true", first_exponent=float(le_true[0]))

        tracker.set_stage("lyapunov_mse")
        le_mse = compute_map_lyapunov_along_trajectory(
            mse_map,
            true_traj,
            n_steps=forced_steps,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_mse", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_mse", first_exponent=float(le_mse[0]))

        tracker.set_stage("lyapunov_jac")
        le_jac = compute_map_lyapunov_along_trajectory(
            jac_map,
            true_traj,
            n_steps=forced_steps,
            n_lyap=cfg.n_lyap,
            time_per_step=cfg.dt,
            progress_callback=lambda snap: tracker.update(stage="lyapunov_jac", **snap),
            progress_block_size=cfg.lyapunov_progress_block_size,
        )
        tracker.update(stage="lyapunov_jac", first_exponent=float(le_jac[0]))

        tracker.set_stage("figure_generation")
        make_loss_figure(cfg, mse_hist, jac_hist, output_path=loss_fig_path)
        make_figure8_paper_eval(cfg, solver, true_next, mse_next, jac_next, output_path=figure8_path)

        results = {
            "config": cfg,
            "dataset_meta": dataset_meta,
            "mse_history": mse_hist,
            "jac_history": jac_hist,
            "mse_params": tree_to_numpy(mse_params),
            "jac_params": tree_to_numpy(jac_params),
            "lyapunov_table5_ks": {
                "true": le_true,
                "mse": le_mse,
                "jac": le_jac,
            },
            "figure8_one_step": {
                "true": true_next,
                "mse": mse_next,
                "jac": jac_next,
            },
            "forced_trajectory": true_traj,
            "jacobian_shapes": jacobian_shapes,
        }
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        save_partial_results(partial_results_path, results)

        tracker.finish(
            best_mse_epoch=mse_hist["best_epoch"],
            selected_mse_epoch=mse_hist["selected_epoch"],
            best_jac_epoch=jac_hist["best_epoch"],
            selected_jac_epoch=jac_hist["selected_epoch"],
            true_lambda1=float(le_true[0]),
            mse_lambda1=float(le_mse[0]),
            jac_lambda1=float(le_jac[0]),
        )

        print("\nSaved:")
        print(f"  {results_path}")
        print(f"  {loss_fig_path}")
        print(f"  {figure8_path}")
        print(f"  {progress_path}")
        print("\nTable 5 KS row (paper-eval):")
        print("  True:", np.array2string(le_true, precision=4, separator=', '))
        print("  MSE :", np.array2string(le_mse, precision=4, separator=', '))
        print("  JAC :", np.array2string(le_jac, precision=4, separator=', '))
    except Exception as exc:
        tracker.fail(error=str(exc))
        raise


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {time.time() - t0:.1f}s")
