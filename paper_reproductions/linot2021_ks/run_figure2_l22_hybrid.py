from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "traj_train.npy"
TEST_PATH = ROOT / "data" / "traj_analysis.npy"
OUT_DIR = ROOT / "paper_reproductions" / "linot2021_ks"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"


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


def fit_hybrid_curve(
    x_train: np.ndarray,
    x_test: np.ndarray,
    dims: list[int],
    *,
    hidden_layers: tuple[int, ...] = (500,),
    epochs: int = 150,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    seed: int = 0,
    train_cap: int = 5000,
    test_cap: int = 2000,
    initial_curve: dict[str, list[float]] | None = None,
    progress_callback=None,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)

    if train_cap and len(x_train) > train_cap:
        x_train = x_train[rng.choice(len(x_train), size=train_cap, replace=False)]
    if test_cap and len(x_test) > test_cap:
        x_test = x_test[rng.choice(len(x_test), size=test_cap, replace=False)]

    mean = x_train.mean(axis=0)
    x_train_c = x_train - mean
    x_test_c = x_test - mean

    _, _, vt = np.linalg.svd(x_train_c, full_matrices=False)
    coeff_train = x_train_c @ vt.T
    coeff_test = x_test_c @ vt.T

    existing: dict[int, tuple[float, float | None]] = {}
    if initial_curve is not None:
        existing_dims = initial_curve.get("dims", [])
        existing_mse = initial_curve.get("mse", [])
        existing_residual = initial_curve.get("residual_loss", [])
        for i, dim in enumerate(existing_dims):
            residual = existing_residual[i] if i < len(existing_residual) else None
            existing[int(dim)] = (float(existing_mse[i]), None if residual is None else float(residual))

    mse_values: list[float] = []
    final_losses: list[float] = []

    for dim in dims:
        if dim in existing:
            mse, residual = existing[dim]
            mse_values.append(mse)
            final_losses.append(float("nan") if residual is None else residual)
            if progress_callback is not None:
                progress_callback(
                    {
                        "current_dim": int(dim),
                        "curve": {
                            "dims": list(dims[: len(mse_values)]),
                            "mse": list(mse_values),
                            "residual_loss": list(final_losses),
                        },
                    "resumed": True,
                }
            )
            continue

        if progress_callback is not None:
            progress_callback(
                {
                    "active_dim": int(dim),
                    "curve": {
                        "dims": list(dims[: len(mse_values)]),
                        "mse": list(mse_values),
                        "residual_loss": list(final_losses),
                    },
                    "phase": "starting_dim",
                    "resumed": False,
                }
            )

        x_lat_train = coeff_train[:, :dim]
        y_res_train = coeff_train[:, dim:]
        x_lat_test = coeff_test[:, :dim]
        y_res_test = coeff_test[:, dim:]

        key = jax.random.PRNGKey(seed + dim)
        params = init_mlp(key, [dim, *hidden_layers, 64 - dim], scale=0.1)
        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        x_train_j = jnp.array(x_lat_train)
        y_train_j = jnp.array(y_res_train)
        x_test_j = jnp.array(x_lat_test)
        y_test_j = jnp.array(y_res_test)

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

        n_train = len(x_train_j)
        for _ in range(epochs):
            order = rng.permutation(n_train)
            for start in range(0, n_train - batch_size + 1, batch_size):
                idx = order[start:start + batch_size]
                xb = x_train_j[idx]
                yb = y_train_j[idx]
                params, opt_state, _ = train_step(params, opt_state, xb, yb)

        test_res_pred = np.array(jax.vmap(lambda x: mlp_forward(params, x))(x_test_j))
        coeff_rec = np.concatenate([x_lat_test, test_res_pred], axis=1)
        x_rec = coeff_rec @ vt + mean
        mse = float(np.mean((x_rec - x_test) ** 2))
        final_loss = float(loss_fn(params, x_test_j, y_test_j))
        mse_values.append(mse)
        final_losses.append(final_loss)
        if progress_callback is not None:
            progress_callback(
                {
                    "current_dim": int(dim),
                    "curve": {
                        "dims": list(dims[: len(mse_values)]),
                        "mse": list(mse_values),
                        "residual_loss": list(final_losses),
                    },
                    "active_dim": int(dim),
                    "phase": "completed_dim",
                    "resumed": False,
                }
            )

    return {"dims": dims, "mse": mse_values, "residual_loss": final_losses}


def save_plot(curve: dict[str, list[float]], path: Path) -> None:
    dims = curve["dims"]
    mse = curve["mse"]
    plt.figure(figsize=(7.5, 4.8))
    plt.semilogy(dims, mse, "o-", lw=2, ms=6, color="C1")
    plt.xlabel("Latent dimension d")
    plt.ylabel("Test reconstruction MSE")
    plt.title("Figure 2 hybrid baseline for L = 22")
    plt.grid(True, alpha=0.3)
    plt.xticks(dims)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    x_train = np.load(TRAIN_PATH)
    x_test = np.load(TEST_PATH)
    dims = list(range(2, 18, 2))

    curve = fit_hybrid_curve(x_train, x_test, dims)
    save_plot(curve, FIG_DIR / "figure2_l22_hybrid_d.png")

    summary = {
        "domain_length": 22.0,
        "state_dim": int(x_train.shape[1]),
        "curve": curve,
        "note": "Hybrid PCA+decoder baseline using d on the x-axis.",
    }
    with (RES_DIR / "figure2_l22_hybrid_d.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(FIG_DIR / "figure2_l22_hybrid_d.png")
    print(RES_DIR / "figure2_l22_hybrid_d.json")


if __name__ == "__main__":
    main()
