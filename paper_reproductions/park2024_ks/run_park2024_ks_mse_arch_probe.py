from __future__ import annotations

import json
import time
from pathlib import Path

import jax

from .config import Park2024KSConfig
from .model import init_vector_field_mlp
from .modified_ks_fd import ModifiedKSFD
from .run_park2024_ks import build_dataset
from .train import train_flow_map

jax.config.update("jax_enable_x64", True)


ROOT = Path("paper_reproductions/park2024_ks")
DATA_DIR = ROOT / "data"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_probe() -> dict:
    cfg = Park2024KSConfig()
    solver = ModifiedKSFD(
        n_inner=cfg.n_inner,
        domain_length=cfg.domain_length,
        c_param=cfg.c_param,
        dt=cfg.dt,
    )
    data = build_dataset(solver, cfg)

    probe_specs = [
        {
            "name": "plain_512_256",
            "architecture": "plain_mlp",
            "hidden_widths": (512, 256),
            "init_style": "pytorch_linear",
        },
        {
            "name": "plain_256_128",
            "architecture": "plain_mlp",
            "hidden_widths": (256, 128),
            "init_style": "pytorch_linear",
        },
        {
            "name": "linear_plus_256_128",
            "architecture": "linear_plus_mlp",
            "hidden_widths": (256, 128),
            "init_style": "pytorch_linear",
        },
        {
            "name": "linear_plus_512_256",
            "architecture": "linear_plus_mlp",
            "hidden_widths": (512, 256),
            "init_style": "pytorch_linear",
        },
    ]

    progress_path = DATA_DIR / "park2024_ks_mse_arch_probe_progress.json"
    results_path = DATA_DIR / "park2024_ks_mse_arch_probe_results.json"

    progress = {
        "status": "running",
        "started_at": time.time(),
        "epochs_per_run": 300,
        "batch_size": cfg.train_size,
        "learning_rate": 1.0e-4,
        "weight_decay": 5.0e-4,
        "runs": [],
    }
    save_json(progress_path, progress)

    summaries = []
    for idx, spec in enumerate(probe_specs, start=1):
        run_name = spec["name"]
        key = jax.random.PRNGKey(100 + idx)
        init_params = init_vector_field_mlp(
            key,
            cfg.state_dim,
            spec["hidden_widths"],
            init_style=spec["init_style"],
            architecture=spec["architecture"],
        )

        history_path = DATA_DIR / f"park2024_ks_mse_arch_probe_{run_name}.json"
        partial = {"epoch": [], "train_loss": [], "test_loss": [], "test_relative_error": []}

        def callback(snapshot: dict) -> None:
            partial["epoch"].append(snapshot["epoch"])
            partial["train_loss"].append(snapshot["train_loss"])
            partial["test_loss"].append(snapshot["test_loss"])
            partial["test_relative_error"].append(snapshot["test_relative_error"])
            partial["best_epoch"] = snapshot["best_epoch"]
            partial["best_test_loss"] = snapshot["best_test_loss"]
            partial["selected_epoch"] = snapshot["selected_epoch"]
            save_json(history_path, partial)

        _, hist = train_flow_map(
            init_params,
            data["x_train"],
            data["y_train"],
            x_test=data["x_test"],
            y_test=data["y_test"],
            loss_mode="mse",
            dt=cfg.dt,
            epochs=300,
            batch_size=cfg.train_size,
            learning_rate=1.0e-4,
            weight_decay=5.0e-4,
            lr_schedule="constant",
            select_best=False,
            grad_clip=None,
            seed=100 + idx,
            eval_every=25,
            progress_callback=callback,
        )
        save_json(history_path, hist)

        best_rel = float(min(hist["test_relative_error"]))
        best_rel_epoch = int(hist["epochs"][hist["test_relative_error"].index(best_rel)])
        summary = {
            **spec,
            "final_train_loss": float(hist["train_loss"][-1]),
            "final_test_loss": float(hist["test_loss"][-1]),
            "final_test_relative_error": float(hist["test_relative_error"][-1]),
            "best_test_loss": float(hist["best_test_loss"]),
            "best_epoch": int(hist["best_epoch"]),
            "best_test_relative_error": best_rel,
            "best_test_relative_error_epoch": best_rel_epoch,
            "test_loss_improved_after_epoch1": bool(min(hist["test_loss"]) < hist["test_loss"][0] - 1e-12),
            "test_rel_improved_after_epoch1": bool(best_rel < hist["test_relative_error"][0] - 1e-12),
            "history_file": str(history_path),
        }
        summaries.append(summary)
        progress["runs"] = summaries
        progress["latest_run"] = run_name
        save_json(progress_path, progress)

    progress["status"] = "completed"
    progress["finished_at"] = time.time()
    progress["runs"] = summaries
    save_json(progress_path, progress)
    save_json(results_path, {"runs": summaries})
    return {"runs": summaries, "progress_path": str(progress_path), "results_path": str(results_path)}


def main() -> None:
    result = run_probe()
    print(result["progress_path"])
    print(result["results_path"])
    for run in result["runs"]:
        print(
            f"{run['name']}: final_rel={run['final_test_relative_error']:.4f}, "
            f"best_rel={run['best_test_relative_error']:.4f} @ epoch {run['best_test_relative_error_epoch']}"
        )


if __name__ == "__main__":
    main()
