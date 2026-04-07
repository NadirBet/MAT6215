from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_reproductions.linot2021_ks.data_utils import OUT_DIR, RES_DIR, ensure_dataset, write_json
from paper_reproductions.linot2021_ks.run_figure2_l22_hybrid import fit_hybrid_curve


FIG_DIR = OUT_DIR / "figures"
DM_ESTIMATES = {
    22: 8,
    44: 18,
    66: 28,
}
DOMAIN_CONFIGS = {
    22: {
        "dims": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32],
        "hidden_layers": (500,),
        "epochs": 250,
        "train_cap": 8000,
        "test_cap": 4000,
    },
    44: {
        "dims": [4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40],
        "hidden_layers": (500,),
        "epochs": 700,
        "train_cap": 12000,
        "test_cap": 5000,
    },
    66: {
        "dims": [8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
        "hidden_layers": (500, 500),
        "epochs": 1000,
        "train_cap": 16000,
        "test_cap": 6000,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Linot Figure 2 as a combined hybrid PCA+decoder plot for L=22,44,66."
    )
    parser.add_argument("--domains", nargs="+", type=float, default=[22.0, 44.0, 66.0])
    parser.add_argument("--dims", nargs="+", type=int, default=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32])
    parser.add_argument("--hidden-width", type=int, default=500)
    parser.add_argument("--paper-like", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-cap", type=int, default=5000)
    parser.add_argument("--test-cap", type=int, default=2000)
    parser.add_argument("--allow-generate", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def curve_cache_path(domain_key: int) -> Path:
    return RES_DIR / f"figure2_curve_l{domain_key}.json"


def save_plot(curves: dict[str, dict], path: Path) -> None:
    color_map = {"22": "C0", "44": "C1", "66": "C2"}
    plt.figure(figsize=(8.2, 5.0))
    for label, curve in sorted(curves.items(), key=lambda kv: int(kv[0].split("=")[-1].strip())):
        domain_tag = label.split("=")[-1].strip()
        plt.semilogy(
            curve["dims"],
            curve["mse"],
            "o-",
            lw=2,
            ms=5,
            label=label,
            color=color_map.get(domain_tag, None),
        )
    plt.xlabel("Latent dimension d")
    plt.ylabel("Test reconstruction MSE")
    plt.title("Figure 2 hybrid reconstruction curves for KS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_shifted_plot(curves: dict[str, dict], path: Path) -> None:
    color_map = {"22": "C0", "44": "C1", "66": "C2"}
    plt.figure(figsize=(8.2, 5.0))
    for label, curve in sorted(curves.items(), key=lambda kv: int(kv[0].split("=")[-1].strip())):
        domain_tag = int(label.split("=")[-1].strip())
        d_m = DM_ESTIMATES[domain_tag]
        shifted_dims = [d - d_m for d in curve["dims"]]
        plt.semilogy(
            shifted_dims,
            curve["mse"],
            "o-",
            lw=2,
            ms=5,
            label=f"{label} (d_M={d_m})",
            color=color_map.get(str(domain_tag), None),
        )
    plt.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.xlabel("Latent dimension shift d - d_M")
    plt.ylabel("Test reconstruction MSE")
    plt.title("Figure 2 hybrid reconstruction curves centered by d_M")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    progress_path = RES_DIR / "figure2_all_hybrid_progress.json"
    summary_path = RES_DIR / "figure2_all_hybrid_d.json"

    existing_summary = load_json(summary_path) or {}
    curves: dict[str, dict] = existing_summary.get("curves", {})
    domain_meta: dict[str, dict] = existing_summary.get("domains", {})

    payload = {
        "status": "running",
        "time": time.time(),
        "domains": args.domains,
        "completed_domains": [label for label in curves.keys()],
        "current_domain": None,
        "domain_progress": {},
    }
    write_json(progress_path, payload)

    for offset, L in enumerate(args.domains):
        domain_key = int(round(L))
        label = f"L = {domain_key}"
        x_train, x_test, meta = ensure_dataset(
            L,
            allow_generate=args.allow_generate,
            progress_path=RES_DIR / f"figure2_l{int(round(L))}_dataset_progress.json",
        )
        domain_cfg = DOMAIN_CONFIGS.get(domain_key, {})
        dims_source = domain_cfg.get("dims", args.dims) if args.paper_like else args.dims
        valid_dims = [d for d in dims_source if 1 <= d < x_train.shape[1]]
        hidden_layers = domain_cfg.get("hidden_layers", (args.hidden_width,)) if args.paper_like else (args.hidden_width,)
        epochs = domain_cfg.get("epochs", args.epochs) if args.paper_like else args.epochs
        train_cap = domain_cfg.get("train_cap", args.train_cap) if args.paper_like else args.train_cap
        test_cap = domain_cfg.get("test_cap", args.test_cap) if args.paper_like else args.test_cap
        partial_curve = load_json(curve_cache_path(domain_key))

        payload["current_domain"] = label
        payload["last_config"] = {
            "label": label,
            "hidden_layers": list(hidden_layers),
            "epochs": epochs,
            "train_cap": train_cap,
            "test_cap": test_cap,
        }
        payload["domain_progress"][label] = {
            "dims_target": valid_dims,
            "dims_done": [] if partial_curve is None else partial_curve.get("dims", []),
        }
        write_json(progress_path, payload)

        def domain_progress(snapshot: dict) -> None:
            curve = snapshot["curve"]
            write_json(curve_cache_path(domain_key), curve)
            active_dim = snapshot.get("active_dim")
            phase = snapshot.get("phase")
            payload["domain_progress"][label] = {
                "current_dim": snapshot.get("current_dim"),
                "active_dim": active_dim,
                "phase": phase,
                "dims_target": valid_dims,
                "dims_done": curve["dims"],
                "mse": curve["mse"],
                "resumed_last_dim": bool(snapshot.get("resumed", False)),
            }
            payload["time"] = time.time()
            write_json(progress_path, payload)

        curve = fit_hybrid_curve(
            x_train,
            x_test,
            valid_dims,
            hidden_layers=hidden_layers,
            epochs=epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed + offset * 100,
            train_cap=train_cap,
            test_cap=test_cap,
            initial_curve=partial_curve,
            progress_callback=domain_progress,
        )
        write_json(curve_cache_path(domain_key), curve)
        curves[label] = curve
        domain_meta[label] = meta
        if label not in payload["completed_domains"]:
            payload["completed_domains"].append(label)
        payload["last_curve"] = {
            "label": label,
            "dims": curve["dims"],
            "mse": curve["mse"],
        }
        payload["current_domain"] = None
        payload["domain_progress"][label] = {
            "dims_target": valid_dims,
            "dims_done": curve["dims"],
            "mse": curve["mse"],
            "completed": True,
        }
        payload["time"] = time.time()
        write_json(progress_path, payload)

    figure_path = FIG_DIR / "figure2_all_hybrid_d.png"
    save_plot(curves, figure_path)

    shifted_figure_path = FIG_DIR / "figure2_all_hybrid_d_minus_dm.png"
    save_shifted_plot(curves, shifted_figure_path)

    summary = {
        "note": "Combined Figure 2 hybrid PCA+decoder reconstruction curves using d on the x-axis.",
        "paper_like": bool(args.paper_like),
        "manifold_dimension_estimates": {f"L = {k}": v for k, v in DM_ESTIMATES.items()},
        "domains": domain_meta,
        "curves": curves,
        "domain_configs": {f"L = {k}": v for k, v in DOMAIN_CONFIGS.items()},
    }
    write_json(summary_path, summary)
    payload["status"] = "completed"
    payload["figure_path"] = str(figure_path)
    payload["shifted_figure_path"] = str(shifted_figure_path)
    payload["time"] = time.time()
    write_json(progress_path, payload)

    print("Saved:")
    print(figure_path)
    print(shifted_figure_path)
    print(RES_DIR / "figure2_all_hybrid_d.json")
    print(progress_path)


if __name__ == "__main__":
    main()
