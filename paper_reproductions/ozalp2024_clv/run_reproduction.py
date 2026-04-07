from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from paper_reproductions.ozalp2024_clv.angle_diagnostics import (
    compare_angle_distributions,
    plot_angle_histograms,
    plot_lyapunov_spectrum,
    plot_reconstruction_curve,
)
from paper_reproductions.ozalp2024_clv.cae import CAEConfig
from paper_reproductions.ozalp2024_clv.data import DATA_DIR, FIG_DIR, ROOT, ensure_dataset, get_preset, one_step_pairs
from paper_reproductions.ozalp2024_clv.latent_clv import LatentCLVConfig, compute_esn_clv
from paper_reproductions.ozalp2024_clv.project_extension import LinotExtensionConfig, run_linot_extension
from paper_reproductions.ozalp2024_clv.reference_clv import ReferenceCLVConfig, compute_reference_clv
from paper_reproductions.ozalp2024_clv.train_cae import CAETrainingConfig, latent_dimension_sweep, train_cae
from paper_reproductions.ozalp2024_clv.train_esn import ESNSearchConfig, encode_splits, fit_best_esn
from paper_reproductions.ozalp2024_clv.esn import ESNConfig


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def tag_path(path: Path, preset_name: str) -> Path:
    return path.with_name(f"{path.stem}_{preset_name}{path.suffix}")


def latent_sweep_dims(preset_name: str, latent_dim: int) -> list[int]:
    if preset_name == "paper":
        return [6, 8, 12, 16, 20, 24, 28, 32]
    return [4, 6, 8, 10, 12, 16, 20]


def write_results_md(
    *,
    preset_name: str,
    results_path: Path,
    preset: dict,
    cae_result: dict,
    esn_result: dict,
    reference_result: dict,
    latent_result: dict,
    angle_metrics: dict,
    extension_result: dict | None,
) -> None:
    lines = [
        f"# Ozalp 2024 Reproduction ({preset_name})",
        "",
        "## Preset",
        "",
        f"- domain length: `{preset['domain_length']}`",
        f"- state dimension: `{preset['state_dim']}`",
        f"- latent dimension: `{preset['latent_dim']}`",
        f"- reservoir dimension: `{preset['reservoir_dim']}`",
        "",
        "## Core Results",
        "",
        f"- CAE best validation MSE: `{cae_result['best_val_loss']:.6g}` at epoch `{cae_result['best_epoch']}`",
        f"- ESN best validation rollout MSE: `{esn_result['best_validation_mse']:.6g}`",
        f"- reference lambda_1: `{reference_result['lambda_1']:.6g}`",
        f"- ESN lambda_1: `{latent_result['lambda_1']:.6g}`",
        f"- ESN positive exponents: `{latent_result['n_positive']}`",
        f"- mean CLV-angle Wasserstein: `{angle_metrics['mean_wasserstein']:.6g}`",
        "",
        "## Figures",
        "",
        f"- Fig 4 analogue: `fig4_reconstruction_mse_{preset_name}.png`",
        f"- Fig 7 analogue: `fig7_lyapunov_spectrum_{preset_name}.png`",
        f"- Fig 8 analogue: `fig8_clv_angle_distributions_{preset_name}.png`",
    ]
    if extension_result is not None:
        lines.extend(
            [
                "",
                "## Extension",
                "",
                f"- status: `{extension_result.get('status', 'unknown')}`",
                f"- note: `{extension_result.get('reason', 'completed')}`"
                if extension_result.get("status") != "completed"
                else f"- extension lambda_1: `{extension_result['lambda_1']:.6g}`",
            ]
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_single_preset(
    preset_name: str,
    *,
    force: bool = False,
    only: str | None = None,
    allow_generate: bool = False,
) -> dict:
    preset = get_preset(preset_name)
    suffix = preset.name
    only = None if only in (None, "all") else only
    stage_order = ("cae", "esn", "ref", "latent", "diag", "ext", "summary")

    def wants(stage: str) -> bool:
        return only is None or stage_order.index(stage) <= stage_order.index(only)

    cae_ckpt = DATA_DIR / f"{suffix}_cae.pkl"
    cae_hist = DATA_DIR / f"{suffix}_cae_history.json"
    cae_sweep = DATA_DIR / f"{suffix}_cae_sweep.json"
    esn_ckpt = DATA_DIR / f"{suffix}_esn.pkl"
    esn_hist = DATA_DIR / f"{suffix}_esn_history.json"
    results_md = DATA_DIR / f"RESULTS_{suffix}.md"

    train_x, val_x, test_x, meta = ensure_dataset(preset, allow_generate=allow_generate)
    train_pairs = one_step_pairs(train_x)
    val_pairs = one_step_pairs(val_x)
    test_pairs = one_step_pairs(test_x)
    _ = train_pairs, val_pairs, test_pairs

    cae_result = load_pickle(cae_ckpt) if cae_ckpt.exists() and not force else None
    if cae_result is None and wants("cae"):
        cae_config = CAEConfig(n_grid=preset.state_dim, latent_dim=preset.latent_dim)
        train_config = CAETrainingConfig(epochs=200 if preset.name == "project" else 250)
        cae_result = train_cae(
            train_x,
            val_x,
            cae_config=cae_config,
            train_config=train_config,
            checkpoint_path=cae_ckpt,
            history_path=cae_hist,
        )
        sweep_curves = latent_dimension_sweep(
            train_x,
            val_x,
            n_grid=preset.state_dim,
            latent_dims=latent_sweep_dims(preset.name, preset.latent_dim),
            base_train_config=CAETrainingConfig(epochs=80, eval_every=5, patience_epochs=15),
        )
        write_json(cae_sweep, sweep_curves)
    elif cae_result is None:
        cae_result = load_pickle(cae_ckpt)

    esn_result = load_pickle(esn_ckpt) if esn_ckpt.exists() and not force else None
    if esn_result is None and wants("esn"):
        cae_config = CAEConfig(n_grid=preset.state_dim, latent_dim=preset.latent_dim)
        z_train, z_val, z_test = encode_splits(
            cae_result,
            cae_config,
            train_x,
            val_x,
            test_x,
        )
        base_esn = ESNConfig(n_input=preset.latent_dim, n_reservoir=preset.reservoir_dim)
        search_cfg = ESNSearchConfig(
            spectral_radii=(0.7, 0.9, 1.1) if preset.name == "project" else (0.8, 0.95, 1.1),
            input_scales=(0.1, 0.25, 0.5),
            leak_rates=(0.3, 0.6, 1.0),
            ridge_betas=(1.0e-8, 1.0e-6, 1.0e-4),
        )
        esn_result = fit_best_esn(
            z_train,
            z_val,
            base_config=base_esn,
            search_config=search_cfg,
            checkpoint_path=esn_ckpt,
            history_path=esn_hist,
        )
    elif esn_result is None:
        esn_result = load_pickle(esn_ckpt)

    ref_summary_path = DATA_DIR / f"{suffix}_reference_clv_summary.json"
    if ref_summary_path.exists() and not force:
        reference_result = load_pickle(DATA_DIR / f"{suffix}_reference_clv.pkl")
    elif wants("ref"):
        reference_result = compute_reference_clv(
            preset,
            config=ReferenceCLVConfig(n_steps=12000 if preset.name == "paper" else 8000, n_clv=12),
            data_dir=DATA_DIR,
        )
    else:
        reference_result = load_pickle(DATA_DIR / f"{suffix}_reference_clv.pkl")

    esn_clv_summary = DATA_DIR / f"{suffix}_esn_clv_summary.json"
    if esn_clv_summary.exists() and not force:
        latent_result = load_pickle(DATA_DIR / f"{suffix}_esn_clv.pkl")
    elif wants("latent"):
        cae_config = CAEConfig(n_grid=preset.state_dim, latent_dim=preset.latent_dim)
        _, _, z_test = encode_splits(cae_result, cae_config, train_x, val_x, test_x)
        latent_result = compute_esn_clv(
            preset,
            esn_artifact=esn_result,
            z_reference=z_test,
            config=LatentCLVConfig(
                n_steps=4000 if preset.name == "project" else 6000,
                n_clv=12,
                warmup_steps=400,
            ),
            data_dir=DATA_DIR,
            pairs=reference_result["pairs"],
        )
    else:
        latent_result = load_pickle(DATA_DIR / f"{suffix}_esn_clv.pkl")

    fig4 = tag_path(FIG_DIR / "fig4_reconstruction_mse.png", suffix)
    fig7 = tag_path(FIG_DIR / "fig7_lyapunov_spectrum.png", suffix)
    fig8 = tag_path(FIG_DIR / "fig8_clv_angle_distributions.png", suffix)
    if wants("diag"):
        sweep_curves = load_json(cae_sweep) if cae_sweep.exists() else {
            "latent_dim": [preset.latent_dim],
            "val_mse": [cae_result["best_val_loss"]],
        }
        plot_reconstruction_curve(
            sweep_curves,
            fig4,
            title=f"Ozalp Fig. 4 analogue ({suffix})",
        )
        plot_lyapunov_spectrum(
            np.asarray(reference_result["exponents"]),
            np.asarray(latent_result["exponents"]),
            fig7,
            title=f"Ozalp Fig. 7 analogue ({suffix})",
            skip_reference_neutral=2 if preset.name == "paper" else 0,
        )
        angle_metrics = plot_angle_histograms(
            np.asarray(reference_result["angles"]),
            np.asarray(latent_result["angles"]),
            [tuple(pair) for pair in reference_result["pairs"]],
            fig8,
            title=f"Ozalp Fig. 8 analogue ({suffix})",
        )
        write_json(DATA_DIR / f"{suffix}_angle_metrics.json", angle_metrics)
    else:
        angle_metrics = load_json(DATA_DIR / f"{suffix}_angle_metrics.json")

    extension_result = None
    if wants("ext"):
        if preset.name == "project":
            extension_result = run_linot_extension(
                root=ROOT,
                config=LinotExtensionConfig(),
                output_dir=DATA_DIR,
            )
        else:
            extension_result = {
                "status": "skipped",
                "reason": "Cross-paper Linot extension is only meaningful for the project-aligned preset.",
            }

    if wants("summary"):
        write_results_md(
            preset_name=suffix,
            results_path=results_md,
            preset={
                "domain_length": preset.domain_length,
                "state_dim": preset.state_dim,
                "latent_dim": preset.latent_dim,
                "reservoir_dim": preset.reservoir_dim,
            },
            cae_result=cae_result,
            esn_result=esn_result,
            reference_result={
                "lambda_1": float(np.asarray(reference_result["exponents"])[0]),
            },
            latent_result=latent_result,
            angle_metrics=angle_metrics,
            extension_result=extension_result,
        )

    return {
        "preset": preset_name,
        "cae_best_val_loss": float(cae_result["best_val_loss"]),
        "esn_best_validation_mse": float(esn_result["best_validation_mse"]),
        "reference_lambda_1": float(np.asarray(reference_result["exponents"])[0]),
        "esn_lambda_1": float(np.asarray(latent_result["exponents"])[0]),
        "mean_angle_wasserstein": float(angle_metrics["mean_wasserstein"]),
        "results_md": str(results_md),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Ozalp 2024 KSE reproduction scaffold.")
    parser.add_argument("--preset", choices=("paper", "project", "both"), default="project")
    parser.add_argument(
        "--only",
        choices=("all", "cae", "esn", "ref", "latent", "diag", "ext", "summary"),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-generate", action="store_true")
    args = parser.parse_args()

    preset_names = ("paper", "project") if args.preset == "both" else (args.preset,)
    summaries = []
    for preset_name in preset_names:
        summaries.append(
            run_single_preset(
                preset_name,
                force=args.force,
                only=args.only,
                allow_generate=args.allow_generate,
            )
        )

    summary_path = DATA_DIR / "run_reproduction_summary.json"
    write_json(summary_path, {"runs": summaries})
    print(summary_path)
    for item in summaries:
        print(item["results_md"])


if __name__ == "__main__":
    main()
