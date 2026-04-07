from __future__ import annotations

import json

from paper_reproductions.linot2021_ks.run_figure2_all_hybrid import FIG_DIR, RES_DIR, save_shifted_plot


def main() -> None:
    summary_path = RES_DIR / "figure2_all_hybrid_d.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    curves = summary["curves"]
    out_path = FIG_DIR / "figure2_all_hybrid_d_minus_dm.png"
    save_shifted_plot(curves, out_path)

    summary["shifted_note"] = "Additional Figure 2 variant with x-axis centered by d_M."
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
