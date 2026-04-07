from __future__ import annotations

import argparse
from pathlib import Path

from paper_reproductions.linot2021_ks.data_utils import RES_DIR, ensure_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare cached KS datasets for Linot Figure 2 at L=22,44,66."
    )
    parser.add_argument("--domains", nargs="+", type=float, default=[22.0, 44.0, 66.0])
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-test", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for L in args.domains:
        progress_path = RES_DIR / f"prepare_figure2_l{int(round(L))}_progress.json"
        ensure_dataset(
            L,
            N=args.state_dim,
            dt=args.dt,
            seed=args.seed,
            n_warmup=args.warmup,
            n_train=args.n_train,
            n_test=args.n_test,
            allow_generate=True,
            progress_path=progress_path,
        )
        print(f"Prepared dataset for L={L:.0f}")
        print(progress_path)


if __name__ == "__main__":
    main()
