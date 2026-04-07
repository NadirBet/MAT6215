"""
run_all.py - Master Orchestration Script
=========================================
Runs all pending tasks in dependency order, skipping already-completed tasks
(detected by presence of output data files).

Dependency graph:
  T3  -> independent (data ready)
  T4  -> independent (data ready)
  T5  -> independent (data ready)
  T6  -> independent (data ready)
  T8  -> independent (data ready)
  T11 -> independent, but can compare to T8 output
  T12 -> prefers T4 done (constrained_a_results.pkl)
  T13 -> prefers T4 done
  T14 -> independent (retrains from scratch)
  T15 -> all previous tasks done

Usage:
  python run_all.py              # run all pending tasks
  python run_all.py --from T6    # start from T6 (skip T3-T5)
  python run_all.py --only T8    # run only T8
  python run_all.py --list       # list status without running
"""

import sys
import os
import subprocess
import argparse
import time

sys.path = [p for p in sys.path if 'MAT6215' not in p]
sys.path.insert(0, '.')

PYTHON = sys.executable

TASKS = [
    {
        "id": "T3",
        "script": "run_t3_jac_sweep.py",
        "outputs": ["data/jac_sweep_epochs.pkl", "data/jac_sweep_lambda.pkl"],
        "description": "JAC epoch + lambda sweep",
        "deps": [],
    },
    {
        "id": "T4",
        "script": "run_t4_constrained_a.py",
        "outputs": ["data/constrained_a_results.pkl"],
        "description": "Constrained-A variants (4 architectures)",
        "deps": [],
    },
    {
        "id": "T5",
        "script": "run_t5_latent_dim_sweep.py",
        "outputs": ["data/latent_dim_sweep.pkl"],
        "description": "Latent dim sweep d=4..16",
        "deps": [],
    },
    {
        "id": "T6",
        "script": "run_t6_tau_sweep.py",
        "outputs": ["data/tau_sweep_results.pkl"],
        "description": "Tau sweep: latent ODE vs discrete map",
        "deps": [],
    },
    {
        "id": "T8",
        "script": "run_t8_sindy_sweep.py",
        "outputs": ["data/sindy_sweep_results.pkl"],
        "description": "SINDy threshold + library + derivative sweeps",
        "deps": [],
    },
    {
        "id": "T11",
        "script": "run_t11_discrete_sindy.py",
        "outputs": ["data/discrete_sindy_results.pkl"],
        "description": "Discrete-time SINDy (no derivative estimation)",
        "deps": [],   # T8 output optional (comparison only)
    },
    {
        "id": "T12",
        "script": "run_t12_clv_surrogates.py",
        "outputs": ["data/clv_results.pkl"],
        "description": "CLV angles: true KSE vs surrogates",
        "deps": ["data/constrained_a_results.pkl"],
    },
    {
        "id": "T13",
        "script": "run_t13_ensemble_errors.py",
        "outputs": ["data/ensemble_error_results.pkl"],
        "description": "Ensemble short-time forecast errors",
        "deps": ["data/constrained_a_results.pkl"],
    },
    {
        "id": "T14",
        "script": "run_t14_multiseed.py",
        "outputs": ["data/multiseed_results.pkl"],
        "description": "Multi-seed robustness (5 seeds)",
        "deps": [],
    },
    {
        "id": "T15",
        "script": "run_t15_ablations.py",
        "outputs": ["data/ablation_table.pkl"],
        "description": "Ablation table (aggregates all results)",
        "deps": [],   # reads whatever pkl files are available
    },
    {
        "id": "T16",
        "script": "run_t16_traj_supervision.py",
        "outputs": ["data/traj_supervision_results.pkl"],
        "description": "Trajectory-supervision vs vector-field training",
        "deps": [],
    },
    {
        "id": "T19",
        "script": "run_t19_jacobian_geometry.py",
        "outputs": ["data/jacobian_geometry_results.pkl"],
        "description": "Local Jacobian singular value analysis",
        "deps": [],
    },
    {
        "id": "DIAG",
        "script": "run_diagnostics_all.py",
        "outputs": ["data/diagnostics_all.pkl"],
        "description": "Full diagnostics for all surrogates (W1, PDF, autocorr)",
        "deps": ["data/constrained_a_results.pkl"],
    },
    {
        "id": "T22",
        "script": "run_t22_update_report.py",
        "outputs": [],   # always re-run (report update)
        "description": "Update FINAL_REPORT.md with all results",
        "deps": [],
    },
]


def is_done(task):
    return all(os.path.exists(o) for o in task["outputs"])


def deps_met(task):
    return all(os.path.exists(d) for d in task["deps"])


def print_status():
    print("\n" + "="*65)
    print(f"{'Task':<6} {'Status':<10} {'Deps':<8} {'Description'}")
    print("-"*65)
    for t in TASKS:
        done = is_done(t)
        dep_ok = deps_met(t)
        status = "DONE" if done else ("READY" if dep_ok else "WAITING")
        print(f"{t['id']:<6} {status:<10} {'OK' if dep_ok else 'MISSING':<8} {t['description']}")
    print("="*65)


def run_task(task):
    print(f"\n{'='*65}")
    print(f"Running {task['id']}: {task['description']}")
    print(f"Script: {task['script']}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run([PYTHON, task["script"]], check=False)
    elapsed = time.time() - t0
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n{task['id']} {status} in {elapsed:.1f}s")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="MAT6215 Task Runner")
    parser.add_argument("--list", action="store_true", help="List task status only")
    parser.add_argument("--from", dest="from_task", default=None,
                        help="Start from this task ID (skip earlier)")
    parser.add_argument("--only", dest="only_task", default=None,
                        help="Run only this task ID")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if outputs already exist")
    args = parser.parse_args()

    print_status()
    if args.list:
        return

    task_ids = [t["id"] for t in TASKS]
    run_ids = task_ids

    if args.only_task:
        run_ids = [args.only_task]
    elif args.from_task:
        try:
            start_idx = task_ids.index(args.from_task)
            run_ids = task_ids[start_idx:]
        except ValueError:
            print(f"Unknown task: {args.from_task}")
            sys.exit(1)

    tasks_to_run = [t for t in TASKS if t["id"] in run_ids]
    n_done = 0; n_skip = 0; n_fail = 0

    for task in tasks_to_run:
        if is_done(task) and task["outputs"] and not args.force:
            print(f"\n  {task['id']}: SKIP (outputs exist)")
            n_skip += 1
            continue
        if not deps_met(task):
            print(f"\n  {task['id']}: SKIP (dependencies not met: {task['deps']})")
            n_skip += 1
            continue
        success = run_task(task)
        if success:
            n_done += 1
        else:
            n_fail += 1
            print(f"\n  WARNING: {task['id']} failed. Continuing with next task.")

    print(f"\n{'='*65}")
    print(f"Summary: {n_done} completed, {n_skip} skipped, {n_fail} failed")
    print(f"{'='*65}")
    print_status()


if __name__ == "__main__":
    main()
