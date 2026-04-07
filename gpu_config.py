"""
gpu_config.py - JAX Performance Configuration
=============================================
Import this at the top of every training script BEFORE importing jax.

On Windows (current setup):
  - JAX CUDA is NOT supported on Windows natively
  - Uses CPU with maximum thread parallelism (12 cores detected)
  - XLA_FLAGS enables multi-threaded Eigen and intra-op parallelism

To enable GPU (requires WSL2 + Ubuntu):
  1. Install WSL2: wsl --install
  2. Restart, open Ubuntu terminal
  3. Install CUDA in WSL2: https://developer.nvidia.com/cuda-downloads
  4. pip install "jax[cuda12]"
  5. Run all scripts from WSL2 terminal -- 10-30x speedup expected

GPU notes for GTX 1660 Ti (6GB VRAM, CUDA 12.6):
  - All our models fit comfortably (N=64, largest array < 100MB)
  - JAC Jacobian computation (64x64 per sample) would benefit most: ~10x
  - Lyapunov lax.scan (JIT'd): ~5x
  - AE training: ~8x
  - Expected total time per T3 sweep: ~2 min GPU vs ~30 min CPU

CPU parallelism strategy (current):
  - XLA multi-threading: all 12 cores
  - jax.vmap: batches the forward pass across samples (already used)
  - jax.pmap: not used (single device)
  - jax.lax.scan: replaces Python for-loops in Benettin (already used)
"""

import os

# Enable all CPU threads for XLA
os.environ.setdefault("XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=12")
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

# Suppress noisy TF/XLA warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Lyapunov convergence policy ───────────────────────────────────────────────
# Sweep checkpoints use a two-tier strategy:
#   1. Screen at 500 steps (fast, reliable for coarse ranking)
#   2. Escalate to 1500 steps if any |exponent| < NEAR_ZERO_THRESHOLD
#      (near-zero exponents converge slowly; n_pos can flip around zero)
# Final canonical results always use 1500+ steps.
NEAR_ZERO_THRESHOLD = 0.02  # in same units as Lyapunov exponents

# Try to use GPU if available (will silently fall back to CPU on Windows)
try:
    import jax
    gpus = jax.devices("gpu")
    if gpus:
        print(f"[gpu_config] GPU found: {gpus}")
    else:
        print(f"[gpu_config] No GPU — using CPU with {os.environ['OMP_NUM_THREADS']} threads")
except RuntimeError:
    # jax.devices("gpu") raises RuntimeError when no GPU backend is available
    print(f"[gpu_config] No GPU backend — using CPU with {os.environ['OMP_NUM_THREADS']} threads")
