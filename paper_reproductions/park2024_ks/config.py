from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Park2024KSConfig:
    """
    Configuration for the Park 2024 KS reproduction.

    Notes:
    - The paper states 127 interior nodes, dt=0.25, 3000 epochs,
      train/test size [3000, 3000], MLP, lambda=1.
    - The downloaded stacNODE repository confirms the KS settings
      L=128, dx=1, dt=0.25, c=0.4, and 127 interior nodes.
    """

    n_inner: int = 127
    domain_length: float = 128.0
    c_param: float = 0.4
    dt: float = 0.25

    total_simulation_time: float = 1501.0
    train_size: int = 3000
    test_size: int = 3000
    figure8_steps: int = 300
    use_repo_initial_condition: bool = True

    epochs: int = 3000
    batch_size: int = 500
    learning_rate: float = 1.0e-3
    weight_decay: float = 5.0e-4
    jac_lambda: float = 1.0
    mse_batch_size: int = 3000
    mse_learning_rate: float = 1.0e-4
    mse_lr_schedule: str = "constant"
    mse_weight_decay: float = 5.0e-4
    mse_select_best: bool = True
    mse_init_style: str = "pytorch_linear"

    hidden_widths: tuple[int, ...] = (512, 256)

    lyap_steps: int = 30000
    lyap_warmup: int = 1000
    n_lyap: int = 15
    jacobian_chunk_size: int = 64
    training_eval_every: int = 25
    jacobian_progress_every: int = 25
    lyapunov_progress_block_size: int = 250

    data_dir: Path = field(default_factory=lambda: Path("paper_reproductions/park2024_ks/data"))
    fig_dir: Path = field(default_factory=lambda: Path("paper_reproductions/park2024_ks/figures"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_dim(self) -> int:
        return self.n_inner
