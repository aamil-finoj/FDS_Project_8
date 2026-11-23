"""Selected experimental settings.

None of these numbers are prescribed by the assignment (it only says a, sigma > 0).
They are our choices; the notebook explains why each was chosen.
"""
from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class DataConfig:
    a: float = 2.0            # prior support [-a, a]
    sigma: float = 0.15       # observation-noise std (units of x)
    q: float = 0.5            # P(t > 0) under the prior; 0.5 = symmetric baseline
    n_train: int = 10_000
    n_val: int = 2_000
    n_test: int = 2_000
    data_seed: int = 2026     # fixes the dataset (separate from model-init seeds)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    hidden: int = 64          # units per hidden layer
    n_layers: int = 3         # hidden layers in the shared trunk
    lr: float = 2e-3          # Adam learning rate
    batch_size: int = 256
    max_epochs: int = 400
    patience: int = 40        # early stopping: epochs without val improvement
    grad_clip: float = 5.0    # max gradient norm (stability guard)
    weight_decay: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MDNConfig:
    M: int = 3                # number of mixture components
    var_floor: float = 1e-4   # v_min added after softplus (a VARIANCE, so std floor = 0.01)
    mu_init: str = "default"  # "default" (PyTorch init) or "spread" (means start spread over [-a, a])
    truncate: bool = False    # if True, truncate each Gaussian component to [-a, a]

    def to_dict(self):
        d = asdict(self)
        if not self.truncate:
            d.pop("truncate")
        return d


# Bump this whenever model/training code changes so stale cached runs are never reused.
CODE_VERSION = "v1"
