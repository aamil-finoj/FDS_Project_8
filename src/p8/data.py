"""Seeding and the generative model of the assignment."""
import os
import random

import numpy as np
import torch

from .config import DataConfig


def set_seed(seed: int) -> None:
    """Fix Python, NumPy and PyTorch (CPU + CUDA) random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sample_prior(n: int, a: float, q: float, rng: np.random.Generator) -> np.ndarray:
    """Sample t from the (possibly asymmetric) piecewise-uniform prior.

    p_q(t) = (1-q)/a on [-a, 0) and q/a on [0, a].  q = 0.5 gives Unif[-a, a].
    Sampling: draw the sign (+ with prob. q), then |t| ~ Unif[0, a].
    """
    magnitude = rng.uniform(0.0, a, size=n)
    sign = np.where(rng.uniform(size=n) < q, 1.0, -1.0)
    return sign * magnitude


def generate_data(n: int, a: float, sigma: float, q: float, rng: np.random.Generator):
    """Draw n pairs (x, t) with t ~ p_q(t) and x = t^2 + eps, eps ~ N(0, sigma^2)."""
    t = sample_prior(n, a, q, rng)
    x = t**2 + sigma * rng.standard_normal(n)
    return x.astype(np.float64), t.astype(np.float64)


def make_splits(cfg: DataConfig) -> dict:
    """Generate one dataset and split it into train / val / test.

    All three splits are i.i.d. draws from the same model, generated from a single
    RNG seeded with cfg.data_seed, so the split is fully reproducible.
    Returns {"train": (x, t), "val": (x, t), "test": (x, t)}.
    """
    rng = np.random.default_rng(cfg.data_seed)
    n_total = cfg.n_train + cfg.n_val + cfg.n_test
    x, t = generate_data(n_total, cfg.a, cfg.sigma, cfg.q, rng)
    i1, i2 = cfg.n_train, cfg.n_train + cfg.n_val
    return {"train": (x[:i1], t[:i1]), "val": (x[i1:i2], t[i1:i2]), "test": (x[i2:], t[i2:])}
