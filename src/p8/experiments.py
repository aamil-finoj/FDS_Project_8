"""Small helpers to run repeated (multi-seed / multi-setting) MDN experiments.

A DataContext bundles everything that depends only on the data configuration
(splits, evaluation inputs, reference grids, reference NLL) so it is computed once
and shared by every model trained on that dataset.
"""
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DataConfig, MDNConfig, TrainConfig
from .data import make_splits
from .evaluate import paired_excess_nll, recovery_x_grid, summarise_mdn
from .reference import reference_density_grid, reference_log_density
from .training import train_mdn

T_GRID = np.linspace(-2.0, 2.0, 4001)   # common t-grid (spacing 0.001) for a = 2
N_EVAL = 400                            # metrics are averaged over the first N_EVAL test inputs


@dataclass
class DataContext:
    dcfg: DataConfig
    splits: dict
    x_eval: np.ndarray
    x_rec: np.ndarray
    ref_eval: np.ndarray
    ref_rec: np.ndarray
    ref_nll: float
    ref_lp: np.ndarray        # per-test-point exact log p(t_i | x_i), for paired comparisons


@lru_cache(maxsize=None)
def get_context(dcfg: DataConfig) -> DataContext:
    """Build (once) the data split and all reference quantities for a data configuration."""
    assert dcfg.a == 2.0, "T_GRID assumes a = 2"
    splits = make_splits(dcfg)
    x_te, t_te = splits["test"]
    x_eval = x_te[:N_EVAL]
    x_rec = recovery_x_grid(dcfg.sigma, dcfg.a)
    ref_lp = reference_log_density(t_te, x_te, dcfg.a, dcfg.sigma, dcfg.q)
    return DataContext(
        dcfg=dcfg, splits=splits, x_eval=x_eval, x_rec=x_rec,
        ref_eval=reference_density_grid(x_eval, T_GRID, dcfg.a, dcfg.sigma, dcfg.q),
        ref_rec=reference_density_grid(x_rec, T_GRID, dcfg.a, dcfg.sigma, dcfg.q),
        ref_nll=float(-ref_lp.mean()), ref_lp=ref_lp,
    )


def run_mdn_setting(dcfg: DataConfig, tcfg: TrainConfig, mcfg: MDNConfig, seed: int, use_cache=True):
    """Train (or load from cache) one MDN and return (model, history, summary-row dict)."""
    ctx = get_context(dcfg)
    model, hist, _ = train_mdn(dcfg, tcfg, mcfg, seed=seed, splits=ctx.splits, use_cache=use_cache)
    row = summarise_mdn(model, dcfg, ctx.splits, T_GRID, ctx.x_eval, ctx.x_rec,
                        ref_eval=ctx.ref_eval, ref_rec=ctx.ref_rec, ref_nll=ctx.ref_nll)
    _, excess_se = paired_excess_nll(model, *ctx.splits["test"], ctx.ref_lp)
    row.update(q=dcfg.q, sigma=dcfg.sigma, M=mcfg.M, var_floor=mcfg.var_floor, mu_init=mcfg.mu_init,
               seed=seed, best_epoch=hist["best_epoch"], epochs_run=hist["epochs_run"],
               val_nll=hist["best_val"],          # validation NLL of the restored checkpoint (for model choice)
               excess_nll_se=excess_se)           # paired standard error of excess_nll on the test set
    return model, hist, row


def get_model(dcfg: DataConfig, tcfg: TrainConfig, mcfg: MDNConfig, seed: int, use_cache=True):
    """Just the trained MDN (from cache if available) - for plotting, without recomputing metrics."""
    return train_mdn(dcfg, tcfg, mcfg, seed=seed, splits=get_context(dcfg).splits, use_cache=use_cache)[0]


def run_sweep(settings, tcfg: TrainConfig, seeds, use_cache=True, progress=True):
    """settings: iterable of (DataConfig, MDNConfig). Trains every setting for every seed.

    Returns a tidy DataFrame with one row per (setting, seed). Nothing is filtered out:
    failed / degenerate runs stay in the table.
    """
    rows, settings = [], list(settings)
    total, done = len(settings) * len(seeds), 0
    for dcfg, mcfg in settings:
        for seed in seeds:
            _, _, row = run_mdn_setting(dcfg, tcfg, mcfg, seed, use_cache=use_cache)
            rows.append(row)
            done += 1
            if progress and (done % 10 == 0 or done == total):
                print(f"  {done}/{total} runs done")
    return pd.DataFrame(rows)


def mean_sd_table(df, by, cols):
    """Mean and standard deviation across seeds, formatted as 'mean ± sd'."""
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.copy().astype(object)
    for c in cols:
        out[c] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(m[c], s[c])]
    return out
