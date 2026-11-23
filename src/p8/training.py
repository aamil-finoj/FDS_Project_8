"""Training loop (shared by all models) + on-disk caching of trained runs.

Protocol for every model:
  * Adam optimiser, mini-batches drawn by a seeded random permutation each epoch
  * gradient-norm clipping (documented guard against rare NLL spikes)
  * after every epoch compute the full validation loss; keep the best-validation
    weights; stop after `patience` epochs without improvement (early stopping)
  * restore the best-validation checkpoint at the end
The test split is never touched here.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import CODE_VERSION, DataConfig, MDNConfig, TrainConfig
from .data import make_splits, set_seed
from .models import MDN, MSENet, mae_loss, mdn_nll_loss, mse_loss

LOSSES = {"mse": mse_loss, "mae": mae_loss, "mdn": mdn_nll_loss}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = Path("artifacts/cache")


def _to_tensor(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEVICE)


def fit(model, loss_fn, train, val, tcfg: TrainConfig, seed: int):
    """Generic training loop. Returns (model with best-val weights, history dict)."""
    set_seed(seed)
    model.to(DEVICE)
    xtr, ttr = map(_to_tensor, train)
    xva, tva = map(_to_tensor, val)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = len(xtr)
    hist = {"train": [], "val": []}
    best_val, best_state, best_epoch, bad = float("inf"), None, -1, 0
    t0 = time.time()
    for epoch in range(tcfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(DEVICE)
        running, count = torch.zeros((), device=DEVICE), 0  # summed on-device: avoids a GPU sync per step
        for i in range(0, n, tcfg.batch_size):
            idx = perm[i:i + tcfg.batch_size]
            loss = loss_fn(model, xtr[idx], ttr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            running += loss.detach() * len(idx)
            count += len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model, xva, tva).item()
        hist["train"].append(running.item() / count)
        hist["val"].append(v)
        if v < best_val - 1e-5:
            best_val, best_epoch, bad = v, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= tcfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    hist.update(best_val=best_val, best_epoch=best_epoch, epochs_run=epoch + 1,
                seconds=time.time() - t0)
    return model, hist


# ----------------------------------------------------------------------------- builders
def build_model(kind: str, train, tcfg: TrainConfig, mcfg: MDNConfig | None = None, a=2.0):
    x_mean, x_std = float(np.mean(train[0])), float(np.std(train[0]))
    if kind in ("mse", "mae"):  # same point-predictor architecture, different loss
        return MSENet(x_mean, x_std, tcfg.hidden, tcfg.n_layers)
    return MDN(x_mean, x_std, M=mcfg.M, hidden=tcfg.hidden, n_layers=tcfg.n_layers,
               var_floor=mcfg.var_floor, mu_init=mcfg.mu_init, a=a, truncate=mcfg.truncate)


def _run_key(kind, dcfg, tcfg, mcfg, seed):
    payload = dict(code=CODE_VERSION, kind=kind, data=dcfg.to_dict(), train=tcfg.to_dict(),
                   mdn=mcfg.to_dict() if mcfg else None, seed=seed)
    s = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16], payload


def train_run(kind: str, dcfg: DataConfig, tcfg: TrainConfig, mcfg: MDNConfig | None = None,
              seed: int = 0, splits=None, use_cache=True, verbose=False):
    """Train one model ('mse', 'mae' or 'mdn'; single Gaussian = 'mdn' with M=1) and cache it.

    The cache key hashes every setting (+ CODE_VERSION), so a cached result is reused
    only for an identical configuration. Returns (model, history, settings_dict).
    """
    key, payload = _run_key(kind, dcfg, tcfg, mcfg, seed)
    path = CACHE_DIR / f"{kind}_{key}.pt"
    if splits is None:
        splits = make_splits(dcfg)
    model = build_model(kind, splits["train"], tcfg, mcfg, a=dcfg.a)
    if use_cache and path.exists():
        blob = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(blob["state_dict"])
        model.to(DEVICE).eval()
        return model, blob["history"], payload
    loss_fn = LOSSES[kind]
    model, hist = fit(model, loss_fn, splits["train"], splits["val"], tcfg, seed)
    if verbose:
        print(f"[{kind}] seed={seed} best val={hist['best_val']:.4f} @ epoch {hist['best_epoch']}"
              f" ({hist['epochs_run']} epochs, {hist['seconds']:.1f}s)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "history": hist, "settings": payload}, path)
    return model, hist, payload


def train_mse_model(dcfg, tcfg, seed=0, **kw):
    return train_run("mse", dcfg, tcfg, None, seed, **kw)


def train_mae_model(dcfg, tcfg, seed=0, **kw):
    """Point predictor trained with absolute error: estimates the conditional median."""
    return train_run("mae", dcfg, tcfg, None, seed, **kw)


def train_single_gaussian_model(dcfg, tcfg, var_floor=1e-4, seed=0, **kw):
    """Single-Gaussian conditional network N(mu(x), v(x)) = the MDN code with M = 1."""
    return train_run("mdn", dcfg, tcfg, MDNConfig(M=1, var_floor=var_floor), seed, **kw)


def train_mdn(dcfg, tcfg, mcfg, seed=0, **kw):
    return train_run("mdn", dcfg, tcfg, mcfg, seed, **kw)
