"""Standalone tests for Part 3 & §7.3: the MDN and truncated MDN in src/p8/models.py & evaluate.py.

Run:  python tests/test_mdn.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import norm, truncnorm

from p8.config import DataConfig, MDNConfig, TrainConfig
from p8.data import make_splits
from p8 import evaluate as E
from p8.models import (
    MDN,
    log_norm_cdf_diff,
    mdn_nll_loss,
    truncated_gaussian_log_pdf,
)
from p8.training import DEVICE, train_mdn

FAILURES = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def run_all():
    torch.manual_seed(0)
    a = 2.0

    # 1. Standard MDN invariants & agreement with SciPy
    m = MDN(1.5, 1.2, M=3, var_floor=1e-4, a=a, truncate=False).to(DEVICE)
    x = np.array([-0.5, 0.2, 1.0, 3.5, 4.5])
    t = np.array([0.1, -0.4, 1.0, -1.8, 1.95])
    pi, mu, var = E.mdn_params(m, x)
    check("weights sum to 1", np.allclose(pi.sum(1), 1.0, atol=1e-6), f"max err={np.abs(pi.sum(1)-1).max():.2e}")
    check("variances strictly above floor", bool((var > m.var_floor).all()), f"min={var.min():.2e}")
    manual_lp = np.log((pi * norm.pdf(t[:, None], mu, np.sqrt(var))).sum(1))
    with torch.no_grad():
        torch_lp = m.log_prob(E._t(t), E._t(x)).cpu().numpy()
    check("log_prob matches SciPy", np.allclose(manual_lp, torch_lp, atol=1e-5),
          f"max diff={np.abs(manual_lp - torch_lp).max():.2e}")

    # 2. logsumexp forward + backward stability in extreme tails
    xt = torch.tensor([1.0], device=DEVICE)
    tt = torch.tensor([30.0], device=DEVICE)
    lp, m_, v_ = m(xt)
    naive_lp = torch.log((lp.exp() * torch.exp(-0.5 * (tt[:, None] - m_) ** 2 / v_) / torch.sqrt(2 * np.pi * v_)).sum())
    stable_loss = -m.log_prob(tt, xt).mean()
    m.zero_grad()
    stable_loss.backward()
    grads_ok = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    check("naive underflows at t=30", naive_lp.item() == float("-inf"), f"naive={naive_lp.item()}")
    check("logsumexp finite + finite grads at t=30", np.isfinite(stable_loss.item()) and grads_ok,
          f"nll={stable_loss.item():.1f}")

    # 3. log_norm_cdf_diff accuracy in central and extreme tail regimes
    pairs = [(-2.0, 2.0), (-10.0, 10.0), (-1e-3, 1e-3), (-25.0, -20.0), (20.0, 25.0)]
    lo_t = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    hi_t = torch.tensor([p[1] for p in pairs], dtype=torch.float64)
    got = log_norm_cdf_diff(lo_t, hi_t).numpy()
    expected = []
    for lo, hi in pairs:
        if lo + hi > 0:
            lo, hi = -hi, -lo
        expected.append(norm.logcdf(hi) + np.log(-np.expm1(norm.logcdf(lo) - norm.logcdf(hi))))
    expected = np.array(expected)
    check("log_norm_cdf_diff matches float64 SciPy", np.allclose(got, expected, atol=1e-10),
          f"max err={np.abs(got - expected).max():.2e}")

    # 4. Truncated Gaussian log_pdf matches scipy.stats.truncnorm
    mu_t = torch.tensor([-3.5, -1.5, 0.0, 1.8, 3.5], dtype=torch.float64)
    var_t = torch.tensor([0.2, 0.05, 0.3, 0.02, 0.5], dtype=torch.float64)
    t_test = torch.tensor([-1.9, -1.4, 0.2, 1.95, 1.7], dtype=torch.float64)
    got_tn = truncated_gaussian_log_pdf(t_test, mu_t, var_t, a).numpy()
    sd_np = np.sqrt(var_t.numpy())
    exp_tn = truncnorm.logpdf(t_test.numpy(), (-a - mu_t.numpy()) / sd_np, (a - mu_t.numpy()) / sd_np,
                              loc=mu_t.numpy(), scale=sd_np)
    check("truncated_gaussian_log_pdf matches truncnorm", np.allclose(got_tn, exp_tn, atol=1e-8),
          f"max err={np.abs(got_tn - exp_tn).max():.2e}")

    # 5. Truncated MDN density normalisation, zero out-of-support mass, and evaluate.py helpers
    tm = MDN(1.5, 1.2, M=3, var_floor=1e-4, a=a, truncate=True).to(DEVICE)
    with torch.no_grad():
        # Push one component mean outside [-a, a] to verify boundary safety
        tm.mu_head.bias.copy_(torch.tensor([-2.5, 0.0, 2.5], device=DEVICE))
    grid = np.linspace(-a, a, 40_001)
    h = (2 * a) / (len(grid) - 1)
    dens = E.predict_conditional_density(tm, x, grid)
    integrals = np.trapezoid(dens, grid, axis=1)
    check("truncated MDN integrates to 1 on [-a, a]", np.allclose(integrals, 1.0, atol=1e-4),
          f"max err={np.abs(integrals - 1.0).max():.2e}")

    out_mass = E.out_of_support_mass(tm, x, a)
    check("truncated MDN out_of_support_mass == 0", np.allclose(out_mass, 0.0, atol=1e-6),
          f"max={out_mass.max():.2e}")

    with torch.no_grad():
        lp_out = tm.log_prob(E._t([-2.1, 2.1]), E._t([1.0, 1.0])).cpu().numpy()
    check("truncated MDN log_prob outside [-a, a] is -inf", bool(np.all(lp_out == float("-inf"))))

    # Branch probability and mixture_mean vs fine-grid integration
    p_pos = E.model_branch_probability(tm, x)
    pos_mask = grid >= 0
    p_pos_grid = np.trapezoid(dens[:, pos_mask], grid[pos_mask], axis=1)
    check("truncated model_branch_probability matches grid", np.allclose(p_pos, p_pos_grid, atol=1e-4),
          f"max err={np.abs(p_pos - p_pos_grid).max():.2e}")

    mean_exact = E.mixture_mean(tm, x)
    mean_grid = np.trapezoid(dens * grid[None, :], grid, axis=1)
    check("truncated mixture_mean matches grid", np.allclose(mean_exact, mean_grid, atol=1e-4),
          f"max err={np.abs(mean_exact - mean_grid).max():.2e}")

    # Sampling stays inside [-a, a] and matches mean
    samples = E.sample_conditional(tm, [1.0, 4.5], 20_000, seed=42)
    check("truncated samples strictly inside [-a, a]",
          bool((samples >= -a).all() and (samples <= a).all()),
          f"range=[{samples.min():.4f}, {samples.max():.4f}]")
    check("truncated sample mean matches mixture_mean",
          np.allclose(samples.mean(1), E.mixture_mean(tm, [1.0, 4.5]), atol=0.03),
          f"diff={np.abs(samples.mean(1) - E.mixture_mean(tm, [1.0, 4.5])).max():.4f}")

    # Backward pass finite when means are pushed past [-a, a]
    tm.zero_grad()
    loss_tm = mdn_nll_loss(tm, E._t(x), E._t(t))
    loss_tm.backward()
    tm_grads_ok = all(torch.isfinite(p.grad).all().item() for p in tm.parameters() if p.grad is not None)
    check("truncated MDN finite loss & grads", np.isfinite(loss_tm.item()) and tm_grads_ok)

    # 6. Short training smoke test
    dcfg = DataConfig(n_train=1000, n_val=200, n_test=200)
    sp = make_splits(dcfg)
    trained, hist, _ = train_mdn(dcfg, TrainConfig(max_epochs=15, patience=10),
                                 MDNConfig(M=3, truncate=True), seed=0, splits=sp, use_cache=False)
    smoke_out = E.out_of_support_mass(trained, sp["test"][0], dcfg.a)
    check("truncated MDN short train finite & zero out_mass",
          np.isfinite(hist["best_val"]) and np.allclose(smoke_out, 0.0, atol=1e-6),
          f"best_val={hist['best_val']:.3f}, max_out={smoke_out.max():.2e}")

    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} checks failed: {FAILURES}")
    print("ALL MDN CHECKS PASSED")


if __name__ == "__main__":
    run_all()
