"""Evaluation: predicted densities, NLL, sampling, branch metrics, Wasserstein distance.

Conventions
-----------
* All MDN quantities use VARIANCES v; a standard deviation is sqrt(v).
* Branch probability of a model:  P_hat_+(x) = P(t > 0 | x) = sum_j pi_j Phi(mu_j / sqrt(v_j)).
  This is exactly the integral of the predicted density over (0, inf). The reference
  P_+(x) = int_0^a p(t|x) dt is the same quantity, because the reference has no mass
  outside [-a, a].
* A Gaussian mixture has unbounded support. We report the predicted mass outside
  [-a, a] separately (out_of_support) and, for grid-based comparisons (Wasserstein,
  plots normalised on [-a, a]), we state when the model density is truncated to
  [-a, a] and renormalised. The trained model itself is never modified.
"""
import numpy as np
import torch
from scipy.special import ndtr
from scipy.stats import wasserstein_distance

from .reference import reference_density_grid
from .training import DEVICE


def _t(a):
    return torch.as_tensor(np.asarray(a, dtype=np.float32), device=DEVICE)


@torch.no_grad()
def mdn_params(model, x):
    """Return numpy arrays (pi, mu, var), each of shape (len(x), M)."""
    log_pi, mu, var = model(_t(np.atleast_1d(x)))
    return log_pi.exp().cpu().numpy(), mu.cpu().numpy(), var.cpu().numpy()


@torch.no_grad()
def predict_point(model, x):
    return model(_t(np.atleast_1d(x))).cpu().numpy()


def predict_conditional_density(model, x_values, t_grid):
    """Predicted p_theta(t | x) on a grid, shape (len(x_values), len(t_grid)). Not renormalised."""
    pi, mu, var = mdn_params(model, x_values)
    t = np.asarray(t_grid)[None, :, None]
    comp = np.exp(-0.5 * (t - mu[:, None, :]) ** 2 / var[:, None, :]) / np.sqrt(2 * np.pi * var[:, None, :])
    return (pi[:, None, :] * comp).sum(-1)


def mixture_mean(model, x):
    """E_theta[t | x] = sum_j pi_j mu_j  (the MDN's own point summary)."""
    pi, mu, _ = mdn_params(model, x)
    return (pi * mu).sum(-1)


@torch.no_grad()
def log_prob_points(model, x, t):
    """Per-point log p_theta(t_i | x_i) as a float64 numpy array."""
    return model.log_prob(_t(t), _t(x)).cpu().numpy().astype(np.float64)


def evaluate_nll(model, x, t):
    """Average held-out NLL (nats) of a probabilistic model, plus its standard error."""
    lp = log_prob_points(model, x, t)
    return float(-lp.mean()), float(lp.std(ddof=1) / np.sqrt(len(lp)))


def paired_excess_nll(model, x, t, ref_lp):
    """Excess NLL = mean_i [log p(t_i|x_i) - log p_theta(t_i|x_i)] and its PAIRED standard error.

    Both models are scored on the same test points, so the per-point differences are used for the
    standard error. This removes the large point-to-point variation that both NLLs share and gives a
    much tighter (and the correct) uncertainty for comparing models.
    """
    d = np.asarray(ref_lp) - log_prob_points(model, x, t)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def model_cdf(model, x, t):
    """Predicted CDF F_theta(t_i | x_i) = sum_j pi_j Phi((t_i - mu_j) / sqrt(v_j)) for paired arrays.

    Evaluated at the true t on held-out data this is the probability integral transform (PIT):
    for a correct conditional model the PIT values are Uniform(0, 1).
    """
    pi, mu, var = mdn_params(model, x)
    t = np.asarray(t, dtype=float)[:, None]
    return (pi * ndtr((t - mu) / np.sqrt(var))).sum(-1)


def truncated_nll(model, x, t, a):
    """NLL of the model density restricted to the prior support [-a, a] and renormalised.

    log p_trunc = log p_theta - log m(x), with m(x) the model mass inside [-a, a]. This is a diagnostic
    only (the trained model is unchanged): it shows how much likelihood the leaked mass costs.
    """
    x = np.asarray(x, dtype=float)
    inside = model_mass_in_interval(model, x, np.full(len(x), -a), np.full(len(x), a))
    return float(-(log_prob_points(model, x, t) - np.log(inside)).mean())


def sample_conditional(model, x, n_samples, seed=0):
    """Draw n_samples t ~ p_theta(t | x) for each x; returns shape (len(x), n_samples).

    Two-step ancestral sampling: pick component j ~ Categorical(pi(x)), then
    t ~ N(mu_j, v_j) using std = sqrt(v_j).
    """
    rng = np.random.default_rng(seed)
    pi, mu, var = mdn_params(model, x)
    n_x, M = pi.shape
    cum = np.cumsum(pi, axis=1)
    u = rng.uniform(size=(n_x, n_samples))
    comp = (u[:, :, None] > cum[:, None, :]).sum(-1).clip(max=M - 1)
    rows = np.arange(n_x)[:, None]
    return mu[rows, comp] + np.sqrt(var[rows, comp]) * rng.standard_normal((n_x, n_samples))


def sample_reference(x_values, t_grid, a, sigma, q, n_samples, seed=0):
    """Exact-posterior samples by inverse-CDF on a fine grid (used as a residual baseline)."""
    rng = np.random.default_rng(seed)
    dens = reference_density_grid(x_values, t_grid, a, sigma, q)
    out = np.empty((len(dens), n_samples))
    for i, d in enumerate(dens):
        cdf = np.concatenate([[0], np.cumsum(0.5 * (d[1:] + d[:-1]) * np.diff(t_grid))])
        cdf /= cdf[-1]
        out[i] = np.interp(rng.uniform(size=n_samples), cdf, t_grid)
    return out


# --------------------------------------------------------------------- branch metrics
def model_mass_in_interval(model, x, lo, hi):
    """Predicted probability mass in [lo, hi] (lo/hi broadcast against x)."""
    pi, mu, var = mdn_params(model, x)
    sd = np.sqrt(var)
    lo = np.asarray(lo, dtype=float).reshape(-1, 1)
    hi = np.asarray(hi, dtype=float).reshape(-1, 1)
    return (pi * (ndtr((hi - mu) / sd) - ndtr((lo - mu) / sd))).sum(-1)


def model_branch_probability(model, x):
    """P_hat(t > 0 | x) = sum_j pi_j Phi(mu_j / sqrt v_j)."""
    pi, mu, var = mdn_params(model, x)
    return (pi * ndtr(mu / np.sqrt(var))).sum(-1)


def out_of_support_mass(model, x, a):
    """Predicted mass outside the prior support [-a, a] (should be ~0 for a good model)."""
    x = np.atleast_1d(x)
    return 1.0 - model_mass_in_interval(model, x, np.full(len(x), -a), np.full(len(x), a))


def branch_neighbourhoods(x, sigma, a, k=3.0):
    """Neighbourhoods N_+- = [+-sqrt(x) - w, +-sqrt(x) + w] around the reference modes.

    w(x) = k * sigma / (2 sqrt(x)) - k local posterior std's, from the curvature of
    log p at the mode (d^2/dt^2 log p = -4x / sigma^2  =>  sd = sigma / (2 sqrt x)).
    Intervals are clipped to the support [-a, a].
    """
    r = np.sqrt(x)
    w = k * sigma / (2 * r)
    pos = (np.clip(r - w, -a, a), np.clip(r + w, -a, a))
    neg = (np.clip(-r - w, -a, a), np.clip(-r + w, -a, a))
    return pos, neg


def recovery_x_grid(sigma, a, n=21, sep_k=2.0):
    """Observations where two-branch recovery is meaningful: sep_k*sigma <= x <= a^2 - sep_k*sigma.

    Below sep_k*sigma the reference modes merge (dip ratio > exp(-sep_k^2/2)); near/above a^2
    the modes are distorted by the support boundary. The neighbourhoods are disjoint here.
    """
    return np.linspace(sep_k * sigma, a**2 - sep_k * sigma, n)


def evaluate_branch_metrics(model, x_rec, t_grid, a, sigma, q, k=3.0, recovered_frac=0.5, ref=None):
    """Mass captured near each reference branch, relative to the reference mass there.

    capture_+-(x) = m_model(N_+-) / m_ref(N_+-).  A branch counts as *recovered* at x when
    capture >= recovered_frac (default: the model puts at least half the reference mass there).
    `ref` (optional) = precomputed reference_density_grid(x_rec, t_grid, ...).
    """
    (plo, phi), (nlo, nhi) = branch_neighbourhoods(x_rec, sigma, a, k)
    if ref is None:
        ref = reference_density_grid(x_rec, t_grid, a, sigma, q)
    def ref_mass(lo, hi):
        out = []
        for i in range(len(x_rec)):
            m = (t_grid >= lo[i]) & (t_grid <= hi[i])
            out.append(np.trapezoid(ref[i, m], t_grid[m]))
        return np.array(out)
    cap_pos = model_mass_in_interval(model, x_rec, plo, phi) / ref_mass(plo, phi)
    cap_neg = model_mass_in_interval(model, x_rec, nlo, nhi) / ref_mass(nlo, nhi)
    return {
        "capture_pos": cap_pos, "capture_neg": cap_neg,
        "recovered_pos": float(np.mean(cap_pos >= recovered_frac)),
        "recovered_neg": float(np.mean(cap_neg >= recovered_frac)),
    }


# --------------------------------------------------------------------- Wasserstein-1
def evaluate_wasserstein(model, x_values, t_grid, a, sigma, q, ref=None):
    """W1 between the model density (truncated to [-a, a], renormalised) and the reference.

    Both densities are evaluated on the same t-grid and turned into probability weights
    (density * trapezoid cell width, normalised to sum 1) before calling
    scipy.stats.wasserstein_distance. Returns an array with one W1 value per x.
    """
    if ref is None:
        ref = reference_density_grid(x_values, t_grid, a, sigma, q)
    pred = predict_conditional_density(model, x_values, t_grid)
    cell = np.gradient(t_grid)  # ~ trapezoid weights for a uniform grid
    out = np.empty(len(x_values))
    for i in range(len(x_values)):
        wp, wr = pred[i] * cell, ref[i] * cell
        out[i] = wasserstein_distance(t_grid, t_grid, wp / wp.sum(), wr / wr.sum())
    return out


def summarise_mdn(model, dcfg, splits, t_grid, x_eval, x_rec, ref_eval=None, ref_rec=None, ref_nll=None):
    """One-row summary of a trained MDN: everything the experiment tables need.

    ref_eval / ref_rec : precomputed reference grids for x_eval / x_rec (saves time in sweeps)
    ref_nll            : reference (exact-posterior) test NLL -> reported 'excess_nll' = NLL - ref_nll
    """
    x_te, t_te = splits["test"]
    nll, nll_se = evaluate_nll(model, x_te, t_te)
    p_hat = model_branch_probability(model, x_eval)
    w1 = evaluate_wasserstein(model, x_eval, t_grid, dcfg.a, dcfg.sigma, dcfg.q, ref=ref_eval)
    br = evaluate_branch_metrics(model, x_rec, t_grid, dcfg.a, dcfg.sigma, dcfg.q, ref=ref_rec)
    mse_of_mean = float(np.mean((mixture_mean(model, x_te) - t_te) ** 2))
    return {
        "test_nll": nll, "test_nll_se": nll_se,
        "excess_nll": nll - ref_nll if ref_nll is not None else np.nan,
        "branch_err": float(np.mean(np.abs(p_hat - dcfg.q))),
        "w1": float(np.mean(w1)),
        "recovered_pos": br["recovered_pos"], "recovered_neg": br["recovered_neg"],
        "capture_pos": float(np.mean(br["capture_pos"])), "capture_neg": float(np.mean(br["capture_neg"])),
        "out_mass": float(np.mean(out_of_support_mass(model, x_eval, dcfg.a))),
        "mse_of_mean": mse_of_mean,
    }
