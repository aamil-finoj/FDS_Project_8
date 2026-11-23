"""Exact (numerical) posterior p(t | x): the reference every model is graded against.

Posterior up to a constant:  log p~(t | x) = -(x - t^2)^2 / (2 sigma^2) + log prior(t).
For the asymmetric prior we write prior(t) = w(t) / (2a) with w = 2q (t >= 0) or 2(1-q) (t < 0);
the constant 1/(2a) and the Gaussian constant cancel in the normalisation.
"""
from functools import lru_cache

import numpy as np
from scipy import integrate


def _validate(a, sigma, q):
    """Reject parameters for which the posterior is not defined (NaN also fails these tests)."""
    if not (a > 0 and sigma > 0 and 0.0 <= q <= 1.0):
        raise ValueError(f"need a > 0, sigma > 0 and 0 <= q <= 1; got a={a}, sigma={sigma}, q={q}")


def _prior_weight(t, q):
    """Relative prior weight w(t): 2q on t >= 0, 2(1-q) on t < 0 (w = 1 everywhere when q = 0.5)."""
    t = np.asarray(t, dtype=float)
    return np.where(t >= 0, 2.0 * q, 2.0 * (1.0 - q))


def log_unnormalized_posterior(t, x, a, sigma, q=0.5):
    """log p~(t | x) = -(x - t^2)^2 / (2 sigma^2) + log w(t) on [-a, a]; -inf outside.

    With q = 0 or 1 one side has prior weight 0, so its log density is -inf (not a warning).
    """
    _validate(a, sigma, q)
    t = np.asarray(t, dtype=float)
    with np.errstate(divide="ignore"):
        val = -((x - t**2) ** 2) / (2.0 * sigma**2) + np.log(_prior_weight(t, q))
    return np.where(np.abs(t) <= a, val, -np.inf)


def max_log_likelihood(x, a, sigma):
    """Max over t in [-a, a] of -(x - t^2)^2/(2 sigma^2).

    Attained where t^2 is as close as possible to x, i.e. t^2 = clip(x, 0, a^2).
    Subtracting this before exponentiating keeps everything in a safe range.
    """
    t2 = np.clip(x, 0.0, a**2)
    return -((x - t2) ** 2) / (2.0 * sigma**2)


def _half_integral(x, a, sigma, side, power, m, epsabs=1e-12, epsrel=1e-10):
    """int over [0, a] (side=+1) or [-a, 0] (side=-1) of u**power * exp(-(x-u^2)^2/(2 sigma^2) - m) du.

    The prior weight jumps at 0, so each half is integrated on its own. quad is told where the peak
    (u = side * sqrt(clip(x, 0, a^2))) sits so it cannot miss a narrow mode. m = max_log_likelihood
    keeps the integrand at most 1, so nothing under- or overflows.
    """
    peak = float(np.sqrt(np.clip(x, 0.0, a**2)))
    pts = [side * peak] if 0.0 < peak < a else None
    lo, hi = (0.0, a) if side > 0 else (-a, 0.0)

    def f(u):
        return u**power * np.exp(-((x - u * u) ** 2) / (2.0 * sigma**2) - m)

    value, _ = integrate.quad(f, lo, hi, points=pts, epsabs=epsabs, epsrel=epsrel, limit=200)
    return value


@lru_cache(maxsize=200_000)
def _halves(x, a, sigma, epsabs, epsrel):
    """Cached (m, int_0^a f, int_-a^0 f) with f = exp(-(x-u^2)^2/(2 sigma^2) - m).

    These do not depend on q (the prior only multiplies each half by a weight), so every q, and every
    function that needs Z(x) or P(t > 0 | x), reuses the same two integrals for a given x.
    """
    m = float(max_log_likelihood(x, a, sigma))
    return (m,
            _half_integral(x, a, sigma, +1, 0, m, epsabs, epsrel),
            _half_integral(x, a, sigma, -1, 0, m, epsabs, epsrel))


def log_normalizing_constant(x, a, sigma, q=0.5, epsabs=1e-12, epsrel=1e-10):
    """log Z(x), Z(x) = int_{-a}^{a} exp(log p~(u | x)) du, by scipy.integrate.quad (scalar x).

    We integrate each half [-a, 0] and [0, a] separately (the prior weight jumps at 0)
    and tell quad where the peak sits so it never misses a narrow mode. Values are scaled by
    exp(-max log-likelihood) for stability. The two half-integrals are cached per x.
    """
    _validate(a, sigma, q)
    m, pos, neg = _halves(float(x), float(a), float(sigma), float(epsabs), float(epsrel))
    return float(np.log(2.0 * q * pos + 2.0 * (1.0 - q) * neg) + m)


def reference_density(t, x, a, sigma, q=0.5):
    """Normalised p(t | x) at arbitrary points t (scalar x), using the quadrature Z(x)."""
    log_z = log_normalizing_constant(x, a, sigma, q)
    return np.exp(log_unnormalized_posterior(t, x, a, sigma, q) - log_z)


def reference_log_density(t, x, a, sigma, q=0.5):
    """log p(t | x) for paired arrays (t_i, x_i) - used for the reference (Bayes-optimal) NLL."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    log_z = np.array([log_normalizing_constant(xi, a, sigma, q) for xi in x])
    return log_unnormalized_posterior(t, x, a, sigma, q) - log_z


def reference_density_grid(x_values, t_grid, a, sigma, q=0.5):
    """Reference density for many x on a common t-grid, shape (len(x_values), len(t_grid)).

    Each row is normalised with the quadrature constant Z(x) (not the grid sum), so the
    values are the true density even where the grid is too coarse for a perfect Riemann
    sum (e.g. at the prior jump at t = 0 when q != 0.5, or a sharp boundary peak).
    """
    x_values = np.atleast_1d(np.asarray(x_values, dtype=float))
    log_z = np.array([log_normalizing_constant(xi, a, sigma, q) for xi in x_values])
    logp = log_unnormalized_posterior(t_grid[None, :], x_values[:, None], a, sigma, q)
    return np.exp(logp - log_z[:, None])


def reference_branch_probability(x, a, sigma, q=0.5):
    """P(t > 0 | x) = int_0^a p(t | x) dt, computed by quadrature (theory says it equals q).

    Both halves are integrated numerically (not by assuming the symmetry that makes the answer q),
    so this is a genuine check of the quadrature rather than a restatement of the theory.
    """
    _validate(a, sigma, q)
    _, pos, neg = _halves(float(x), float(a), float(sigma), 1e-12, 1e-10)
    return 2.0 * q * pos / (2.0 * q * pos + 2.0 * (1.0 - q) * neg)


def reference_mean(x, a, sigma, q=0.5):
    """E[t | x] by quadrature - the population quantity the MSE network estimates."""
    _validate(a, sigma, q)
    m = max_log_likelihood(x, a, sigma)
    num = (2.0 * q * _half_integral(x, a, sigma, +1, 1, m)
           + 2.0 * (1.0 - q) * _half_integral(x, a, sigma, -1, 1, m))
    return num / np.exp(log_normalizing_constant(x, a, sigma, q) - m)


def reference_modes(x, a, sigma):
    """Analytic mode locations of p(t | x) (any q - the prior only rescales each half).

    d/dt log p = 2 t (x - t^2) / sigma^2, so:
      x <= 0       -> single mode at t = 0 (no real noiseless inverse)
      0 < x <= a^2 -> modes at t = +-sqrt(x), and t = 0 is a local minimum
      x > a^2      -> modes pinned to the support boundary t = +-a
    """
    if x <= 0:
        return np.array([0.0])
    r = min(np.sqrt(x), a)
    return np.array([-r, r])


def dip_ratio(x, sigma, a=None):
    """p(0 | x) / p(mode | x) for a symmetric prior.

    For 0 < x <= a^2 the modes are at +-sqrt(x), and the ratio is exp(-x^2 / (2 sigma^2)).
    Close to 1 -> the two branches are 'merged' (barely any valley between them);
    close to 0 -> two clearly separated branches.
    Beyond the support (x > a^2) the modes are pinned to +-a, so the ratio is p(0)/p(a) =
    exp(-(x^2 - (x - a^2)^2) / (2 sigma^2)); pass `a` to get that case right (without `a` the
    formula for the in-support case is used, as before).
    """
    if x <= 0:
        return 1.0
    if a is not None and x > a**2:
        return float(np.exp(-(x**2 - (x - a**2) ** 2) / (2.0 * sigma**2)))
    return np.exp(-(x**2) / (2.0 * sigma**2))


def classify_region(x, a, sigma, sep_k=2.0):
    """Label an observation by its inverse-branch structure.

    'no real inverse' : x <= 0
    'merged'          : 0 < x < sep_k * sigma  (dip ratio > exp(-sep_k^2/2) ~ 0.135 for k=2)
    'two branches'    : sep_k * sigma <= x <= a^2
    'beyond support'  : x > a^2 (modes stuck at +-a)
    """
    if x <= 0:
        return "no real inverse"
    if x < sep_k * sigma:
        return "merged"
    if x <= a**2:
        return "two branches"
    return "beyond support"
