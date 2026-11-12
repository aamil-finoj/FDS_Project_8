"""Exact (numerical) posterior p(t | x): the reference every model is graded against.

Posterior up to a constant:  log p~(t | x) = -(x - t^2)^2 / (2 sigma^2) + log prior(t).
For the asymmetric prior we write prior(t) = w(t) / (2a) with w = 2q (t >= 0) or 2(1-q) (t < 0);
the constant 1/(2a) and the Gaussian constant cancel in the normalisation.
"""
import numpy as np
from scipy import integrate


def _prior_weight(t, q):
    """Relative prior weight w(t): 2q on t >= 0, 2(1-q) on t < 0 (w = 1 everywhere when q = 0.5)."""
    t = np.asarray(t, dtype=float)
    return np.where(t >= 0, 2.0 * q, 2.0 * (1.0 - q))


def log_unnormalized_posterior(t, x, a, sigma, q=0.5):
    """log p~(t | x) = -(x - t^2)^2 / (2 sigma^2) + log w(t) on [-a, a]; -inf outside."""
    t = np.asarray(t, dtype=float)
    val = -((x - t**2) ** 2) / (2.0 * sigma**2) + np.log(_prior_weight(t, q))
    return np.where(np.abs(t) <= a, val, -np.inf)


def max_log_likelihood(x, a, sigma):
    """Max over t in [-a, a] of -(x - t^2)^2/(2 sigma^2).

    Attained where t^2 is as close as possible to x, i.e. t^2 = clip(x, 0, a^2).
    Subtracting this before exponentiating keeps everything in a safe range.
    """
    t2 = np.clip(x, 0.0, a**2)
    return -((x - t2) ** 2) / (2.0 * sigma**2)


def log_normalizing_constant(x, a, sigma, q=0.5, epsabs=1e-12, epsrel=1e-10):
    """log Z(x), Z(x) = int_{-a}^{a} exp(log p~(u | x)) du, by scipy.integrate.quad.

    We integrate each half [-a, 0] and [0, a] separately (the prior weight jumps at 0)
    and tell quad where the peak sits (u = sqrt(clip(x, 0, a^2))) so it never misses
    a narrow mode. Values are scaled by exp(-max log-likelihood) for stability.
    """
    m = max_log_likelihood(x, a, sigma)
    peak = float(np.sqrt(np.clip(x, 0.0, a**2)))

    def f(u):
        return np.exp(-((x - u * u) ** 2) / (2.0 * sigma**2) - m)

    pts_pos = [peak] if 0.0 < peak < a else None
    pts_neg = [-peak] if 0.0 < peak < a else None
    half_pos, _ = integrate.quad(f, 0.0, a, points=pts_pos, epsabs=epsabs, epsrel=epsrel, limit=200)
    half_neg, _ = integrate.quad(f, -a, 0.0, points=pts_neg, epsabs=epsabs, epsrel=epsrel, limit=200)
    z_scaled = 2.0 * q * half_pos + 2.0 * (1.0 - q) * half_neg
    return np.log(z_scaled) + m


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
    """P(t > 0 | x) = int_0^a p(t | x) dt, computed by quadrature (theory says it equals q)."""
    m = max_log_likelihood(x, a, sigma)
    peak = float(np.sqrt(np.clip(x, 0.0, a**2)))
    pts = [peak] if 0.0 < peak < a else None
    f = lambda u: np.exp(-((x - u * u) ** 2) / (2.0 * sigma**2) - m)
    half_pos, _ = integrate.quad(f, 0.0, a, points=pts, epsabs=1e-12, epsrel=1e-10, limit=200)
    pos = 2.0 * q * half_pos
    return pos / np.exp(log_normalizing_constant(x, a, sigma, q) - m)


def reference_mean(x, a, sigma, q=0.5):
    """E[t | x] by quadrature - the population quantity the MSE network estimates."""
    m = max_log_likelihood(x, a, sigma)
    peak = float(np.sqrt(np.clip(x, 0.0, a**2)))
    pts = [peak] if 0.0 < peak < a else None
    f = lambda u: u * np.exp(-((x - u * u) ** 2) / (2.0 * sigma**2) - m)
    pos, _ = integrate.quad(f, 0.0, a, points=pts, epsabs=1e-12, epsrel=1e-10, limit=200)
    neg, _ = integrate.quad(f, -a, 0.0, points=[-peak] if pts else None, epsabs=1e-12, epsrel=1e-10, limit=200)
    num = 2.0 * q * pos + 2.0 * (1.0 - q) * neg
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


def dip_ratio(x, sigma):
    """p(0 | x) / p(+sqrt(x) | x) = exp(-x^2 / (2 sigma^2)) for 0 < x <= a^2 (symmetric prior).

    Close to 1 -> the two branches are 'merged' (barely any valley between them);
    close to 0 -> two clearly separated branches.
    """
    return np.exp(-(x**2) / (2.0 * sigma**2)) if x > 0 else 1.0


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
