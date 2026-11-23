"""Standalone tests for Part 1: the exact posterior in src/p8/reference.py.

Run:  python tests/test_reference.py        (needs only numpy and scipy; no torch, no GPU; a few seconds)

Every check compares the quadrature code against something independent (a fine-grid midpoint
rule, an analytic identity, or a direct formula). Numpy warnings are turned into errors, so a silent
overflow, log(0) or divide-by-zero is a failure.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

import p8.reference as R

warnings.simplefilter("error")

A = 2.0
N = 400_000
H = 2 * A / N
# Cell centres of 400 000 equal cells. Their boundary falls exactly on t = 0, so the prior's jump
# (q != 0.5) never lies inside a cell and the midpoint rule keeps its second-order accuracy.
FINE = -A + (np.arange(N) + 0.5) * H
FAILURES = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def integ(y):
    """Independent integral over [-a, a]: the midpoint rule on the cell centres above."""
    return H * np.sum(y)


def grid_logZ(x, sigma, q):
    """Independent normaliser on the 400 001-point grid, in log space."""
    lp = R.log_unnormalized_posterior(FINE, x, A, sigma, q)
    m = lp.max()
    return m + np.log(integ(np.exp(lp - m)))


def main():
    xs = [-0.4, 0.0, 0.05, 0.1, 0.3, 1.0, 2.5, 3.9, 4.0, 4.6]
    for sigma in (0.05, 0.15, 0.5):
        for q in (0.5, 0.9):
            # 1. normaliser agrees with the independent grid integral. The tolerance is set by the grid, not the code:
            #    at x beyond the support (4.6) the density is cut off sharply, and the grid error there falls
            #    by 64x when the grid is refined 8x (second order), while the code agrees to 1e-16 elsewhere.
            err = max(abs(R.log_normalizing_constant(x, A, sigma, q) - grid_logZ(x, sigma, q)) for x in xs)
            check(f"log Z vs fine-grid midpoint rule (sigma={sigma}, q={q})", err < 1e-5, f"max err {err:.1e}")

            # 2. density integrates to 1; branch probability equals q; conditional mean matches a grid mean
            e_int = e_p = e_mean = 0.0
            for x in xs:
                d = R.reference_density(FINE, x, A, sigma, q)
                e_int = max(e_int, abs(integ(d) - 1.0))
                e_p = max(e_p, abs(R.reference_branch_probability(x, A, sigma, q) - q))
                e_mean = max(e_mean, abs(R.reference_mean(x, A, sigma, q) - integ(FINE * d)))
            check(f"density integrates to 1 (sigma={sigma}, q={q})", e_int < 1e-5, f"max err {e_int:.1e}")
            check(f"P(t>0|x) = q exactly (sigma={sigma}, q={q})", e_p < 1e-9, f"max err {e_p:.1e}")
            check(f"E[t|x] vs grid mean (sigma={sigma}, q={q})", e_mean < 1e-5, f"max err {e_mean:.1e}")

    # 3. symmetric prior: p(t | x) = p(-t | x), and E[t | x] = 0
    d = R.reference_density(FINE, 1.0, A, 0.15)
    check("symmetry p(t)=p(-t) for q=0.5", np.abs(d - d[::-1]).max() < 1e-12)
    check("E[t|x]=0 for q=0.5", abs(R.reference_mean(1.0, A, 0.15)) < 1e-10)

    # 4. the grid version and the paired log-density version agree with the scalar version
    xg = np.array([-0.4, 0.3, 1.0, 3.9])
    grid = R.reference_density_grid(xg, FINE[::1000], A, 0.15, 0.9)
    one = np.array([R.reference_density(FINE[::1000], x, A, 0.15, 0.9) for x in xg])
    check("reference_density_grid == reference_density row by row", np.abs(grid - one).max() < 1e-12)
    t_pairs = np.array([-1.0, 0.5, 1.2, 1.9]); x_pairs = np.array([1.0, 0.25, 1.44, 3.6])
    ld = R.reference_log_density(t_pairs, x_pairs, A, 0.15, 0.9)
    ref = np.array([np.log(R.reference_density(t, x, A, 0.15, 0.9)) for t, x in zip(t_pairs, x_pairs)])
    check("reference_log_density == log reference_density", np.abs(ld - ref).max() < 1e-10)

    # 5. quadrature tolerance does not change the answer
    tol = max(abs(R.log_normalizing_constant(x, A, 0.15, 0.9, epsabs=1e-6, epsrel=1e-6)
                  - R.log_normalizing_constant(x, A, 0.15, 0.9)) for x in xs)
    check("loose vs tight quadrature tolerance", tol < 1e-5, f"max diff {tol:.1e}")

    # 6. analytic modes match the numerical maxima, where the two branches are resolved
    ok = True
    for x in (0.5, 1.0, 2.5, 3.9):
        d = R.reference_density(FINE, x, A, 0.15)
        mid = N // 2
        num = {-FINE[:mid][d[:mid].argmax()], FINE[mid:][d[mid:].argmax()]}
        ok &= np.allclose(sorted(num), np.abs(R.reference_modes(x, A, 0.15)), atol=2e-4)
    check("analytic modes = +-sqrt(x) = numerical maxima", ok)
    check("modes: x<=0 -> single mode at 0", list(R.reference_modes(-0.3, A, 0.15)) == [0.0])
    check("modes pinned to +-a beyond support", list(R.reference_modes(4.6, A, 0.15)) == [-2.0, 2.0])

    # 7. region labels and the dip ratio
    labels = [R.classify_region(x, A, 0.15) for x in (-0.1, 0.0, 0.2, 0.3, 1.0, 4.0, 4.01)]
    check("classify_region labels and boundaries",
          labels == ["no real inverse", "no real inverse", "merged", "two branches", "two branches", "two branches", "beyond support"],
          str(labels))
    for x in (0.1, 1.0, 3.9):  # inside the support: p(0)/p(sqrt x)
        lp = R.log_unnormalized_posterior(np.array([0.0, np.sqrt(x)]), x, A, 0.15)
        check(f"dip_ratio inside support (x={x})", abs(np.log(R.dip_ratio(x, 0.15)) - (lp[0] - lp[1])) < 1e-9)
    for x in (4.2, 4.6):  # beyond support the modes sit at +-a, so the ratio is p(0)/p(a)
        lp = R.log_unnormalized_posterior(np.array([0.0, A]), x, A, 0.15)
        check(f"dip_ratio beyond support (x={x})", abs(np.log(R.dip_ratio(x, 0.15, a=A)) - (lp[0] - lp[1])) < 1e-9)

    # 8. edge cases stay finite, warning-free and normalised (warnings are errors in this file).
    #    x=10 with sigma=0.05 is a spike about 1e-4 wide at the boundary, close to the 1e-5 grid spacing,
    #    so the grid integral is only good to about 4e-4 there (it falls 64x per 8x finer grid).
    for sigma in (0.05, 0.5):
        for q in (0.0, 0.5, 1.0):
            for x in (-5.0, 0.0, 1e-9, 4.0, 10.0):
                lz = R.log_normalizing_constant(x, A, sigma, q)
                d = R.reference_density(FINE, x, A, sigma, q)
                check(f"edge case finite and normalised (x={x}, sigma={sigma}, q={q})",
                      np.isfinite(lz) and abs(integ(d) - 1.0) < 1e-3, f"int={integ(d):.6f}")

    # 9. bad parameters are rejected with a clear error
    for kw in (dict(a=-1.0, sigma=0.15, q=0.5), dict(a=A, sigma=0.0, q=0.5), dict(a=A, sigma=0.15, q=1.5)):
        try:
            R.log_normalizing_constant(1.0, **kw)
            check(f"rejects bad parameters {kw}", False)
        except ValueError:
            check(f"rejects bad parameters {kw}", True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
