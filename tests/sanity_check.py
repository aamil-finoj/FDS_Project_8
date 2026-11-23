"""Quick sanity checks of the reference posterior and the MDN plumbing (run: python tests/sanity_check.py)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, torch
from p8.reference import *
from p8.config import DataConfig, TrainConfig, MDNConfig
from p8.data import make_splits
from p8.training import train_mdn, train_mse_model, DEVICE
from p8.evaluate import *

a, s = 2.0, 0.15
grid = np.linspace(-a, a, 4001)
for x in [-0.4, 0.05, 1.0, 3.9, 4.6]:
    for q in [0.5, 0.9]:
        d = reference_density(grid, x, a, s, q)
        print(f"x={x:5} q={q} int={np.trapezoid(d, grid):.6f} P+={reference_branch_probability(x,a,s,q):.6f} "
              f"E[t|x]={reference_mean(x,a,s,q):+.5f} gridvsquad={np.abs(reference_density_grid([x],grid,a,s,q)[0]-d).max():.2e}")
print("symmetry max |p(t)-p(-t)|:", np.abs(reference_density(grid,1.0,a,s)-reference_density(-grid,1.0,a,s)).max())
for q in [0.5, 0.9]:  # conditional median (what an absolute-error network estimates): 0 if symmetric, on the + branch if q = 0.9
    for x in [1.0, 3.0]:
        med = reference_median(x, a, s, q)
        print(f"median x={x} q={q}: {med:+.4f}  F(median)={reference_cdf(med, x, a, s, q):.6f}  band={reference_median_band(x, a, s, q)}")
print("device", DEVICE)
d = DataConfig(); tc = TrainConfig(max_epochs=60)
sp = make_splits(d)
for dev_note in ["gpu"]:
    t0=time.time(); m,h,_ = train_mdn(d, tc, MDNConfig(M=3), seed=0, splits=sp, use_cache=False, verbose=True); print("time", time.time()-t0)
pi, mu, var = mdn_params(m, sp["test"][0])
print("pi sum err", np.abs(pi.sum(1)-1).max(), "min var", var.min(), "finite", np.isfinite(var).all())
print("nll", evaluate_nll(m, *sp["test"]))
print("P+ at x=1:", model_branch_probability(m, [1.0]), "out mass", out_of_support_mass(m,[1.0],a))
xs = sp["test"][0][:200]
print("W1 mean", evaluate_wasserstein(m, xs, grid, a, s, 0.5).mean())
print(evaluate_branch_metrics(m, recovery_x_grid(s,a), grid, a, s, 0.5)["recovered_neg"])
sm = sample_conditional(m, [1.0], 5000); print("sample mean |t|", np.abs(sm).mean(), "resid sd", (1.0-sm**2).std())
