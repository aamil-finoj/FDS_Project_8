"""Rohan's extension: Stress-testing MDN rare-branch recovery under reduced sample sizes (N_train in {1000, 2500, 10000}) and extreme imbalance (q in {0.90, 0.97, 0.99})."""
import sys, time
from dataclasses import replace
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p8.config import DataConfig, TrainConfig, MDNConfig
from p8.experiments import run_sweep, mean_sd_table, get_context
from p8.evaluate import predict_conditional_density
from p8.reference import reference_density_grid

RESULTS = REPO_ROOT / "artifacts" / "results"
FIGURES = REPO_ROOT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

BASE_DATA = DataConfig()
TRAIN = TrainConfig(batch_size=64, max_epochs=200)
SEEDS = [0, 1, 2, 3, 4]

# Define settings
N_LIST = [1000, 2500, 10000]
Q_LIST = [0.90, 0.97, 0.99]

settings = []
for n in N_LIST:
    for q in Q_LIST:
        settings.append((replace(BASE_DATA, n_train=n, q=q), MDNConfig(M=3, mu_init="default")))

# Also test spread init on the most challenging data sizes at q=0.99
for n in [1000, 2500]:
    settings.append((replace(BASE_DATA, n_train=n, q=0.99), MDNConfig(M=3, mu_init="spread")))

print(f"Starting sweep of {len(settings)} settings x {len(SEEDS)} seeds = {len(settings) * len(SEEDS)} runs...")
t0 = time.time()
df = run_sweep(settings, TRAIN, SEEDS, use_cache=True, progress=True)

# Add n_train column from dcfg
# In summarise_mdn, n_train wasn't stored, but we can extract it or map it from settings
# Let's inspect the df columns
# Let's enrich df with n_train and minority_n
# Since run_sweep iterates in order:
n_trains = []
for dcfg, mcfg in settings:
    for s in SEEDS:
        n_trains.append(dcfg.n_train)
df["n_train"] = n_trains
df["minority_n"] = (df["n_train"] * (1.0 - df["q"])).round().astype(int)

out_csv = RESULTS / "part5_sample_size_sweep.csv"
df.to_csv(out_csv, index=False)
print(f"Saved {out_csv} in {time.time() - t0:.1f}s")

# Display table
default_df = df[df.mu_init == "default"]
print("\n--- Summary: N_train x q (mu_init = default, mean +- sd across 5 seeds) ---")
summary_table = mean_sd_table(default_df, ["n_train", "q"], ["capture_neg", "recovered_neg", "excess_nll", "branch_err"])
print(summary_table)

# Plot 1: Recovery and Capture vs N_train for different q
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
colors = {0.90: "#1f77b4", 0.97: "#ff7f0e", 0.99: "#d62728"}

for q in Q_LIST:
    sub = default_df[default_df.q == q].groupby("n_train")
    mean_cap, std_cap = sub["capture_neg"].mean(), sub["capture_neg"].std()
    mean_rec, std_rec = sub["recovered_neg"].mean(), sub["recovered_neg"].std()
    mean_nll, std_nll = sub["excess_nll"].mean(), sub["excess_nll"].std()

    axes[0].errorbar(mean_cap.index, mean_cap, yerr=std_cap, marker="o", capsize=3, color=colors[q], label=f"q = {q} (minority {int((1-q)*100)}%)")
    axes[1].errorbar(mean_rec.index, mean_rec, yerr=std_rec, marker="s", capsize=3, color=colors[q], label=f"q = {q}")
    axes[2].errorbar(mean_nll.index, mean_nll, yerr=std_nll, marker="^", capsize=3, color=colors[q], label=f"q = {q}")

axes[0].set(xscale="log", xlabel="Training set size N_train", ylabel="Minority-branch capture", title="Minority capture vs sample size", xticks=N_LIST)
axes[0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[0].axhline(1.0, color="gray", ls="--", alpha=0.6, label="Ideal capture (1.0)")
axes[0].legend(fontsize=8)

axes[1].set(xscale="log", xlabel="Training set size N_train", ylabel="Recovery rate", title="Minority recovery rate (capture >= 0.5)", xticks=N_LIST)
axes[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[1].axhline(0.9, color="gray", ls=":", alpha=0.6, label="90% threshold")
axes[1].legend(fontsize=8)

axes[2].set(xscale="log", xlabel="Training set size N_train", ylabel="Excess NLL (nats)", title="Excess NLL vs sample size", xticks=N_LIST)
axes[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
axes[2].legend(fontsize=8)

plt.tight_layout()
fig_path = FIGURES / "07_sample_size_stress_test.png"
fig.savefig(fig_path, dpi=200)
plt.close(fig)
print(f"Saved figure to {fig_path}")

# Plot 2: Compare default vs spread initialisation on extreme low-data regime (N=1000, 2500 at q=0.99)
fig, ax = plt.subplots(figsize=(8, 4))
sub_init = df[df.q == 0.99]
g = sub_init.groupby(["n_train", "mu_init"])["capture_neg"]
means = g.mean().unstack("mu_init")
stds = g.std().unstack("mu_init")

x = np.arange(len(means))
width = 0.35
ax.bar(x - width/2, means["default"], width, yerr=stds["default"], capsize=4, label="Default init (near 0)", color="#4c72b0")
ax.bar(x + width/2, means["spread"], width, yerr=stds["spread"], capsize=4, label="Spread init (across [-a, a])", color="#55a868")
ax.axhline(0.5, color="red", ls="--", label="Recovery threshold (0.5)")
ax.axhline(1.0, color="gray", ls=":", label="Exact reference (1.0)")
ax.set_xticks(x)
ax.set_xticklabels([f"N = {n}\n(~{int(n*0.01)} minority pts)" for n in means.index])
ax.set_ylabel("Minority-branch capture")
ax.set_title("Extreme imbalance (q = 0.99): Effect of initialisation on rare-branch capture")
ax.legend()
plt.tight_layout()
fig_init_path = FIGURES / "07_sample_size_initialisation.png"
fig.savefig(fig_init_path, dpi=200)
plt.close(fig)
print(f"Saved figure to {fig_init_path}")
