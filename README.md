# MAC 312 – Project 8: Inverse problems with multiple valid solutions

We observe `x = t² + ε` (with `t ~ Unif[-a, a]`, `ε ~ N(0, σ²)`) and try to recover `t`.
Because squaring loses the sign, most observations have **two** valid answers (`±√x`).
We show that a squared-error network can only predict the conditional mean (which sits *between*
the two answers). A mixture density network (MDN) can recover the whole conditional density. We then
measure how reliably it recovers both branches when the prior is imbalanced and the noise grows.

**Main deliverable:** [`Project8_Inverse_Problems.ipynb`](Project8_Inverse_Problems.ipynb). It contains
all maths, code, experiments, figures and conclusions, and runs top to bottom from a clean kernel.

## Repository layout

| Path | What it is |
|---|---|
| `Project8_Inverse_Problems.ipynb` | Master notebook (assignment Parts 1–5 + extension) |
| `src/p8/` | Reusable code imported by the notebook |
| `src/p8/reference.py` | Exact posterior `p(t│x)` via `scipy.integrate.quad` (the ground truth) |
| `src/p8/models.py` | PyTorch MSE network and MDN (single Gaussian = MDN with `M=1`) |
| `src/p8/training.py` | Training loop: Adam, early stopping on validation, run caching |
| `src/p8/evaluate.py` | NLL, sampling, branch probabilities, branch recovery, Wasserstein-1 |
| `src/p8/experiments.py` | Multi-seed sweep helpers |
| `tools/build_notebook.py` | Generates the notebook from source (edit cells here, then rebuild) |
| `tests/sanity_check.py` | Quick standalone checks of the reference and the MDN |
| `figures/` | All figures, saved by the notebook |
| `artifacts/results/` | CSV tables of every experiment + the settings used |

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on macOS/Linux)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # or plain `pip install torch` for CPU
pip install -r requirements.txt
python -m ipykernel install --user --name scaf-project8
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 Project8_Inverse_Problems.ipynb
```

A full clean run trains about 200 small networks. It took about 1.5 h on an RTX 3050 laptop GPU.
Trained runs are cached in `artifacts/cache/` (keyed by a hash of every setting), so later re-runs
take a few minutes. Delete that folder to force retraining from scratch.

To change a notebook cell, edit `tools/build_notebook.py`, run `python tools/build_notebook.py`,
and re-execute the notebook.

## Reproducibility
* The dataset is fixed by `data_seed=2026`, and model initialisation uses seeds 0–4. Python, NumPy and PyTorch are all seeded.
* Train/validation/test = 10 000 / 2 000 / 2 000. Model selection and early stopping use validation only. The test set is used only for reporting.
* Every setting is written to `artifacts/results/settings.json`.
* GPU training is not bit-for-bit deterministic, so re-running without the cache can change the last digits of some metrics.
