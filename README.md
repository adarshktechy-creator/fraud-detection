# Fraud Detection under Extreme Class Imbalance

End-to-end pipeline on a synthetic credit-card transaction dataset (~0.4% fraud):

1. Handles class imbalance (class weighting vs SMOTE vs nothing) and justifies the choice
2. Trains a supervised model (Gradient Boosting) **and** an unsupervised anomaly detector (Isolation Forest) and compares them
3. Evaluates with **Precision-Recall AUC** (plus ROC-AUC for contrast)
4. Drift detection (PSI + KS test) between a training window and a simulated later window
5. Alert logic: flags **RETRAIN / WARNING / OK**
6. Cost-based threshold selection (false positive cost vs false negative cost)

The full write-up is in [`WRITEUP.md`](WRITEUP.md).

## Setup

```bash
git clone <your-repo-url>
cd fraud-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Reproduce the results

```bash
python -m src.run_pipeline
```

This regenerates the data (fixed seed `42`), trains all models, and writes to `results/`:

| File | Contents |
|---|---|
| `summary.json` | all metrics, threshold analysis, drift report |
| `pr_curves.png` | Precision-Recall curves of the three main models |
| `cost_curve.png` | total cost vs decision threshold |
| `drift_distributions.png` | train vs later-window feature histograms |

## Re-running the evaluation

Everything is deterministic. To evaluate under different assumptions edit and re-run:

- Costs: `FP_COST`, `FN_FIXED_COST` in `src/costs.py`
- Drift alert bounds: `PSI_WARN`, `PSI_CRIT`, `FRAC_FEATURES_CRIT` in `src/drift.py`
- Fraud rate / data size / seed: `data.generate(...)` arguments, or `SEED` in `src/run_pipeline.py`

then run `python -m src.run_pipeline` again.

## Using the real Kaggle dataset instead

Download `creditcard.csv` from Kaggle ("Credit Card Fraud Detection"), and in `run_pipeline.py` replace the
`data.generate(...)` call with a loader that sorts by `Time`, uses the first ~70% as the training window and the
rest as the later window, and sets `F` to the `V1..V28, Amount` columns (target column `Class`).

## Project structure

```
src/data.py          synthetic data + simulated drift
src/drift.py         PSI, KS test, alert decision logic
src/costs.py         cost model and threshold sweep
src/run_pipeline.py  full pipeline
results/             generated outputs
```
