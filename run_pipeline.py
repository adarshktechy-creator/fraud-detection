"""End-to-end pipeline: python -m src.run_pipeline"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from . import data, drift, costs

SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)
F = data.FEATURES

base, later = data.generate(seed=SEED)
base.to_csv("data/train_window.csv", index=False)
later.to_csv("data/later_window.csv", index=False)
print(f"Train window: {len(base)} rows, fraud rate {base.is_fraud.mean():.3%}")
print(f"Later window: {len(later)} rows, fraud rate {later.is_fraud.mean():.3%}")

X_dev, X_te, y_dev, y_te, amt_dev, amt_te = train_test_split(
    base[F], base.is_fraud, base.amount, test_size=0.3, stratify=base.is_fraud, random_state=SEED)
# validation split is used ONLY for choosing the decision threshold (avoids tuning on test)
X_tr, X_val, y_tr, y_val, amt_tr, amt_val = train_test_split(
    X_dev, y_dev, amt_dev, test_size=0.25, stratify=y_dev, random_state=SEED)
y_te_v, amt_te_v = y_te.values, amt_te.values
y_val_v, amt_val_v = y_val.values, amt_val.values

# ---------- 1. Imbalance strategies (supervised) ----------
results = {}
def evaluate(name, score, y=y_te_v):
    results[name] = {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score)}
    print(f"{name:32s} PR-AUC={results[name]['pr_auc']:.4f}  ROC-AUC={results[name]['roc_auc']:.4f}")

print("\n== Model comparison (hold-out test set) ==")
prevalence = y_te_v.mean()
results["baseline_random"] = {"pr_auc": prevalence, "roc_auc": 0.5}
print(f"{'random baseline':32s} PR-AUC={prevalence:.4f}  ROC-AUC=0.5000")

gb_plain = HistGradientBoostingClassifier(random_state=SEED).fit(X_tr, y_tr)
evaluate("GBM (no imbalance handling)", gb_plain.predict_proba(X_te)[:, 1])

gb_w = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(X_tr, y_tr)
s_w = gb_w.predict_proba(X_te)[:, 1]
evaluate("GBM + class weighting", s_w)

# SMOTE applied ONLY to training data (never to test) to avoid leakage
X_sm, y_sm = SMOTE(sampling_strategy=0.1, random_state=SEED).fit_resample(X_tr, y_tr)
gb_s = HistGradientBoostingClassifier(random_state=SEED).fit(X_sm, y_sm)
s_s = gb_s.predict_proba(X_te)[:, 1]
evaluate("GBM + SMOTE (10% minority)", s_s)

# ---------- 2. Unsupervised anomaly detector ----------
iso = IsolationForest(n_estimators=300, contamination=float(y_tr.mean()), random_state=SEED, n_jobs=-1)
iso.fit(X_tr)  # no labels used
s_iso = -iso.score_samples(X_te)  # higher = more anomalous
evaluate("Isolation Forest (unsupervised)", s_iso)

# ---------- 3. PR curves ----------
plt.figure(figsize=(7, 5))
for label, key, sc in [("GBM + class weighting", "GBM + class weighting", s_w),
                       ("GBM + SMOTE", "GBM + SMOTE (10% minority)", s_s),
                       ("Isolation Forest", "Isolation Forest (unsupervised)", s_iso)]:
    p, r, _ = precision_recall_curve(y_te_v, sc)
    plt.plot(r, p, label=f"{label} (AP={results[key]['pr_auc']:.3f})")
plt.axhline(prevalence, ls="--", c="gray", label="random baseline")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/pr_curves.png", dpi=150); plt.close()

# ---------- 4. Cost-based threshold ----------
s_val = gb_w.predict_proba(X_val)[:, 1]
thrs, cst_val = costs.sweep(y_val_v, s_val, amt_val_v)
best_thr = float(thrs[int(np.argmin(cst_val))])  # chosen on validation only
# F1-optimal threshold, also chosen on validation, for comparison
pv, rv, tv = precision_recall_curve(y_val_v, s_val)
f1_thr = float(tv[np.argmax(2 * pv[:-1] * rv[:-1] / np.clip(pv[:-1] + rv[:-1], 1e-9, None))])
cst = np.array([costs.total_cost(y_te_v, s_w, amt_te_v, t) for t in thrs])  # test-set cost curve for plotting
best_i = int(np.argmin(np.abs(thrs - best_thr)))
no_model_cost = float((amt_te_v[y_te_v == 1] + costs.FN_FIXED_COST).sum())
pred = s_w >= best_thr
cost_at_f1 = costs.total_cost(y_te_v, s_w, amt_te_v, f1_thr)
cost_half = costs.total_cost(y_te_v, s_w, amt_te_v, 0.5)
thr_summary = {
    "cost_optimal_threshold_(chosen_on_validation)": best_thr,
    "test_set_cost_at_that_threshold": float(costs.total_cost(y_te_v, s_w, amt_te_v, best_thr)),
        "cost_at_default_0.5": float(cost_half),
    "cost_at_max_F1_threshold": float(cost_at_f1),
    "f1_threshold": f1_thr,
    "cost_with_no_model": no_model_cost,
    "precision_at_optimal": float(precision_score(y_te_v, pred)),
    "recall_at_optimal": float(recall_score(y_te_v, pred)),
    "f1_at_optimal": float(f1_score(y_te_v, pred)),
    "assumptions": {"FP_COST": costs.FP_COST, "FN_FIXED_COST": costs.FN_FIXED_COST,
                    "FN_cost": "transaction amount + FN_FIXED_COST"},
}
print("\n== Cost analysis ==")
print(json.dumps(thr_summary, indent=2))
plt.figure(figsize=(7, 5))
plt.plot(thrs, cst); plt.axvline(best_thr, c="r", ls="--", label=f"threshold picked on validation = {best_thr:.3f}")
plt.xscale("log"); plt.xlabel("Decision threshold (log)"); plt.ylabel("Total cost ($)")
plt.title("Test-set cost vs threshold (class-weighted GBM)"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/cost_curve.png", dpi=150); plt.close()

# ---------- 5. Drift detection + alerting ----------
print("\n== Drift detection ==")
def run_drift(label, window):
    rep = drift.drift_report(X_tr, window[F], F)
    sc_ref = gb_w.predict_proba(X_tr)[:, 1]
    sc_new = gb_w.predict_proba(window[F])[:, 1]
    spsi = drift.psi(sc_ref, sc_new)
    status, reasons = drift.alert_decision(rep, spsi)
    print(f"[{label}] status={status} score_PSI={spsi:.3f} reasons={reasons}")
    return rep, spsi, status, reasons, sc_new

# control: held-out slice of same distribution (should be OK), vs drifted later window
control = X_te.assign(is_fraud=y_te_v)
rep_c, spsi_c, st_c, rs_c, _ = run_drift("control (same distribution)", control)
rep_d, spsi_d, st_d, rs_d, sc_later = run_drift("later window (drifted)", later)
print(pd.DataFrame(rep_d).round(4).to_string(index=False))

# performance degradation on later window
pr_later = average_precision_score(later.is_fraud, sc_later)
print(f"PR-AUC on later window: {pr_later:.4f} (vs {results['GBM + class weighting']['pr_auc']:.4f} on test)")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, c in zip(ax, ["amount", "merchant_risk"]):
    a.hist(X_tr[c], bins=50, alpha=.5, density=True, label="train window", range=(0, X_tr[c].quantile(.99)))
    a.hist(later[c], bins=50, alpha=.5, density=True, label="later window", range=(0, X_tr[c].quantile(.99)))
    a.set_title(f"{c} distribution"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/drift_distributions.png", dpi=150); plt.close()

summary = {"model_metrics": results, "threshold_analysis": thr_summary,
           "drift": {"control": {"status": st_c, "score_psi": spsi_c, "reasons": rs_c},
                     "later_window": {"status": st_d, "score_psi": spsi_d, "reasons": rs_d,
                                      "per_feature": rep_d, "pr_auc_later_window": pr_later}}}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
print(f"\nSaved plots and summary.json to {OUT}/")
