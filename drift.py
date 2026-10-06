"""Drift detection (PSI + KS) and retraining alert logic."""
import numpy as np
from scipy.stats import ks_2samp


def psi(expected, actual, bins=10):
    """Population Stability Index using quantile bins from the reference window."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(ref, new, features):
    rows = []
    for c in features:
        p = psi(ref[c].values, new[c].values)
        ks = ks_2samp(ref[c].values, new[c].values)
        rows.append({"feature": c, "psi": p, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue})
    return rows


PSI_WARN, PSI_CRIT = 0.10, 0.25   # standard industry rules of thumb
FRAC_FEATURES_CRIT = 0.25          # retrain if >=25% of features are critically drifted


def alert_decision(report, score_psi):
    """Return ('OK'|'WARNING'|'RETRAIN', reasons)."""
    reasons = []
    n = len(report)
    crit = [r["feature"] for r in report if r["psi"] >= PSI_CRIT]
    warn = [r["feature"] for r in report if PSI_WARN <= r["psi"] < PSI_CRIT]
    if score_psi >= PSI_CRIT:
        reasons.append(f"model score PSI={score_psi:.3f} >= {PSI_CRIT}")
    if len(crit) / n >= FRAC_FEATURES_CRIT:
        reasons.append(f"{len(crit)}/{n} features critically drifted: {crit}")
    if reasons:
        return "RETRAIN", reasons
    if warn or crit or score_psi >= PSI_WARN:
        return "WARNING", [f"moderate drift in {warn + crit}, score PSI={score_psi:.3f}"]
    return "OK", ["no significant drift"]
