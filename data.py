"""Synthetic credit-card transaction generator (~0.4% fraud) with a simulated drift window."""
import numpy as np
import pandas as pd

FEATURES = ["amount", "hour", "merchant_risk", "dist_from_home_km",
            "txn_last_24h", "avg_amount_30d", "card_age_days", "is_online"]


def _make(n, fraud_rate, rng, drift=False):
    y = (rng.random(n) < fraud_rate).astype(int)
    f = y == 1
    nf = ~f
    df = pd.DataFrame(index=range(n))
    # legit vs fraud behave differently, with overlap so the task is not trivial
    df["amount"] = np.where(f, rng.lognormal(5.0, 1.0, n), rng.lognormal(3.6, 1.0, n))
    hour = np.where(f, rng.normal(2.5, 4.0, n), rng.normal(14, 4.5, n))
    df["hour"] = np.mod(hour, 24)
    df["merchant_risk"] = np.where(f, rng.beta(4, 2.5, n), rng.beta(2, 6, n))
    df["dist_from_home_km"] = np.where(f, rng.exponential(120, n), rng.exponential(25, n))
    df["txn_last_24h"] = np.where(f, rng.poisson(5, n), rng.poisson(1.8, n))
    df["avg_amount_30d"] = rng.lognormal(3.7, 0.6, n)
    df["card_age_days"] = np.where(f, rng.exponential(250, n), rng.exponential(900, n))
    df["is_online"] = np.where(f, rng.random(n) < 0.7, rng.random(n) < 0.35).astype(int)
    if drift:
        # simulated covariate shift: spending inflation, more online shopping, riskier merchants
        df["amount"] *= 1.6
        df["is_online"] = np.where(rng.random(n) < 0.25, 1, df["is_online"])
        df["merchant_risk"] = np.clip(df["merchant_risk"] + 0.12, 0, 1)
        df["dist_from_home_km"] *= 1.3
    df["is_fraud"] = y
    return df


def generate(n_train=200_000, n_later=60_000, fraud_rate=0.004, seed=42):
    rng = np.random.default_rng(seed)
    base = _make(n_train, fraud_rate, rng)
    later = _make(n_later, fraud_rate, rng, drift=True)
    return base, later
