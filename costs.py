"""Cost-sensitive threshold selection."""
import numpy as np

FP_COST = 5.0          # review cost + customer friction per blocked legit transaction ($)
FN_FIXED_COST = 15.0   # chargeback/ops fee per missed fraud ($); plus the transaction amount


def total_cost(y, score, amount, thr):
    pred = score >= thr
    fp = (pred) & (y == 0)
    fn = (~pred) & (y == 1)
    return fp.sum() * FP_COST + (amount[fn] + FN_FIXED_COST).sum()


def sweep(y, score, amount, n=400):
    thrs = np.unique(np.quantile(score, np.linspace(0.5, 0.99999, n)))
    costs = np.array([total_cost(y, score, amount, t) for t in thrs])
    return thrs, costs
