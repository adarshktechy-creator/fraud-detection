# Write-up: Fraud Detection with Drift Monitoring

## 1. Problem and data
Binary fraud classification on ~260k synthetic card transactions (8 features: amount, hour, merchant risk,
distance from home, transactions in last 24h, 30-day average amount, card age, online flag). Fraud rate is
~0.4% (roughly 1 in 240). A "training window" (200k rows) is used for fitting/evaluation and a **later window**
(60k rows) simulates production data after covariate shift (amounts inflated 1.6x, more online purchases,
riskier merchants, longer distances). Data is synthetic because the pipeline must run anywhere without downloads;
the README explains how to swap in the Kaggle dataset.

Split: 70% dev / 30% test (stratified); dev is split again into 75% train / 25% validation. Validation is used
only to choose the decision threshold, so the test set is never tuned on.

## 2. Handling class imbalance
| Strategy | PR-AUC (test) | ROC-AUC (test) |
|---|---|---|
| Random baseline | 0.004 | 0.500 |
| GBM, no handling | 0.552 | 0.824 |
| **GBM + class weighting** | **0.839** | 0.996 |
| GBM + SMOTE (minority raised to 10%) | 0.826 | 0.992 |
| Isolation Forest (unsupervised) | 0.635 | 0.986 |

**Choice: class weighting.** It matched or beat SMOTE here, adds no synthetic rows (SMOTE interpolates between
fraud points and can create unrealistic transactions, especially for categorical/binary features such as
`is_online`), is cheaper to train, and keeps the original data distribution so predicted probabilities stay easier
to monitor for drift. SMOTE was applied only to training data, never to validation/test, to avoid leakage. Note that
the gap between the two (0.839 vs 0.826) comes from a single split and should not be over-interpreted.

## 3. Supervised vs unsupervised
The gradient-boosted classifier clearly beats Isolation Forest on PR-AUC (0.839 vs 0.635) because it uses labels
and learns exactly which patterns indicate fraud. Isolation Forest never sees labels; it only flags
"unusual" transactions, so it also surfaces rare-but-legitimate behaviour (lower precision). Its value is
operational: it works with no labels, and can catch *new* fraud types the supervised model has never seen. In a
real system I would run both: the supervised score as the primary decision and the anomaly score as a secondary
signal or feature.

## 4. Why PR-AUC and not ROC-AUC alone
With 0.4% positives, the false positive *rate* denominator is huge (~99.6% of rows), so even a model that blocks
many legitimate transactions keeps a tiny FPR and a high ROC-AUC. Here Isolation Forest gets ROC-AUC 0.986
while its PR-AUC is only 0.635, and the unweighted GBM reaches 0.82 ROC-AUC but just 0.55 PR-AUC. PR-AUC uses
precision (fraction of alerts that are real fraud) and recall, ignores true negatives, and its random baseline
equals the fraud rate (0.004), so it reflects the work analysts actually face.

## 5. Drift detection and alerting
For each feature I compute the **Population Stability Index (PSI)** (quantile bins from the training window) and a
**two-sample KS test** against the later window. I also compute PSI on the model's output score (a label-free
proxy for model health, since labels arrive late in fraud).

Alert logic (`src/drift.py`): PSI < 0.10 OK; 0.10-0.25 warning; ≥ 0.25 critical. **RETRAIN** is raised if the
score PSI ≥ 0.25 or at least 25% of features are critically drifted; otherwise WARNING if anything is moderately
drifted.

Results: a control slice drawn from the same distribution gives **OK** (score PSI 0.000). The drifted window gives
**RETRAIN** (score PSI 0.319). Strongest feature drift: `merchant_risk` (PSI 2.67), then `amount` (PSI 0.21);
`dist_from_home_km` is small (0.06). PR-AUC on the later window dropped from 0.839 to 0.819, a real but modest
degradation, which shows why monitoring input and score distributions matters: they signalled the shift before
any labels were available.

Limitation: PSI with quantile bins is unreliable for binary features. `is_online` shifted noticeably (KS = 0.16)
but its PSI is ~0 because the quantile bins collapse. The KS test caught it, which is one reason to use both.

## 6. Cost analysis and threshold choice
Assumptions (editable in `src/costs.py`): a false positive costs $5 (manual review plus customer friction); a missed
fraud costs the transaction amount plus a $15 chargeback/operations fee. Because missing fraud costs more than
a false alarm, the optimal threshold is well below the F1-optimal one.

| Policy | Total cost on test set |
|---|---|
| No model (miss all fraud) | ₹4,996,102 |
| Threshold 0.926 (max F1 on validation) | ₹1,089,126 |
| Default threshold 0.5 | ₹633,622 |
| **Threshold 0.318 (min cost on validation)** | ₹644,578 |

At 0.318 the model catches ~92% of fraud (recall 0.92) with precision ~0.36, so roughly 2 of 3 alerts are false
positives. That is acceptable at these costs. Using the model cuts cost by ~87% compared with doing nothing.
Honest caveat: the validation-chosen threshold (0.318) and the default 0.5 have almost the same test cost, so
the cost curve is flat near the optimum; the real lesson is that accuracy-style thresholds are much worse
than cost-based ones. In production the threshold should be re-tuned whenever costs change or the model is retrained.

## 7. Limitations and next steps
- Synthetic data: results show the method works, not real-world performance. Run it on the Kaggle dataset too.
- Single train/test split and one seed; use time-based cross-validation and confidence intervals.
- Drift check on score distribution is label-free, but concept drift (fraud tactics changing) needs delayed-label
  monitoring of PR-AUC.
- Possible extensions: autoencoder detector, probability calibration, SHAP explanations, scheduled monitoring job
  on AWS (e.g. SageMaker Model Monitor) triggering retraining when the alert fires.
