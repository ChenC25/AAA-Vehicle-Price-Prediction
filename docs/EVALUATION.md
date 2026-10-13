# Evaluation corrections

The original exploratory workflow mixed several model versions and reused notebook variables. This revision establishes one maintained code path instead of attaching corrected labels to old saved outputs.

| Issue | Correction |
|---|---|
| Group baseline categories and targets could align by different pandas indices | Reset the complete key frame and attach targets positionally; preserve test-row order |
| Early blend/isotonic fitting used the labels called test | Fit post-processing only on the middle chronological validation period |
| Later reporting reused pre-refit prediction variables | Derive all report arrays from the same bundle and same final test frame |
| Full-training target means were applied to training rows | Training encoding uses strictly previous date blocks; inference maps are fitted on training only |
| Ordinary KFold stack with preprocessing outside its folds | Manually build forward-time OOF predictions, independently fitting the full pipeline for each fold; omit uncovered initial rows from meta-model fitting |
| Cleaner calculated maps/caps separately on scoring datasets | Store fitted condition maps and mileage cap in the pipeline and reuse unchanged |
| Label-based global outlier filtering changed holdout population | Remove target-z-score filtering; document valid-date/positive-price population instead |
| NNLS weights normalized afterward were called optimal convex blending | Exact constrained two-component least-squares solution with weights in [0,1] summing to one |
| README metrics and notebook outputs came from different runs | Withdraw unverifiable headline improvements; require one new report with configuration and split windows |
| Training/scoring notebooks duplicated incompatible cleaning logic | All notebook entrypoints use shared scripts; legacy bundles require retraining |

## Interpretation

These changes do not prove that the historical model was useless, nor do they establish a new full-data score. They make the revised result reproducible and easier to audit.

The first date block's target encoding is fixed at zero, not derived from a future prior. Later training blocks use historical category statistics. At inference the entire earlier training period is available. This creates a deliberate cold-history distribution difference; validate block count/smoothing using development data.

Feature cleaning within each outer stacking fold is fitted only to that fold's training examples. Numeric imputation and label-independent condition estimation use the whole earlier training fold; they are not claimed to be online-updated per individual transaction.

XGBoost/CatBoost train on `log1p(price)`; Ridge combines their log-scale OOF predictions. The stack's prediction is then inverse-transformed. LightGBM trains on dollar prices. The convex blend and isotonic map fit on validation prices, then freeze. Corrected default parameters are not an attempt to recreate the historical tuned ensemble.

Unit tests cover structural invariants. A small sampled run verifies the installed model libraries and complete artifact/report/scoring flow. Full-data quality, confidence intervals, subgroup errors, external-set independence and production performance remain separate work.

## No full-data retraining in this update

Full retraining is unavailable. Existing trained binary artifacts remain unchanged, and corrected training/scoring code is provided for the next retraining opportunity. This update repairs the maintained code and documentation; it does not retrofit learned encodings, stack folds, or calibration into the old binaries. Smoke artifacts are validation-only and are not published as pricing models.
