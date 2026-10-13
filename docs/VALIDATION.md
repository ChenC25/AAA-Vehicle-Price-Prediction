# Validation record — October 13, 2026

## Checks performed

- `python -m pytest -q -W error::FutureWarning`: **7 passed**.
- Regression coverage: misaligned/noncontiguous baseline indices; frozen cleaning and batch independence; training TE excluding current/future targets; whole-date split/fold separation; exact convex blend; bundle persistence and test-label isolation; explicit invalid-date scoring with row order preserved.
- Real-data end-to-end check: `python scripts/train.py --sample-size 3000 --trees 20 --folds 2 --output artifacts/smoke` completed with XGBoost, CatBoost, LightGBM, forward stacking, validation blend/calibration, report export and bundle persistence.
- The sampled split contained 2,700 training, 150 validation and 150 test examples. [Machine-readable smoke report](smoke-evaluation.json) records data/source hashes and package versions.
- Customer scoring with the smoke bundle completed on all **30,978** input rows: **30,961** finite predictions and **17** rows explicitly marked `invalid_sale_date` with empty predictions. No rows were silently dropped, and missing dates were not guessed.
- New notebook entrypoints contain no saved model-result outputs. Original notebooks and saved outputs are preserved in `archive/`.

## What was not performed

Full-data retraining is unavailable and **was not run**. The two historical trained `.joblib` files were not modified, replaced or claimed to have corrected learned parameters. Smoke scores and predictions establish that the code path executes; they do not establish production-quality pricing, revised full-data accuracy or business impact.

A future full-data run must use the corrected pipeline and produce its own metric report before replacement accuracy claims are made. External-set independence, duplicate-vehicle overlap and decision-time feature availability still require domain verification.
