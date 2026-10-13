# AAA Vehicle Price Prediction

Vehicle sale-price modeling developed from America's Auto Auction transaction data. The project predicts `VRSALEAMT` using mileage, vehicle specifications, condition, location and sale-time context.

## Corrected pipeline for a future training run

**Model status:** Full-data retraining is unavailable for this update. The existing trained `.joblib` files are unchanged. The corrected workflow below has been checked with unit tests and a 3,000-row smoke run; it is not a replacement full-data model.

The corrected entrypoints are `scripts/train.py` and `scripts/score.py`. The notebooks in `Prediction_Final/` call these scripts rather than duplicating training or scoring logic.

```text
Valid positive-price transactions → chronological train / validation / test
      ↓
Training-fitted condition mapping, mileage cap and imputation
      ↓
Prior-date target encoding + low-cardinality one-hot encoding
      ↓
XGBoost and CatBoost (log target)
      ↓
Forward-fold out-of-fold predictions → Ridge meta-model
      + raw-price LightGBM
      ↓
Validation-only convex blend and optional isotonic post-processing
      ↓
Separate later-period test evaluation → frozen v3 scoring bundle
```

Each stacking fold independently fits its entire preprocessing/model pipeline. Identical sale dates remain in the same temporal fold. Training target encodings use only earlier date blocks; the first block receives a fixed zero encoding because there are no historical labels. Inference uses frozen full-training mappings, with training-mean fallback for unseen categories.

The canonical CatBoost model consumes the same numeric/one-hot/target-encoded features as XGBoost; this path does not claim native categorical handling. Model settings are conservative reproducible defaults, not the historical tuned settings.

## Setup

Run from the repository root using Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

On macOS, XGBoost/LightGBM may require an OpenMP runtime (`brew install libomp`). Do not silently enable duplicate OpenMP runtimes to conceal a runtime conflict.

## Train and evaluate

First run a small end-to-end check spanning the source file:

```bash
python scripts/train.py --sample-size 3000 --trees 20 --folds 2 --output artifacts/smoke
```

This randomly samples across the whole file with bounded-memory priority sampling; it is not the first 3,000 transactions. It produces **smoke-test results**, not production-quality metrics.

When full-data training becomes available (not performed in this update):

```bash
python scripts/train.py --output artifacts
```

The full run performs multiple forward-fold fits over millions of rows and can take substantial time/memory. Choose model settings and whether to use `--no-calibration` using development/validation evidence, before examining final test results.

Outputs:

- `vehicle_bundle_v3.joblib`: fitted feature transforms, base models, meta-model, blend and optional isotonic map.
- `evaluation.json`: run type, configuration, split windows and one consistent test metric table.
- `test_predictions.csv`: aligned targets and predictions, including simple baselines and component models.

The date split uses training before the 90th-percentile sale date, validation until the 95th-percentile date, and test afterward. It retains whole dates, so row proportions are approximate. Invalid dates and nonfinite/nonpositive prices are excluded under an explicit population policy. It does not remove holdout examples using target-based z-scores.

## Score with a future v3 model

Use this path only after training a new v3 bundle. It cannot be applied to the existing legacy bundles. The smoke bundle can validate this command but is not suitable for real pricing decisions:

```bash
python scripts/score.py --data "Test Customer Data.xlsx" --bundle artifacts/vehicle_bundle_v3.joblib --output artifacts/scored_customers.csv --invalid-dates mark
```

Use `--output artifacts/scored_customers.xlsx` for Excel output. The supplied customer file contains 17 missing sale dates. `--invalid-dates mark` preserves these rows with an empty prediction and `ScoringStatus=invalid_sale_date`; the default strict mode stops with a clear error. Dates are not silently invented. Cleaning statistics and category mappings are never refit on customer data; single-row and batch scoring use the same transforms. Load serialized bundles only from trusted sources.

The two `.joblib` files already committed under `Prediction_Final/` are **historical artifacts**, not compatible v3 models. They have not been retrained or relabeled as corrected models. The corrected scoring path requires a newly trained v3 bundle. For historical investigation, retain the original notebook workflow under `archive/` and its corresponding historical dependency environment; this update does not certify its saved metrics or refit its preprocessing.

## Historical model records

For traceability, the previous README recorded the following values. They belong to historical artifacts/experiments, not the corrected v3 workflow, and have not been revalidated by this update:

| Historical report | R² | MAE | RMSE |
|---|---:|---:|---:|
| Time-holdout final ensemble | 0.9456 | 1,508 | 2,805 |
| Time-holdout tuned XGBoost | 0.9436 | 1,514 | 2,858 |
| External-data blend | 0.9096 | 1,356 | 2,517 |

Do not combine these figures with different saved notebook outputs. The quoted final-versus-XGBoost RMSE difference is about 1.85% within that historical table; it does not substantiate the former 77% baseline claim.

## Results and validation status

Previous README and notebook numbers came from multiple experimental paths. Some early post-processing fitted on the same labels subsequently reported as test results, and group-price baseline construction had a row-index alignment risk. The historical “approximately 77% RMSE improvement” is therefore withdrawn as a verified headline.

The revised workflow reports results from one run using distinct test arrays. Unit tests and a small real-data smoke run validate implementation behavior; a full-data rerun is required before publishing replacement performance claims. See [evaluation notes](docs/EVALUATION.md) and [validation record](docs/VALIDATION.md).

MAE/RMSE are measured in sale-price units; R² is relative to the evaluation set's target variance; MAPE uses strictly positive prices without an arbitrary denominator floor. Log-price models optimize a different objective from dollar squared error. Isotonic post-processing can overfit or clip high-end extrapolation; compare it with the uncalibrated blend on the final report without selecting a winner repeatedly from test results.

## Scope and limitations

- Sale-time context is required. For a pre-auction prediction use case, the date and all other inputs must be known at the actual decision time; this is not an established long-horizon forecasting model.
- External-data independence, repeated-vehicle overlap and feature-availability timing still need domain verification.
- The pipeline is a reproducible offline training/scoring workflow. It does not establish a live production deployment or measured business impact.
- A clean chronological split reduces leakage but does not eliminate population selection, measurement bias or market drift.
- Generated models and customer prediction files are ignored by Git. Existing data files are unchanged.

## Repository structure

```text
vehicle_pipeline.py          Shared feature, stack, blend, baseline and scoring logic
scripts/train.py             Training + one consistent test report
scripts/score.py             Frozen-bundle scoring
Prediction_Final/*.ipynb      Thin notebook entrypoints (no stale saved outputs)
tests/test_pipeline.py       Leakage, alignment, parity and persistence checks
docs/EVALUATION.md            Changes from exploratory evaluation
docs/VALIDATION.md            Checks actually run; full-run limitations
archive/                     Original notebooks, including historical outputs
```

The original exploratory notebooks and result outputs are preserved under `archive/` for reference (not maintained scoring entrypoints), and in [Git history](https://github.com/ChenC25/AAA-Vehicle-Price-Prediction/tree/697614a2f61814446dbdef5c409c1f293786a3b3).
