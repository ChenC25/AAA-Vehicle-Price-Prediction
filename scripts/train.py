"""Run from repository root: python scripts/train.py [--sample-size 3000 --trees 20]."""
import argparse
import hashlib
import importlib.metadata
import platform
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from vehicle_pipeline import (TARGET, chronological_split, fit_bundle, predict_bundle,
                              predict_stack, baseline_predictions, regression_metrics)


def read_sample(path, sample_size=None):
    if sample_size is None:
        return pd.read_parquet(path)
    # Reservoir-style random priority sample spanning the full file, bounded memory.
    import pyarrow.parquet as pq
    rng = np.random.default_rng(42)
    kept, priorities = None, None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50000):
        frame = batch.to_pandas()
        keys = rng.random(len(frame))
        if kept is not None:
            frame = pd.concat([kept, frame], ignore_index=True)
            keys = np.concatenate([priorities, keys])
        count = min(sample_size, len(frame))
        selected = np.argpartition(keys, count-1)[:count]
        kept, priorities = frame.iloc[selected].reset_index(drop=True), keys[selected]
    if kept is None:
        raise ValueError('Empty input dataset')
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('Customer Database.parquet'))
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    parser.add_argument('--sample-size', type=int)
    parser.add_argument('--trees', type=int, default=300)
    parser.add_argument('--folds', type=int, default=3)
    parser.add_argument('--no-calibration', action='store_true')
    args = parser.parse_args()
    if args.trees < 1 or args.folds < 2 or (args.sample_size is not None and args.sample_size < 100):
        parser.error('Positive trees, at least two folds and sample-size >= 100 required')
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    train, valid, test = chronological_split(read_sample(args.data, args.sample_size))
    models = [XGBRegressor(n_estimators=args.trees, max_depth=6, learning_rate=.05,
                          tree_method='hist', n_jobs=2, random_state=42),
              CatBoostRegressor(iterations=args.trees, depth=6, learning_rate=.05,
                                thread_count=2, random_seed=42, verbose=False,
                                allow_writing_files=False)]
    raw = LGBMRegressor(n_estimators=args.trees, num_leaves=31, learning_rate=.05,
                        n_jobs=2, random_state=42, verbosity=-1)
    bundle = fit_bundle(train, valid, models, raw, args.folds, not args.no_calibration)
    X, y = test.drop(columns=TARGET), test[TARGET].to_numpy()
    predictions = baseline_predictions(train, test)
    predictions.update({
        'xgboost_log': np.maximum(np.expm1(bundle['bases'][0].predict(X)), 0),
        'catboost_log': np.maximum(np.expm1(bundle['bases'][1].predict(X)), 0),
        'temporal_stack': predict_stack(X, bundle['bases'], bundle['meta']),
        'lightgbm_raw': bundle['raw'].predict(X),
        'blend': predict_bundle(bundle, X, calibrate=False),
        'blend_isotonic': predict_bundle(bundle, X),
    })
    metrics = {name: regression_metrics(y, p) for name, p in predictions.items()}
    baseline_rmse = min(value['rmse'] for name, value in metrics.items()
                        if name.startswith(('global_', 'mean_')))
    with args.data.open('rb') as source:
        data_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    source_hash = hashlib.sha256((Path(__file__).read_bytes() +
                                 (Path(__file__).resolve().parents[1] / 'vehicle_pipeline.py').read_bytes())).hexdigest()
    report = {'data_sha256': data_hash, 'source_sha256': source_hash,
              'python': platform.python_version(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'catboost', 'lightgbm']},
              'run_type': 'sample_smoke' if args.sample_size else 'full_dataset',
              'data': str(args.data), 'data_size_bytes': args.data.stat().st_size,
              'trees': args.trees, 'folds': args.folds, 'seed': 42,
              'protocol': bundle['protocol'],
              'split': {name: {'rows': len(part), 'start': str(part.sale_date.min()),
                                'end': str(part.sale_date.max())}
                        for name, part in [('train', train), ('validation', valid), ('test', test)]},
              'test_metrics': metrics,
              'final_vs_best_baseline_rmse_reduction': 1-metrics['blend_isotonic']['rmse']/baseline_rmse}
    bundle['evaluation'] = report
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output / 'vehicle_bundle_v3.joblib', compress=3)
    (args.output / 'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    pd.DataFrame({'sale_date': test.sale_date, 'actual': y, **predictions}).to_csv(
        args.output / 'test_predictions.csv', index=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
