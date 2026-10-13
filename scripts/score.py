"""Score with a newly trained v3 bundle; never refit cleaning on customer data."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import pandas as pd
import numpy as np
from vehicle_pipeline import predict_bundle, sale_dates


def score_frame(bundle, frame, invalid_dates='error'):
    if invalid_dates not in {'error', 'mark'}:
        raise ValueError('invalid_dates must be error or mark')
    missing = set(bundle['required_columns']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing core input columns: {sorted(missing)}')
    result = frame.copy()
    valid = sale_dates(frame).notna().to_numpy()
    if not valid.all() and invalid_dates == 'error':
        raise ValueError(f'{int((~valid).sum())} rows have invalid sale dates; use --invalid-dates mark to retain them as unscored')
    predictions = np.full(len(frame), np.nan)
    if valid.any():
        predictions[valid] = predict_bundle(bundle, frame.iloc[np.flatnonzero(valid)])
    result['PredictedSalePrice'] = predictions
    result['ScoringStatus'] = np.where(valid, 'ok', 'invalid_sale_date')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/vehicle_bundle_v3.joblib'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/scored_customers.csv'))
    parser.add_argument('--invalid-dates', choices=['error', 'mark'], default='error',
                        help='mark preserves invalid rows with an empty prediction and explicit status')
    args = parser.parse_args()
    frame = pd.read_parquet(args.data) if args.data.suffix == '.parquet' else pd.read_excel(args.data)
    bundle = joblib.load(args.bundle)  # Load only a bundle from a trusted source.
    if bundle.get('schema_version') != 3:
        raise ValueError('Expected v3 bundle; historical bundles require retraining')
    frame = score_frame(bundle, frame, args.invalid_dates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == '.xlsx':
        frame.to_excel(args.output, index=False)
    else:
        frame.to_csv(args.output, index=False)
    print(f'Exported {len(frame):,} rows ({int(frame.ScoringStatus.eq("ok").sum()):,} scored, '
          f'{int(frame.ScoringStatus.ne("ok").sum()):,} marked) -> {args.output}')


if __name__ == '__main__':
    main()
