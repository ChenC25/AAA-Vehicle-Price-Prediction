"""Chronological vehicle-price training and shared, frozen inference transforms."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit

TARGET = 'VRSALEAMT'
HIGH_CARD = ('Model', 'Trim', 'Series', 'EngineModel')
CATEGORICAL = ('Location_Region', 'VRLOCATION', 'Vehicle_botcolor', 'Vehicle_trantype',
               'Vehicle_drive', 'Vehicle_condition_drivable', 'BodyClass', 'DriveType',
               'FuelTypePrimary', 'EngineConfiguration', 'GVWR', 'PlantCountry',
               'IsEV', 'EVType', 'MarketSegment', 'Make', 'Misc_SalesChannel')
NUMERIC = ('VRMILEAGE', 'Vehicle_year', 'Vehicle_cylinders', 'Vehicle_doors',
           'Vehicle_engine', 'Vehicle_condition_overall', 'EngineHP', 'BasePrice',
           'SALEDATE_WeekofYearNumber', 'SALEDATE_MonthofYearNumber',
           'SALEDATE_Quarter', 'SALEDATE_Year', 'vehicle_age', 'mileage_per_year',
           'log_mileage', 'log_age', 'drivable_flag', 'GVWR_class',
           'Vehicle_condition_overall_missing', 'condition_imputed_from_lights',
           'EngineHP_missing', 'BasePrice_missing', 'Vehicle_cylinders_missing')


def sale_dates(frame):
    # The raw transaction date is authoritative when supplied.
    if 'VRSALEDATE' in frame:
        raw = frame['VRSALEDATE'].astype('string').str.replace(r'\.0$', '', regex=True)
        return pd.to_datetime(raw, format='%Y%m%d', errors='coerce')
    if 'sale_date' in frame:
        return pd.to_datetime(frame['sale_date'], errors='coerce')
    raise ValueError('VRSALEDATE or sale_date is required')


def chronological_split(frame):
    """Positive-price, valid-date population; never filter using target z-scores."""
    data = frame.copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors='coerce')
    data['sale_date'] = sale_dates(data)
    valid = data['sale_date'].notna() & np.isfinite(data[TARGET]) & (data[TARGET] > 0)
    data = data.loc[valid].sort_values('sale_date', kind='stable').reset_index(drop=True)
    if len(data) < 20:
        raise ValueError('Need at least 20 valid positive-price transactions')
    q90, q95 = data.sale_date.quantile([.90, .95])
    parts = [data[data.sale_date < q90],
             data[(data.sale_date >= q90) & (data.sale_date < q95)],
             data[data.sale_date >= q95]]
    if any(part.empty for part in parts):
        raise ValueError('Date cutoffs must produce three nonempty chronological periods')
    return tuple(part.reset_index(drop=True) for part in parts)


def _numeric(frame, name):
    series = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def _condition(frame):
    series = frame.get('Vehicle_condition_overall', pd.Series(np.nan, index=frame.index))
    return pd.to_numeric(series.astype('string').str.extract(r'(\d+(?:\.\d+)?)')[0], errors='coerce')


class VehicleFeatures(BaseEstimator, TransformerMixin):
    """Fit condition map/cap once; forward-cross-fit target encoding on training.

    The initial training-date block has no historical labels, so its TE is zero.
    All subsequent blocks use strictly earlier dates. At inference, only the
    final training mappings are used. Targets may be raw dollars or log1p dollars.
    """
    def __init__(self, smoothing=50.0, encoding_blocks=5):
        self.smoothing = smoothing
        self.encoding_blocks = encoding_blocks

    def _fit_map(self, X, y, name):
        values = X.get(name, pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str)
        stats = pd.DataFrame({'category': values.to_numpy(), 'target': np.asarray(y)})
        agg = stats.groupby('category').target.agg(['count', 'mean'])
        mean = float(np.mean(y))
        mapping = ((agg['count'] * agg['mean'] + self.smoothing * mean) /
                   (agg['count'] + self.smoothing)).to_dict()
        return mean, mapping

    def fit(self, X, y):
        if self.smoothing <= 0 or self.encoding_blocks < 2:
            raise ValueError('Positive smoothing and at least two encoding blocks required')
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError('Targets must be finite and aligned positionally')
        condition = _condition(X)
        self.condition_mean_ = float(condition.mean()) if condition.notna().any() else 3.0
        lights = X.get('VRLIGHTS', pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str).str.upper().str.strip()
        agg = pd.DataFrame({'lights': lights.to_numpy(), 'condition': condition.to_numpy()}).groupby('lights').condition.agg(['count', 'mean'])
        self.condition_map_ = ((agg['count'] * agg['mean'] + 100 * self.condition_mean_) /
                               (agg['count'] + 100)).to_dict()
        mileage = _numeric(X, 'VRMILEAGE').clip(lower=0)
        self.mileage_cap_ = float(mileage.quantile(.9995)) if mileage.notna().any() else np.inf
        self.te_maps_ = {name: self._fit_map(X, y, name) for name in HIGH_CARD}
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for name in NUMERIC:
            out[name] = _numeric(X, name)
        dates = sale_dates(X)
        out['SALEDATE_Year'] = dates.dt.year.astype(float)
        out['SALEDATE_MonthofYearNumber'] = dates.dt.month.astype(float)
        out['SALEDATE_Quarter'] = dates.dt.quarter.astype(float)
        out['SALEDATE_WeekofYearNumber'] = dates.dt.isocalendar().week.astype(float)
        condition = _condition(X)
        missing = condition.isna()
        lights = X.get('VRLIGHTS', pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str).str.upper().str.strip()
        proxy = lights.map(self.condition_map_)
        out['Vehicle_condition_overall_missing'] = missing.astype(float)
        out['condition_imputed_from_lights'] = (missing & proxy.notna()).astype(float)
        out['Vehicle_condition_overall'] = condition.fillna(proxy).fillna(self.condition_mean_)
        out['Vehicle_year'] = out.Vehicle_year.clip(1900, 2100)
        out['vehicle_age'] = (out.SALEDATE_Year - out.Vehicle_year).clip(lower=0)
        out['VRMILEAGE'] = out.VRMILEAGE.clip(0, self.mileage_cap_)
        out['mileage_per_year'] = out.VRMILEAGE / out.vehicle_age.replace(0, .5)
        out['log_mileage'] = np.log1p(out.VRMILEAGE)
        out['log_age'] = np.log1p(out.vehicle_age)
        drive = X.get('Vehicle_condition_drivable', pd.Series('Unknown', index=X.index))
        out['drivable_flag'] = drive.astype(str).str.upper().eq('Y').astype(float)
        gvwr = X.get('GVWR', pd.Series('Unknown', index=X.index))
        out['GVWR_class'] = pd.to_numeric(gvwr.astype(str).str.extract(r'Class\s*(\d+)')[0], errors='coerce')
        for name in ['EngineHP', 'BasePrice', 'Vehicle_cylinders']:
            out[name + '_missing'] = out[name].isna().astype(float)
        for name in CATEGORICAL:
            values = X.get(name, pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str)
            out[name] = values
        for name, (mean, mapping) in self.te_maps_.items():
            values = X.get(name, pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str)
            out['te__' + name] = values.map(mapping).fillna(mean)
        numbers = list(NUMERIC) + ['te__' + name for name in HIGH_CARD]
        out[numbers] = out[numbers].replace([np.inf, -np.inf], np.nan).astype('float32')
        return out

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        out = self.transform(X)
        dates = sale_dates(X)
        if dates.isna().any():
            raise ValueError('Training feature fitting requires valid sale dates')
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 2:
            raise ValueError('Need at least two distinct dates for forward target encoding')
        blocks = np.array_split(unique_dates, min(self.encoding_blocks, len(unique_dates)))
        y = np.asarray(y)
        for block in blocks:
            current = dates.isin(block).to_numpy()
            past = (dates < block[0]).to_numpy()
            for name in HIGH_CARD:
                if past.any():
                    mean, mapping = self._fit_map(X.iloc[np.flatnonzero(past)], y[past], name)
                    values = X.get(name, pd.Series('Unknown', index=X.index)).fillna('Unknown').astype(str)
                    encoded = values.map(mapping).fillna(mean).to_numpy()
                    out.loc[current, 'te__' + name] = encoded[current].astype('float32')
                else:
                    out.loc[current, 'te__' + name] = 0.0
        return out


def model_pipeline(estimator):
    nums = list(NUMERIC) + ['te__' + name for name in HIGH_CARD]
    preprocess = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), nums),
        ('category', OneHotEncoder(handle_unknown='ignore'), list(CATEGORICAL)),
    ], sparse_threshold=1.0)
    return Pipeline([('features', VehicleFeatures()), ('preprocess', preprocess), ('model', estimator)])


def temporal_folds(X, n_splits):
    """Split unique dates, so identical timestamps never straddle fold boundaries."""
    dates = sale_dates(X)
    unique = np.sort(dates.unique())
    if len(unique) <= n_splits:
        raise ValueError('Not enough distinct dates for temporal stacking')
    for past, future in TimeSeriesSplit(n_splits=n_splits).split(unique):
        yield np.flatnonzero(dates.isin(unique[past])), np.flatnonzero(dates.isin(unique[future]))


def fit_temporal_stack(X, y, base_models, n_splits=3):
    log_y = np.log1p(np.asarray(y, dtype=float))
    oof = np.full((len(X), len(base_models)), np.nan)
    for train, valid in temporal_folds(X, n_splits):
        for j, estimator in enumerate(base_models):
            # Feature statistics and encodings are independently refit in every fold.
            pipeline = model_pipeline(clone(estimator))
            pipeline.fit(X.iloc[train], log_y[train])
            oof[valid, j] = pipeline.predict(X.iloc[valid])
    covered = np.isfinite(oof).all(axis=1)
    if covered.sum() < 2:
        raise ValueError('Too few forward out-of-fold samples')
    meta = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    meta.fit(oof[covered], log_y[covered])
    fitted = [model_pipeline(clone(estimator)).fit(X, log_y) for estimator in base_models]
    return fitted, meta


def predict_stack(X, bases, meta):
    log_predictions = np.column_stack([model.predict(X) for model in bases])
    result = np.expm1(meta.predict(log_predictions))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite stack predictions')
    return np.maximum(result, 0)


def convex_blend_weight(first, second, target):
    """Exact two-model constrained LS: w in [0,1], other weight = 1-w."""
    delta = np.asarray(first) - np.asarray(second)
    denominator = float(delta @ delta)
    return float(np.clip(delta @ (np.asarray(target) - second) / denominator, 0, 1)) if denominator else .5


def baseline_predictions(train, test):
    y = train[TARGET].to_numpy()
    fallback = float(np.median(y))
    result = {'global_median': np.full(len(test), fallback)}
    for names in [('Make', 'Model', 'Vehicle_year'), ('Make', 'Model', 'Vehicle_year', 'Trim')]:
        # Reset the whole frame, not only y; attach the labels positionally.
        reference = train[list(names)].reset_index(drop=True).copy()
        reference[TARGET] = y
        means = reference.groupby(list(names), dropna=False)[TARGET].mean()
        keys = test[list(names)].reset_index(drop=True)
        merged = keys.merge(means.rename('prediction'), left_on=list(names), right_index=True, how='left', sort=False)
        result['mean_' + '_'.join(names)] = merged.prediction.fillna(fallback).to_numpy()
    return result


def regression_metrics(target, predictions):
    y, p = np.asarray(target, float), np.asarray(predictions, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any():
        raise ValueError('Metrics require aligned finite predictions and positive targets')
    residual = p - y
    denominator = np.sum((y - y.mean())**2)
    return {'mae': float(np.abs(residual).mean()), 'rmse': float(np.sqrt(np.mean(residual**2))),
            'r2': float(1 - np.sum(residual**2) / denominator) if denominator else None,
            'mape': float(np.mean(np.abs(residual) / y))}


def fit_bundle(train, valid, base_models, raw_model, n_splits=3, calibrate=True):
    X, y = train.drop(columns=TARGET), train[TARGET].to_numpy()
    bases, meta = fit_temporal_stack(X, y, base_models, n_splits)
    raw = model_pipeline(clone(raw_model)).fit(X, y)
    VX, vy = valid.drop(columns=TARGET), valid[TARGET].to_numpy()
    first, second = predict_stack(VX, bases, meta), raw.predict(VX)
    weight = convex_blend_weight(first, second, vy)
    blended = weight * first + (1-weight) * second
    iso = IsotonicRegression(out_of_bounds='clip').fit(blended, vy) if calibrate else None
    return {'schema_version': 3, 'bases': bases, 'meta': meta, 'raw': raw,
            'blend_weight': weight, 'isotonic': iso,
            'protocol': 'forward-fold stack; train-fitted cleaning; forward training TE; validation-only blend/calibration',
            'required_columns': ['VRMILEAGE', 'Vehicle_year', 'Make', 'Model', 'Trim'],
            'training_columns': list(X.columns)}


def predict_bundle(bundle, frame, calibrate=True):
    missing = set(bundle['required_columns']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing core input columns: {sorted(missing)}')
    if sale_dates(frame).isna().any():
        raise ValueError('All scoring rows need a valid sale context date')
    p1 = predict_stack(frame, bundle['bases'], bundle['meta'])
    p2 = bundle['raw'].predict(frame)
    p = bundle['blend_weight'] * p1 + (1-bundle['blend_weight']) * p2
    if calibrate and bundle['isotonic'] is not None:
        p = bundle['isotonic'].predict(p)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions')
    return np.maximum(p, 0)
