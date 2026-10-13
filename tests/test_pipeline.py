import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from vehicle_pipeline import (VehicleFeatures, TARGET, chronological_split, temporal_folds,
                              baseline_predictions, convex_blend_weight, fit_bundle,
                              predict_bundle, regression_metrics)


def fixture_frame(n=120):
    rng = np.random.default_rng(4)
    dates = pd.date_range('2020-01-01', periods=n)
    mileage = rng.integers(1000, 100000, n)
    return pd.DataFrame({'VRSALEDATE': dates.strftime('%Y%m%d'), 'VRMILEAGE': mileage,
                         'Vehicle_year': 2017, 'Make': 'A', 'Model': ['m1','m2']*(n//2),
                         'Trim': 'T', 'VRLIGHTS': ['RED','GREEN']*(n//2),
                         'Vehicle_condition_overall': np.where(np.arange(n)%3, 3., np.nan),
                         TARGET: 20000-mileage*.1})


def test_baseline_alignment_with_shuffled_noncontiguous_indices():
    train = pd.DataFrame({'Make':['A','B','C'], 'Model':'M', 'Vehicle_year':2020,
                          'Trim':'T', TARGET:[100.,200.,300.]}, index=[8,2,5])
    test = train.iloc[[2,0,1]].copy()
    for name, values in baseline_predictions(train, test).items():
        if name.startswith('mean_'):
            np.testing.assert_array_equal(values, [300,100,200])


def test_cleaning_and_encoding_are_frozen_and_batch_independent():
    frame = fixture_frame()
    features = VehicleFeatures().fit(frame, np.log1p(frame[TARGET]))
    single = frame.iloc[[0]].copy()
    single['VRMILEAGE'] = 99999999
    together = pd.concat([single, frame.iloc[[1]].assign(VRMILEAGE=1e12)])
    pd.testing.assert_series_equal(features.transform(single).iloc[0],
                                   features.transform(together).iloc[0])
    assert features.transform(single).VRMILEAGE.iloc[0] <= features.mileage_cap_
    maps = features.te_maps_.copy()
    features.transform(together.assign(VRSALEAMT=1e20))
    assert maps == features.te_maps_


def test_training_te_never_uses_current_or_future_labels():
    frame = fixture_frame(40)
    y = np.log1p(frame[TARGET].to_numpy())
    first = VehicleFeatures(encoding_blocks=4).fit_transform(frame, y)
    altered = y.copy(); altered[10:] += 1000
    second = VehicleFeatures(encoding_blocks=4).fit_transform(frame, altered)
    # Initial block has no history. Second block uses only first block targets.
    np.testing.assert_allclose(first['te__Model'].iloc[:20], second['te__Model'].iloc[:20])
    assert (first['te__Model'].iloc[:10] == 0).all()
    assert not np.allclose(first['te__Model'].iloc[20:], second['te__Model'].iloc[20:])


def test_time_split_and_folds_preserve_whole_dates():
    frame = pd.concat([fixture_frame(), fixture_frame()], ignore_index=True)
    train, valid, test = chronological_split(frame.sample(frac=1, random_state=2))
    assert train.sale_date.max() < valid.sale_date.min() < test.sale_date.min()
    for past, future in temporal_folds(train, 3):
        assert train.iloc[past].sale_date.max() < train.iloc[future].sale_date.min()
    frame.loc[0, TARGET] = 0
    frame.loc[1, 'VRSALEDATE'] = 'invalid'
    a,b,c = chronological_split(frame)
    assert len(a)+len(b)+len(c) == len(frame)-2


def test_exact_convex_blend_optimum():
    a, b = np.array([0.,10.,20.]), np.array([5.,5.,5.])
    y = .25*a+.75*b
    assert convex_blend_weight(a,b,y) == pytest.approx(.25)
    assert convex_blend_weight(a,b,2*a-b) == 1
    assert convex_blend_weight(a,a,y) == .5


def test_bundle_roundtrip_feature_parity_and_test_label_isolation(tmp_path):
    train, valid, test = chronological_split(fixture_frame())
    model = RandomForestRegressor(n_estimators=4, max_depth=3, random_state=7, n_jobs=1)
    bundle = fit_bundle(train,valid,[model,model],model,n_splits=2)
    before = predict_bundle(bundle,test)
    file = tmp_path/'bundle.joblib'; joblib.dump(bundle,file)
    loaded = joblib.load(file)
    after = predict_bundle(loaded,test.assign(VRSALEAMT=1e20))
    np.testing.assert_array_equal(before,after)
    np.testing.assert_allclose(before, np.concatenate([predict_bundle(loaded,test.iloc[[i]]) for i in range(len(test))]))
    assert np.isfinite(before).all()
    with pytest.raises(ValueError,match='Missing core'):
        predict_bundle(loaded,test.drop(columns='Make'))
    assert regression_metrics(test[TARGET], before)['rmse'] >= 0


def test_scoring_marks_invalid_dates_without_dropping_or_reordering_rows():
    from scripts.score import score_frame
    train, valid, test = chronological_split(fixture_frame())
    model = RandomForestRegressor(n_estimators=2, max_depth=2, random_state=3)
    bundle = fit_bundle(train,valid,[model,model],model,n_splits=2)
    customer = test.iloc[:3].copy()
    customer.index = [9,1,7]
    customer.loc[1,'VRSALEDATE'] = None
    scored = score_frame(bundle,customer,invalid_dates='mark')
    assert scored.index.tolist() == [9,1,7]
    assert scored.ScoringStatus.tolist() == ['ok','invalid_sale_date','ok']
    assert np.isnan(scored.loc[1,'PredictedSalePrice'])
    assert np.isfinite(scored.loc[[9,7],'PredictedSalePrice']).all()
    with pytest.raises(ValueError,match='invalid sale dates'):
        score_frame(bundle,customer)
