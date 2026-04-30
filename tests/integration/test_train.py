import json
import os

import pytest

from credit_risk.pipeline.train import train as _train

_EXPECTED_KEYS = {
    'best_cv_pr_auc', 'val_pr_auc', 'optimal_threshold',
    'val_f2_at_threshold', 'best_params', 'model_path', 'threshold_path',
}


def _write_splits(tmp_path, df):
    paths = {
        'train':     str(tmp_path / 'train.csv'),
        'val':       str(tmp_path / 'val.csv'),
        'test':      str(tmp_path / 'test.csv'),
        'model':     str(tmp_path / 'anomaly_pipeline_xgboost.pkl'),
        'threshold': str(tmp_path / 'anomaly_pipeline_xgboost.threshold'),
    }
    n = len(df)
    t, v = int(n * 0.7), int(n * 0.85)
    df.iloc[:t].to_csv(paths['train'], index=False)
    df.iloc[t:v].to_csv(paths['val'], index=False)
    df.iloc[v:].to_csv(paths['test'], index=False)
    return paths


def test_train_creates_pkl(tmp_path, synthetic_df):
    paths = _write_splits(tmp_path, synthetic_df)
    _train(paths=paths, model_name='xgboost', fast_mode=True)
    assert os.path.exists(paths['model'])


def test_train_creates_threshold_file(tmp_path, synthetic_df):
    paths = _write_splits(tmp_path, synthetic_df)
    _train(paths=paths, model_name='xgboost', fast_mode=True)
    assert os.path.exists(paths['threshold'])


def test_train_threshold_json_is_valid(tmp_path, synthetic_df):
    paths = _write_splits(tmp_path, synthetic_df)
    _train(paths=paths, model_name='xgboost', fast_mode=True)
    with open(paths['threshold']) as fh:
        data = json.load(fh)
    assert 'threshold' in data
    assert 0.0 <= data['threshold'] <= 1.0


def test_train_returns_expected_keys(tmp_path, synthetic_df):
    paths = _write_splits(tmp_path, synthetic_df)
    result = _train(paths=paths, model_name='xgboost', fast_mode=True)
    assert _EXPECTED_KEYS <= set(result.keys())


def test_train_val_pr_auc_in_range(tmp_path, synthetic_df):
    paths = _write_splits(tmp_path, synthetic_df)
    result = _train(paths=paths, model_name='xgboost', fast_mode=True)
    assert 0.0 <= result['val_pr_auc'] <= 1.0


def test_train_missing_split_raises(tmp_path):
    paths = {
        'train':     str(tmp_path / 'missing_train.csv'),
        'val':       str(tmp_path / 'missing_val.csv'),
        'test':      str(tmp_path / 'missing_test.csv'),
        'model':     str(tmp_path / 'model.pkl'),
        'threshold': str(tmp_path / 'model.threshold'),
    }
    with pytest.raises(FileNotFoundError):
        _train(paths=paths, model_name='xgboost', fast_mode=True)


def test_train_rf_creates_pkl(tmp_path, synthetic_df):
    paths = {
        'train':     str(tmp_path / 'train.csv'),
        'val':       str(tmp_path / 'val.csv'),
        'test':      str(tmp_path / 'test.csv'),
        'model':     str(tmp_path / 'anomaly_pipeline_rf.pkl'),
        'threshold': str(tmp_path / 'anomaly_pipeline_rf.threshold'),
    }
    n = len(synthetic_df)
    t, v = int(n * 0.7), int(n * 0.85)
    synthetic_df.iloc[:t].to_csv(paths['train'], index=False)
    synthetic_df.iloc[t:v].to_csv(paths['val'],   index=False)
    synthetic_df.iloc[v:].to_csv(paths['test'],   index=False)
    _train(paths=paths, model_name='rf', fast_mode=True)
    assert os.path.exists(paths['model'])


def test_train_no_fraud_cases_raises(tmp_path, synthetic_df):
    paths = {
        'train':     str(tmp_path / 'train.csv'),
        'val':       str(tmp_path / 'val.csv'),
        'test':      str(tmp_path / 'test.csv'),
        'model':     str(tmp_path / 'model.pkl'),
        'threshold': str(tmp_path / 'model.threshold'),
    }
    df_no_fraud = synthetic_df.copy()
    df_no_fraud['isFraud'] = 0
    n = len(df_no_fraud)
    t, v = int(n * 0.7), int(n * 0.85)
    df_no_fraud.iloc[:t].to_csv(paths['train'], index=False)
    df_no_fraud.iloc[t:v].to_csv(paths['val'],   index=False)
    df_no_fraud.iloc[v:].to_csv(paths['test'],   index=False)
    with pytest.raises(ValueError, match="No fraud cases"):
        _train(paths=paths, model_name='xgboost', fast_mode=True)
