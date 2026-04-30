import copy

import numpy as np
import pandas as pd
import pytest

import credit_risk.models.factory as _factory_mod
from credit_risk.config.features import categorical_features, numerical_features


def _synth(n=300, seed=42):
    rng = np.random.default_rng(seed)
    data = {col: rng.uniform(0, 1000, n) for col in numerical_features}
    for col in categorical_features:
        data[col] = rng.choice(['A', 'B', 'C'], n)
    data['isFraud'] = (rng.random(n) < 0.15).astype(int)
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def _cpu_xgboost(monkeypatch):
    """Swap device='cuda' -> 'cpu' for every integration test (no GPU required)."""
    _real = _factory_mod.load_config

    def _patched(model_name, config_dir=None):
        cfg = copy.deepcopy(_real(model_name, config_dir))
        if model_name == 'xgboost':
            cfg['model']['params']['device'] = 'cpu'
        return cfg

    monkeypatch.setattr(_factory_mod, 'load_config', _patched)


@pytest.fixture
def synthetic_df():
    return _synth()


@pytest.fixture(scope='module')
def trained_paths_module(tmp_path_factory):
    """Train a minimal XGBoost model once per test module; shares the pkl across tests."""
    _real = _factory_mod.load_config

    def _patched(model_name, config_dir=None):
        cfg = copy.deepcopy(_real(model_name, config_dir))
        if model_name == 'xgboost':
            cfg['model']['params']['device'] = 'cpu'
        return cfg

    _factory_mod.load_config = _patched
    try:
        df = _synth()
        tmp = tmp_path_factory.mktemp('trained_module')
        paths = {
            'train':     str(tmp / 'train.csv'),
            'val':       str(tmp / 'val.csv'),
            'test':      str(tmp / 'test.csv'),
            'model':     str(tmp / 'anomaly_pipeline_xgboost.pkl'),
            'threshold': str(tmp / 'anomaly_pipeline_xgboost.threshold'),
        }
        n = len(df)
        t, v = int(n * 0.7), int(n * 0.85)
        df.iloc[:t].to_csv(paths['train'], index=False)
        df.iloc[t:v].to_csv(paths['val'], index=False)
        df.iloc[v:].to_csv(paths['test'], index=False)

        from credit_risk.pipeline.train import train as _train
        _train(paths=paths, model_name='xgboost', fast_mode=True)
        yield paths
    finally:
        _factory_mod.load_config = _real
