import pytest
from credit_risk.config.loader import load_config


def test_load_config_returns_dict():
    cfg = load_config('xgboost')
    assert isinstance(cfg, dict)


def test_load_config_has_expected_keys():
    cfg = load_config('xgboost')
    assert {'files', 'features', 'model', 'param_grid', 'fast_grid'} <= set(cfg.keys())


def test_load_config_caching():
    cfg1 = load_config('xgboost')
    cfg2 = load_config('xgboost')
    assert cfg1 is cfg2


def test_load_config_rf():
    cfg = load_config('rf')
    assert 'model' in cfg
    assert 'params' in cfg['model']


def test_load_config_unknown_model_raises():
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config('nonexistent_model_xyz')


def test_load_config_custom_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config('xgboost', config_dir=str(tmp_path))


def test_load_config_features_structure():
    cfg = load_config('xgboost')
    feats = cfg['features']
    assert 'selected' in feats
    assert 'numerical' in feats
    assert 'categorical' in feats
