import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from credit_risk.models.factory import get_model_and_grid


def test_xgboost_returns_xgb_classifier():
    model, _ = get_model_and_grid('xgboost')
    assert isinstance(model, XGBClassifier)


def test_rf_returns_random_forest_classifier():
    model, _ = get_model_and_grid('rf')
    assert isinstance(model, RandomForestClassifier)


def test_rf_balanced_class_weight():
    model, _ = get_model_and_grid('rf')
    assert model.class_weight == 'balanced'


def test_fast_mode_is_single_combination():
    _, grid = get_model_and_grid('xgboost', fast_mode=True)
    total = 1
    for values in grid.values():
        total *= len(values)
    assert total == 1


def test_full_grid_has_multiple_combinations():
    _, grid = get_model_and_grid('xgboost', fast_mode=False)
    total = 1
    for values in grid.values():
        total *= len(values)
    assert total > 1


def test_param_grid_keys_have_model_prefix():
    _, grid = get_model_and_grid('xgboost')
    assert all(k.startswith('model__') for k in grid)


def test_unknown_model_raises():
    # load_config raises FileNotFoundError before the ValueError check is reached
    with pytest.raises((FileNotFoundError, ValueError)):
        get_model_and_grid('svm')


def test_unknown_model_class_raises_value_error(monkeypatch):
    import credit_risk.models.factory as _mod
    monkeypatch.setattr(_mod, '_MODEL_CLASSES', {})
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_and_grid('xgboost')
