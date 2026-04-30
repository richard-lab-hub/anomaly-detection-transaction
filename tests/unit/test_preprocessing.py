import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from credit_risk.config.features import categorical_features, numerical_features
from credit_risk.data.preprocessing import apply_log_transforms, build_preprocessor


def _make_df(n=50):
    rng = np.random.default_rng(0)
    data = {col: rng.uniform(1, 100, n) for col in numerical_features}
    for col in categorical_features:
        data[col] = rng.choice(['A', 'B', 'C'], n)
    return pd.DataFrame(data)


def test_apply_log_transforms_does_not_mutate_input():
    df = _make_df()
    original = df['TransactionAmt'].copy()
    apply_log_transforms(df)
    pd.testing.assert_series_equal(df['TransactionAmt'], original)


def test_apply_log_transforms_applies_log1p():
    df = _make_df()
    result = apply_log_transforms(df)
    expected = np.log1p(df['TransactionAmt'].clip(lower=0))
    pd.testing.assert_series_equal(result['TransactionAmt'], expected)


def test_apply_log_transforms_clips_negative_values():
    df = _make_df()
    df['TransactionAmt'] = -5.0
    result = apply_log_transforms(df)
    assert (result['TransactionAmt'] == np.log1p(0)).all()


def test_apply_log_transforms_missing_column_raises():
    df = pd.DataFrame({'SomeOtherCol': [1, 2, 3]})
    with pytest.raises(ValueError, match="TransactionAmt"):
        apply_log_transforms(df)


def test_build_preprocessor_returns_column_transformer():
    assert isinstance(build_preprocessor(), ColumnTransformer)


def test_build_preprocessor_step_names():
    names = [name for name, _, _ in build_preprocessor().transformers]
    assert 'num' in names
    assert 'cat' in names


def test_build_preprocessor_output_shape():
    df = _make_df()
    result = build_preprocessor().fit_transform(df)
    assert result.shape == (len(df), len(numerical_features) + len(categorical_features))


def test_build_preprocessor_handles_nulls():
    df = _make_df()
    df.loc[0, numerical_features[0]] = np.nan
    df.loc[1, categorical_features[0]] = None
    result = build_preprocessor().fit_transform(df)
    assert not np.isnan(result).any()
