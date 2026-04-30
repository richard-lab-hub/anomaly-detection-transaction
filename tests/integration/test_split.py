import os

import pandas as pd
import pytest

from credit_risk.data.preprocessing import split_and_save


def _paths(tmp):
    return {
        'train': str(tmp / 'train.csv'),
        'val':   str(tmp / 'val.csv'),
        'test':  str(tmp / 'test.csv'),
    }


def test_split_creates_three_files(tmp_path, synthetic_df):
    src = str(tmp_path / 'source.csv')
    synthetic_df.to_csv(src, index=False)
    paths = _paths(tmp_path)
    split_and_save(source_csv=src, paths=paths)
    for key in ('train', 'val', 'test'):
        assert os.path.exists(paths[key]), f"missing {key} split"


def test_split_rows_sum_to_total(tmp_path, synthetic_df):
    src = str(tmp_path / 'source.csv')
    synthetic_df.to_csv(src, index=False)
    paths = _paths(tmp_path)
    split_and_save(source_csv=src, paths=paths)
    total = sum(len(pd.read_csv(paths[k])) for k in ('train', 'val', 'test'))
    assert total == len(synthetic_df)


def test_split_test_is_30_percent(tmp_path, synthetic_df):
    src = str(tmp_path / 'source.csv')
    synthetic_df.to_csv(src, index=False)
    paths = _paths(tmp_path)
    split_and_save(source_csv=src, paths=paths)
    test_frac = len(pd.read_csv(paths['test'])) / len(synthetic_df)
    assert abs(test_frac - 0.30) < 0.02


def test_split_stratification_preserved(tmp_path, synthetic_df):
    src = str(tmp_path / 'source.csv')
    synthetic_df.to_csv(src, index=False)
    paths = _paths(tmp_path)
    split_and_save(source_csv=src, paths=paths)
    source_rate = synthetic_df['isFraud'].mean()
    for key in ('train', 'val', 'test'):
        rate = pd.read_csv(paths[key])['isFraud'].mean()
        assert abs(rate - source_rate) < 0.05, \
            f"{key} fraud rate {rate:.4f} deviates from source {source_rate:.4f}"


def test_split_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_and_save(
            source_csv=str(tmp_path / 'nonexistent.csv'),
            paths=_paths(tmp_path),
        )


def test_split_missing_target_column_raises(tmp_path, synthetic_df):
    df_no_target = synthetic_df.drop(columns=['isFraud'])
    src = str(tmp_path / 'source_bad.csv')
    df_no_target.to_csv(src, index=False)
    with pytest.raises(ValueError, match="isFraud"):
        split_and_save(source_csv=src, paths=_paths(tmp_path))
