import os

import pandas as pd
import pytest

from credit_risk.pipeline.predict import predict as run_predict

_EXPECTED_KEYS = {
    'total_transactions', 'flagged_as_fraud',
    'fraud_percentage', 'threshold_used', 'output_saved_to',
}


def _write_input(tmp_path, synthetic_df, suffix=''):
    path = str(tmp_path / f'input{suffix}.csv')
    synthetic_df.drop(columns=['isFraud']).to_csv(path, index=False)
    return path


def test_predict_creates_output_file(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out.csv')
    run_predict(
        input_csv=_write_input(tmp_path, synthetic_df),
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    assert os.path.exists(out)


def test_predict_output_has_required_columns(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out2.csv')
    run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '2'),
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    result = pd.read_csv(out)
    assert 'fraud_probability' in result.columns
    assert 'fraud_flag' in result.columns


def test_predict_probabilities_in_unit_range(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out3.csv')
    run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '3'),
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    probs = pd.read_csv(out)['fraud_probability']
    assert probs.between(0.0, 1.0).all()


def test_predict_flag_respects_explicit_threshold(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out4.csv')
    threshold = 0.5
    run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '4'),
        output_csv=out,
        model_path=trained_paths_module['model'],
        threshold=threshold,
    )
    result = pd.read_csv(out)
    flagged = result[result['fraud_flag'] == 1]
    if not flagged.empty:
        assert (flagged['fraud_probability'] >= threshold).all()


def test_predict_returns_summary_dict(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out5.csv')
    summary = run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '5'),
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    assert _EXPECTED_KEYS <= set(summary.keys())


def test_predict_total_matches_input_rows(tmp_path, trained_paths_module, synthetic_df):
    out = str(tmp_path / 'out6.csv')
    summary = run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '6'),
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    assert summary['total_transactions'] == len(synthetic_df)


def test_predict_missing_model_raises(tmp_path, synthetic_df):
    out = str(tmp_path / 'out7.csv')
    with pytest.raises(FileNotFoundError):
        run_predict(
            input_csv=_write_input(tmp_path, synthetic_df, '7'),
            output_csv=out,
            model_path=str(tmp_path / 'nonexistent.pkl'),
        )


def test_predict_missing_columns_triggers_warning(tmp_path, trained_paths_module, capsys):
    out = str(tmp_path / 'out_partial.csv')
    df_partial = pd.DataFrame({'TransactionAmt': [100.0, 200.0]})
    inp = str(tmp_path / 'partial.csv')
    df_partial.to_csv(inp, index=False)
    run_predict(
        input_csv=inp,
        output_csv=out,
        model_path=trained_paths_module['model'],
    )
    assert 'Warning' in capsys.readouterr().out


def test_predict_empty_input_raises(tmp_path, trained_paths_module):
    out = str(tmp_path / 'out_empty.csv')
    inp = str(tmp_path / 'empty.csv')
    pd.DataFrame({'TransactionAmt': pd.Series([], dtype='float64')}).to_csv(inp, index=False)
    with pytest.raises(ValueError):
        run_predict(
            input_csv=inp,
            output_csv=out,
            model_path=trained_paths_module['model'],
        )


def test_predict_corrupt_threshold_falls_back_to_half(tmp_path, trained_paths_module, synthetic_df):
    import shutil
    model_copy = str(tmp_path / 'model_copy.pkl')
    threshold_copy = str(tmp_path / 'model_copy.threshold')
    shutil.copy(trained_paths_module['model'], model_copy)
    with open(threshold_copy, 'w') as fh:
        fh.write('not valid json {{{')
    out = str(tmp_path / 'out_corrupt.csv')
    result = run_predict(
        input_csv=_write_input(tmp_path, synthetic_df, '_corrupt'),
        output_csv=out,
        model_path=model_copy,
    )
    assert result['threshold_used'] == 0.5
