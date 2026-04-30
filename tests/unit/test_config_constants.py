import pytest
from credit_risk.config.constants import get_paths, base_parser


def test_get_paths_returns_expected_keys():
    paths = get_paths('/some/dir', 'xgboost')
    assert set(paths.keys()) == {'train', 'val', 'test', 'source', 'model', 'threshold'}


def test_get_paths_all_values_start_with_data_dir():
    paths = get_paths('/mydir', 'xgboost')
    for key in paths:
        assert paths[key].startswith('/mydir'), f"{key} path does not start with data_dir"


def test_get_paths_model_name_in_model_path():
    paths = get_paths('/dir', 'xgboost')
    assert 'xgboost' in paths['model']


def test_get_paths_threshold_path_matches_model():
    paths = get_paths('/dir', 'xgboost')
    assert paths['threshold'].replace('.threshold', '') == paths['model'].replace('.pkl', '')


def test_base_parser_default_model():
    parser = base_parser('test')
    args = parser.parse_args(['--data_dir', '/tmp'])
    assert args.data_dir == '/tmp'
    assert args.model == 'xgboost'


def test_base_parser_rf_choice():
    parser = base_parser('test')
    args = parser.parse_args(['--data_dir', '/tmp', '--model', 'rf'])
    assert args.model == 'rf'


def test_base_parser_invalid_model_exits():
    parser = base_parser('test')
    with pytest.raises(SystemExit):
        parser.parse_args(['--data_dir', '/tmp', '--model', 'svm'])


def test_base_parser_data_dir_required():
    parser = base_parser('test')
    with pytest.raises(SystemExit):
        parser.parse_args([])
