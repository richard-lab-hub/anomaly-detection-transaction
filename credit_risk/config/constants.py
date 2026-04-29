import argparse

TRAIN_FILE   = 'train_split.csv'
VAL_FILE     = 'val_split.csv'
TEST_FILE    = 'test_split.csv'
SOURCE_FILE  = 'train_with_features_target.csv'
MODEL_PREFIX = 'anomaly_pipeline'


def get_paths(data_dir, model_name):
    return {
        'train':     f'{data_dir}/{TRAIN_FILE}',
        'val':       f'{data_dir}/{VAL_FILE}',
        'test':      f'{data_dir}/{TEST_FILE}',
        'source':    f'{data_dir}/{SOURCE_FILE}',
        'model':     f'{data_dir}/{MODEL_PREFIX}_{model_name}.pkl',
        'threshold': f'{data_dir}/{MODEL_PREFIX}_{model_name}.threshold',
    }


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to the folder containing all CSV splits and saved models'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['rf', 'xgboost'],
        help='Model to use: rf, xgboost (default: xgboost)'
    )
    return parser
