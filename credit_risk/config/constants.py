import argparse
from credit_risk.config.loader import load_config


def get_paths(data_dir, model_name):
    files = load_config(model_name)['files']
    prefix = files['model_prefix']
    return {
        'train':     f"{data_dir}/{files['train']}",
        'val':       f"{data_dir}/{files['val']}",
        'test':      f"{data_dir}/{files['test']}",
        'source':    f"{data_dir}/{files['source']}",
        'model':     f"{data_dir}/{prefix}_{model_name}.pkl",
        'threshold': f"{data_dir}/{prefix}_{model_name}.threshold",
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
