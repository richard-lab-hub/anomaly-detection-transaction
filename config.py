import argparse

selected_features = [
    'TransactionDT', 'TransactionAmt',
    'C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D4', 'D8', 'D10', 'D15',
    'V44', 'V45', 'V53', 'V62', 'V86', 'V87',
    'V127', 'V128', 'V149', 'V156',
    'V187', 'V188', 'V189',
    'V199', 'V200', 'V201',
    'V242', 'V243', 'V244', 'V246',
    'V257', 'V258',
    'V280', 'V282', 'V283',
    'V294',
    'V306', 'V307', 'V308', 'V310', 'V312', 'V315', 'V317',
    'id_02', 'id_19', 'id_20', 'id_31',
    'DeviceInfo',
    'P_emaildomain', 'R_emaildomain',
    'card1', 'card2', 'card3', 'card5', 'card6',
    'addr1', 'M4',
]

numerical_features = [
    'TransactionDT', 'TransactionAmt',
    'C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14',
    'D1', 'D2', 'D4', 'D8', 'D10', 'D15',
    'V44', 'V45', 'V53', 'V62', 'V86', 'V87',
    'V127', 'V128', 'V149', 'V156',
    'V187', 'V188', 'V189',
    'V199', 'V200', 'V201',
    'V242', 'V243', 'V244', 'V246',
    'V257', 'V258',
    'V280', 'V282', 'V283',
    'V294',
    'V306', 'V307', 'V308', 'V310', 'V312', 'V315', 'V317', 'id_02'
]

categorical_features = [
     'id_19', 'id_20', 'id_31',
    'DeviceInfo',
    'P_emaildomain', 'R_emaildomain',
    'card1', 'card2', 'card3', 'card5', 'card6',
    'addr1', 'M4',
]

TRAIN_FILE   = 'train_split.csv'
VAL_FILE     = 'val_split.csv'
TEST_FILE    = 'test_split.csv'
SOURCE_FILE  = 'train_with_features_target.csv'
MODEL_PREFIX = 'fraud_pipeline'


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