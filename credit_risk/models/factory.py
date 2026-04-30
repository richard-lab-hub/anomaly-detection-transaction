from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from credit_risk.config.loader import load_config

_MODEL_CLASSES = {
    'xgboost': XGBClassifier,
    'rf':      RandomForestClassifier,
}


def get_model_and_grid(model_name, fast_mode=False):
    cfg = load_config(model_name)
    model_cfg = cfg['model']

    cls = _MODEL_CLASSES.get(model_name)
    if cls is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: rf, xgboost"
        )

    model      = cls(**model_cfg['params'])
    grid_key   = 'fast_grid' if fast_mode else 'param_grid'
    param_grid = cfg[grid_key]

    return model, param_grid
