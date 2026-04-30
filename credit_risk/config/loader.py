import os
import yaml

# config/ lives at the project root: two levels above credit_risk/config/loader.py
_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'config'
)

_cache: dict = {}


def load_config(model_name: str, config_dir: str = None) -> dict:
    effective_dir = config_dir or _CONFIG_DIR
    cache_key = (model_name, effective_dir)
    if cache_key in _cache:
        return _cache[cache_key]
    path = os.path.join(effective_dir, f'{model_name}.yaml')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Expected one of: xgboost.yaml, rf.yaml"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _cache[cache_key] = cfg
    return cfg
