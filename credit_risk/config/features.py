from credit_risk.config.loader import load_config

# Features are shared across models — xgboost.yaml is the canonical source
_features = load_config('xgboost')['features']

selected_features    = _features['selected']
numerical_features   = _features['numerical']
categorical_features = _features['categorical']
