from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def get_model_and_grid(model_name, fast_mode=False):
    if model_name == 'xgboost':
        model = XGBClassifier(
            eval_metric='auc',
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        param_grid = {
            'model__n_estimators':     [500, 800],
            'model__max_depth':        [4, 6, 8],
            'model__learning_rate':    [0.05, 0.1],
            'model__subsample':        [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
        }
        fast_grid = {
            'model__n_estimators':     [300],
            'model__max_depth':        [6],
            'model__learning_rate':    [0.1],
            'model__subsample':        [0.8],
            'model__colsample_bytree': [0.8],
        }

    elif model_name == 'rf':
        model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        param_grid = {
            'model__n_estimators':      [300, 500],
            'model__max_depth':         [10, 20, None],
            'model__min_samples_split': [2, 5],
            'model__max_features':      ['sqrt', 'log2'],
        }

        fast_grid = {
            'model__n_estimators':      [500],
            'model__max_depth':         [10],
            'model__min_samples_split': [5],
            'model__max_features':      ['sqrt'],
        }

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: rf, xgboost"
        )
    
    return model, fast_grid if fast_mode else param_grid
