import warnings
warnings.filterwarnings('ignore')

import json
import os
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

from credit_risk.config.constants import base_parser, get_paths
from credit_risk.models.factory import get_model_and_grid
from credit_risk.data.preprocessing import apply_log_transforms, build_preprocessor, split_and_save


def parse_args():
    parser = base_parser('Train the fraud detection model')
    parser.add_argument(
        '--skip_split',
        action='store_true',
        help='Skip dataset splitting and use already-saved splits on disk'
    )
    parser.add_argument(
        '--fast_mode',
        action='store_true',
        help='XGBoost only: single-combo grid search (~5 min instead of 30-90 min)'
    )
    return parser.parse_args()


def train(paths, model_name, fast_mode=False):
    print(f"\nModel selected : {model_name}")
    print("Loading train and validation splits...")

    for split, path in [('train', paths['train']), ('val', paths['val'])]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{split} split not found: {path}. Run without --skip_split first.")

    train_df = apply_log_transforms(pd.read_csv(paths['train'], index_col=False))
    val_df   = apply_log_transforms(pd.read_csv(paths['val'],   index_col=False))

    X_train = train_df.drop('isFraud', axis=1)
    y_train = train_df['isFraud']
    X_val   = val_df.drop('isFraud', axis=1)
    y_val   = val_df['isFraud']

    print(f"Train samples : {len(X_train):,} | Fraud rate: {y_train.mean():.4f}")
    print(f"Val   samples : {len(X_val):,}   | Fraud rate: {y_val.mean():.4f}")

    model, param_grid = get_model_and_grid(model_name, fast_mode)

    if y_train.sum() == 0:
        raise ValueError("No fraud cases found in training data — cannot train fraud detector.")

    if model_name == 'xgboost':
        # XGBoost handles imbalance via scale_pos_weight — SMOTE is redundant
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
        print(f"scale_pos_weight : {scale_pos_weight:.2f}")
        model.set_params(scale_pos_weight=scale_pos_weight)
        pipeline = SKPipeline([
            ('preprocessor', build_preprocessor()),
            ('model', model)
        ])
    else:
        # RF has no built-in imbalance handling — keep SMOTE
        pipeline = ImbPipeline([
            ('preprocessor', build_preprocessor()),
            ('smote', SMOTE(sampling_strategy='minority', k_neighbors=5, random_state=42)),
            ('model', model)
        ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,
        n_jobs=1,    # SMOTE conflicts with multiprocessing
        refit=False, # only find best params — final fit done below
        verbose=2
    )

    print("\nRunning GridSearchCV on training split...")
    grid_search.fit(X_train, y_train)

    print(f"\nBest CV PR-AUC (train folds) : {grid_search.best_score_:.4f}")
    print(f"Best parameters               : {grid_search.best_params_}")

    pipeline.set_params(**grid_search.best_params_)

    if model_name == 'xgboost':
        preprocessor = pipeline.named_steps['preprocessor']
        X_train_t    = preprocessor.fit_transform(X_train)
        X_val_t      = preprocessor.transform(X_val)
        pipeline.named_steps['model'].set_params(early_stopping_rounds=50)
        pipeline.named_steps['model'].fit(
            X_train_t, y_train,
            eval_set=[(X_val_t, y_val)],
            verbose=False
        )
        print(f"XGBoost best iteration : {pipeline.named_steps['model'].best_iteration}")
    else:
        pipeline.fit(X_train, y_train)

    best_pipeline = pipeline

    val_proba     = best_pipeline.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, val_proba)
    print(f"Validation PR-AUC (unseen) : {val_pr_auc:.4f}")
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    beta = 2
    f2_scores = ((1 + beta**2) * precisions[:-1] * recalls[:-1]
                 / (beta**2 * precisions[:-1] + recalls[:-1] + 1e-9))
    best_idx           = int(np.argmax(f2_scores))
    optimal_threshold  = float(thresholds[best_idx])
    optimal_val_f2     = float(f2_scores[best_idx])
    print(f"Optimal threshold (max-F2 on val)  : {optimal_threshold:.4f}")
    print(f"Val F2 at optimal threshold        : {optimal_val_f2:.4f}")

    joblib.dump(best_pipeline, paths['model'])
    print(f"\nBest pipeline saved → {paths['model']}")

    with open(paths['threshold'], 'w') as fh:
        json.dump({
            'threshold': round(optimal_threshold, 6),
            'val_f2':    round(optimal_val_f2, 4),
            'val_pr_auc': round(float(val_pr_auc), 4),
        }, fh, indent=2)
    print(f"Threshold saved    → {paths['threshold']}")

    return {
        'best_cv_pr_auc':      round(float(grid_search.best_score_), 4),
        'val_pr_auc':          round(float(val_pr_auc), 4),
        'optimal_threshold':   round(optimal_threshold, 6),
        'val_f2_at_threshold': round(optimal_val_f2, 4),
        'best_params': {k: (int(v) if hasattr(v, 'item') else v)
                        for k, v in grid_search.best_params_.items()},
        'model_path':     paths['model'],
        'threshold_path': paths['threshold'],
    }


def main():
    args  = parse_args()
    paths = get_paths(args.data_dir, args.model)

    if not args.skip_split:
        split_and_save(source_csv=paths['source'], paths=paths)

    train(paths=paths, model_name=args.model, fast_mode=args.fast_mode)


if __name__ == '__main__':
    main()
