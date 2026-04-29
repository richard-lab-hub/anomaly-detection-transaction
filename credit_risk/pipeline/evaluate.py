import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive — plots saved to disk
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

from credit_risk.config.constants import base_parser, get_paths
from credit_risk.config.features import numerical_features, categorical_features
from credit_risk.data.preprocessing import apply_log_transforms
from credit_risk.explainability.shap_utils import _fraud_shap

# ColumnTransformer output order: numerical first, then categorical (matches preprocessing.py)
FEATURE_NAMES = numerical_features + categorical_features


def parse_args():
    parser = base_parser(
        'Evaluate the saved fraud detection model on the held-out test set'
    )
    parser.add_argument(
        '--explain', action='store_true',
        help='Generate SHAP explanations after evaluation'
    )
    parser.add_argument(
        '--n_global', type=int, default=2000,
        help='Max test samples used for global SHAP summary plots (default: 2000)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory to save SHAP plots (default: <data_dir>/shap_test/)'
    )
    return parser.parse_args()


def run_shap(pipeline, X_test, y_test, n_global, output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = pipeline.named_steps['preprocessor']
    model        = pipeline.named_steps['model']
    X_test_t     = preprocessor.transform(X_test)

    print("\nBuilding SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)

    # ── Global: bar + beeswarm ────────────────────────────────────────────
    n        = min(n_global, len(X_test_t))
    idx      = np.random.default_rng(42).choice(len(X_test_t), size=n, replace=False)
    X_sample = X_test_t[idx]

    print(f"Computing global SHAP values on {n:,} samples ...")
    sv, _ = _fraud_shap(explainer, X_sample)

    mean_abs = np.abs(sv).mean(axis=0)
    top10    = np.argsort(mean_abs)[::-1][:10]
    top10_features = [
        {'rank': int(r + 1), 'feature': FEATURE_NAMES[fi],
         'mean_shap': round(float(mean_abs[fi]), 4)}
        for r, fi in enumerate(top10)
    ]
    print("\nTop 10 features by mean |SHAP value|:")
    for item in top10_features:
        print(f"  {item['rank']:2d}. {item['feature']:<22s}  {item['mean_shap']:.4f}")

    shap.summary_plot(sv, X_sample, feature_names=FEATURE_NAMES, plot_type='bar', show=False)
    bar_path = os.path.join(output_dir, f'shap_test_bar_{model_name}.png')
    plt.savefig(bar_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSHAP bar plot     → {bar_path}")

    shap.summary_plot(sv, X_sample, feature_names=FEATURE_NAMES, show=False)
    bee_path = os.path.join(output_dir, f'shap_test_beeswarm_{model_name}.png')
    plt.savefig(bee_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"SHAP beeswarm     → {bee_path}")

    # ── Local: waterfall for the first fraud case in the test set ─────────
    fraud_positions = np.where(y_test.values == 1)[0]
    if len(fraud_positions) == 0:
        print("No fraud cases found in test set — skipping local waterfall.")
        return {
            'samples_used':    int(n),
            'top_10_features': top10_features,
            'bar_plot':        bar_path,
            'beeswarm_plot':   bee_path,
            'waterfall_plot':  None,
        }

    row_idx  = int(fraud_positions[0])
    x_row    = X_test_t[row_idx: row_idx + 1]
    sv_row, ev_row = _fraud_shap(explainer, x_row)

    explanation = shap.Explanation(
        values=sv_row[0],
        base_values=ev_row,
        data=x_row[0],
        feature_names=FEATURE_NAMES,
    )
    shap.plots.waterfall(explanation, show=False)
    wf_path = os.path.join(output_dir, f'shap_test_waterfall_{model_name}_row{row_idx}.png')
    plt.savefig(wf_path, bbox_inches='tight', dpi=150)
    plt.close()

    fraud_prob = pipeline.predict_proba(X_test.iloc[[row_idx]])[:, 1][0]
    print(f"SHAP waterfall    → {wf_path}")
    print(f"  (first fraud case: row {row_idx}, fraud probability: {fraud_prob:.4f})")

    return {
        'samples_used':    int(n),
        'top_10_features': top10_features,
        'bar_plot':        bar_path,
        'beeswarm_plot':   bee_path,
        'waterfall_plot':  wf_path,
    }


def test(paths, explain=False, n_global=2000, output_dir=None, model_name=None):
    print("Loading held-out test split...")
    test_df = apply_log_transforms(pd.read_csv(paths['test'], index_col=False))

    X_test = test_df.drop('isFraud', axis=1)
    y_test = test_df['isFraud']

    print(f"Test samples : {len(X_test):,} | Fraud rate: {y_test.mean():.4f}")
    print(f"\nLoading pipeline from {paths['model']}...")

    pipeline = joblib.load(paths['model'])

    # SMOTE is skipped automatically at inference (predict path)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred       = pipeline.predict(X_test)

    pr_auc = average_precision_score(y_test, y_pred_proba)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Legit', 'Fraud'],
                                   output_dict=True)

    print(f"\nTest PR-AUC  : {pr_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)

    if explain:
        run_shap(
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
            n_global=n_global,
            output_dir=output_dir,
            model_name=model_name,
        )

    return {
        'pr_auc':  round(float(pr_auc), 4),
        'accuracy':      round(float(report['accuracy']), 4),
        'fraud_metrics': {k: round(float(v), 4) for k, v in report['Fraud'].items()
                          if isinstance(v, float)},
        'legit_metrics': {k: round(float(v), 4) for k, v in report['Legit'].items()
                          if isinstance(v, float)},
        'confusion_matrix': {
            'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
        },
        'test_samples': int(len(y_test)),
        'fraud_rate':   round(float(y_test.mean()), 4),
    }


def main():
    args       = parse_args()
    paths      = get_paths(args.data_dir, args.model)
    output_dir = args.output_dir or os.path.join(args.data_dir, 'shap_test')

    test(
        paths=paths,
        explain=args.explain,
        n_global=args.n_global,
        output_dir=output_dir,
        model_name=args.model,
    )


if __name__ == '__main__':
    main()
