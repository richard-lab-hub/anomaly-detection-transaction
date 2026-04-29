import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive — plots saved to disk
import matplotlib.pyplot as plt
import shap

from credit_risk.config.features import selected_features, numerical_features, categorical_features
from credit_risk.data.preprocessing import apply_log_transforms
from credit_risk.explainability.shap_utils import _fraud_shap

# ColumnTransformer output order: numerical first, then categorical (matches preprocessing.py)
FEATURE_NAMES = numerical_features + categorical_features


def parse_args():
    parser = argparse.ArgumentParser(description='Run fraud predictions on new unseen transactions')
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the saved pipeline .pkl file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to CSV file with new transactions to score'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='predictions.csv',
        help='Path to save predictions CSV (default: predictions.csv)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Probability threshold for flagging fraud. '
             'If omitted, the threshold saved during training is used. '
             'Falls back to 0.5 if no saved threshold exists.'
    )
    parser.add_argument(
        '--explain', action='store_true',
        help='Generate SHAP waterfall plots for flagged transactions'
    )
    parser.add_argument(
        '--n_explain', type=int, default=5,
        help='Max number of flagged transactions to explain with SHAP (default: 5)'
    )
    return parser.parse_args()


def run_shap_local(pipeline, X, flagged_idx, n_explain, output_dir, model_name):
    """Generate a SHAP waterfall plot for each of the top-n flagged transactions."""
    os.makedirs(output_dir, exist_ok=True)

    rows_to_explain = flagged_idx[:n_explain]
    preprocessor    = pipeline.named_steps['preprocessor']
    model           = pipeline.named_steps['model']

    X_subset   = X.iloc[rows_to_explain]
    X_subset_t = preprocessor.transform(X_subset)

    print(f"\nBuilding SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    print(f"Generating SHAP waterfall for {len(rows_to_explain)} flagged transaction(s) ...")

    for i, original_pos in enumerate(rows_to_explain):
        x_row        = X_subset_t[i: i + 1]
        sv, ev       = _fraud_shap(explainer, x_row)

        explanation = shap.Explanation(
            values=sv[0],
            base_values=ev,
            data=x_row[0],
            feature_names=FEATURE_NAMES,
        )

        shap.plots.waterfall(explanation, show=False)
        wf_path = os.path.join(
            output_dir, f'shap_predict_waterfall_{model_name}_row{original_pos}.png'
        )
        plt.savefig(wf_path, bbox_inches='tight', dpi=150)
        plt.close()

        fraud_prob = pipeline.predict_proba(X.iloc[[original_pos]])[:, 1][0]
        print(f"  Row {original_pos:>6d}  fraud prob {fraud_prob:.4f}  → {wf_path}")


def _load_threshold(model_path, explicit_threshold):
    """Return the threshold to use, with a clear log of its source."""
    if explicit_threshold is not None:
        print(f"Threshold          : {explicit_threshold} (explicitly provided)")
        return explicit_threshold

    threshold_path = os.path.splitext(model_path)[0] + '.threshold'
    if os.path.exists(threshold_path):
        try:
            with open(threshold_path) as fh:
                saved = json.load(fh)
            t = saved['threshold']
            print(f"Threshold          : {t} (loaded from {threshold_path})")
            return t
        except (json.JSONDecodeError, KeyError):
            print("Threshold file corrupt — falling back to 0.5")

    print("Threshold: 0.5 (fallback — no saved threshold found; re-run train.py)")
    return 0.5


def predict(input_csv, output_csv, model_path, threshold=None, model_name='',
            explain=False, n_explain=5):
    print(f"Loading new transactions from {input_csv}...")
    df = pd.read_csv(input_csv, index_col=False)
    df.columns = df.columns.str.replace('-', '_', regex=False)
    df = apply_log_transforms(df)

    missing_cols = [f for f in selected_features if f not in df.columns]
    if missing_cols:
        print(
            f"Warning: {len(missing_cols)} expected feature(s) missing from input — "
            f"will be imputed by the pipeline: {missing_cols}"
        )

    X = df.reindex(columns=selected_features)

    print(f"Transactions to score : {len(X):,}")
    print(f"Loading pipeline from {model_path}...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    if len(X) == 0:
        raise ValueError(f"Input file is empty: {input_csv}")

    pipeline    = joblib.load(model_path)
    threshold   = _load_threshold(model_path, threshold)
    fraud_proba = pipeline.predict_proba(X)[:, 1]
    fraud_flag  = (fraud_proba >= threshold).astype(int)

    results = pd.DataFrame({
        'fraud_probability': fraud_proba.round(4),
        'fraud_flag':        fraud_flag,
    })

    results.to_csv(output_csv, index=False)

    flagged = int(fraud_flag.sum())
    print(f"Flagged fraud  : {flagged:,} / {len(X):,} ({flagged / len(X) * 100:.2f}%)")
    print(f"Predictions saved → {output_csv}")

    if explain:
        flagged_idx = np.where(fraud_flag == 1)[0]
        if len(flagged_idx) == 0:
            print("No transactions flagged — skipping SHAP explanations.")
        else:
            output_dir = os.path.splitext(output_csv)[0] + '_shap'
            run_shap_local(
                pipeline=pipeline,
                X=X,
                flagged_idx=flagged_idx,
                n_explain=n_explain,
                output_dir=output_dir,
                model_name=model_name,
            )

    return {
        'total_transactions': int(len(X)),
        'flagged_as_fraud':   int(flagged),
        'fraud_percentage':   round(float(flagged / len(X) * 100), 2),
        'threshold_used':     threshold,
        'output_saved_to':    output_csv,
    }


def main():
    args       = parse_args()
    model_name = os.path.splitext(os.path.basename(args.model_path))[0].split('_')[-1]

    predict(
        input_csv=args.input,
        output_csv=args.output,
        model_path=args.model_path,
        threshold=args.threshold,
        model_name=model_name,
        explain=args.explain,
        n_explain=args.n_explain,
    )


if __name__ == '__main__':
    main()
