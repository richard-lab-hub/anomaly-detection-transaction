"""
mcp_server.py — MCP Server for the IEEE Fraud Detection Pipeline
----------------------------------------------------------------
Thin wrapper only. All ML logic stays in the existing modules.
This file imports from them and exposes their functions as MCP tools.

Start the server (Terminal 1):
    source myenv/bin/activate
    python mcp_server.py
    → Runs on http://localhost:8000

Debug individual tools in the browser:
    mcp dev mcp_server.py
"""

import warnings
warnings.filterwarnings('ignore')

import glob
import json
import os
from typing import Optional

import joblib
import pandas as pd
from mcp.server.fastmcp import FastMCP

# ── Import all logic from existing modules ────────────────────────────────────
from config        import get_paths, selected_features, numerical_features, categorical_features
from preprocessing import split_and_save
from train         import train
from test          import test, run_shap
from predict       import predict

mcp = FastMCP("Fraud Detection Pipeline")

DEFAULT_DATA_DIR = '/mnt/c/Users/richa/Desktop/AI_Credit_Project/IEEEAICreditRiskProject'


# ── Tool 1 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def split_dataset(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Split train_with_features_target.csv into train (60%), val (10%), test (30%) sets.
    Run this first before train_model if the split files do not exist yet.
    Example: data_dir='/mnt/c/Users/richa/Desktop/AI_Credit_Project/IEEEAICreditRiskProject'
    """
    paths = get_paths(data_dir, 'xgboost')

    if not os.path.exists(paths['source']):
        return json.dumps({
            'status':  'error',
            'message': f"Source CSV not found: {paths['source']}",
        })

    split_and_save(source_csv=paths['source'], paths=paths)   # preprocessing.py

    train_df = pd.read_csv(paths['train'])
    val_df   = pd.read_csv(paths['val'])
    test_df  = pd.read_csv(paths['test'])

    return json.dumps({
        'status':           'success',
        'train_rows':       len(train_df),
        'val_rows':         len(val_df),
        'test_rows':        len(test_df),
        'train_fraud_rate': round(float(train_df['isFraud'].mean()), 4),
        'val_fraud_rate':   round(float(val_df['isFraud'].mean()),   4),
        'test_fraud_rate':  round(float(test_df['isFraud'].mean()),  4),
    })


# ── Tool 2 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def train_model(data_dir: str = DEFAULT_DATA_DIR, model_name: str = 'xgboost', fast_mode: bool = False) -> str:
    """
    Train the fraud detection model with GridSearchCV + SMOTE cross-validation.
    Saves the best pipeline to disk as fraud_pipeline_<model_name>.pkl.
    model_name : 'xgboost' (GPU, recommended) or 'rf' (CPU random forest).
    fast_mode  : True = one-combo grid, finishes in ~5 min, for quick testing.
                 False = full grid search for best accuracy (30-90 min).
    Example: data_dir='...', model_name='xgboost', fast_mode=True
    """
    paths = get_paths(data_dir, model_name)

    for key in ('train', 'val'):
        if not os.path.exists(paths[key]):
            return json.dumps({
                'status':  'error',
                'message': f"Split not found: {paths[key]}. Run split_dataset first.",
            })

    result = train(paths=paths, model_name=model_name, fast_mode=fast_mode)  # train.py
    return json.dumps({'status': 'success', **result})


# ── Tool 3 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def evaluate_model(data_dir: str = DEFAULT_DATA_DIR, model_name: str = 'xgboost') -> str:
    """
    Evaluate the trained model on the held-out test set.
    Returns ROC-AUC, precision, recall, F1-score, and confusion matrix.
    model_name must match the .pkl already saved on disk.
    Example: data_dir='...', model_name='xgboost'
    """
    paths = get_paths(data_dir, model_name)

    if not os.path.exists(paths['model']):
        return json.dumps({
            'status':  'error',
            'message': f"Model not found: {paths['model']}. Run train_model first.",
        })
    if not os.path.exists(paths['test']):
        return json.dumps({
            'status':  'error',
            'message': 'Test split not found. Run split_dataset first.',
        })

    result = test(paths=paths, explain=False)   # test.py
    return json.dumps({'status': 'success', **result})


# ── Tool 4 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def predict_transactions(
    data_dir:   str            = DEFAULT_DATA_DIR,
    input_csv:  str            = '',
    model_name: str            = 'xgboost',
    threshold:  Optional[float] = None,
) -> str:
    """
    Score new transactions for fraud risk using the trained model.
    Saves a new CSV with fraud_probability and fraud_flag columns next to input_csv.
    input_csv : full path to the CSV file containing transactions to score.
                Example: '/mnt/c/.../predict_without_target.csv'
    threshold : probability cutoff to flag a transaction as fraud (0.0-1.0).
                Leave empty to auto-load the optimal threshold saved during training.
                Override with an explicit value only if needed.
    """
    paths = get_paths(data_dir, model_name)

    if not os.path.exists(paths['model']):
        return json.dumps({
            'status':  'error',
            'message': 'Model not found. Run train_model first.',
        })
    if not os.path.exists(input_csv):
        return json.dumps({
            'status':  'error',
            'message': f"Input CSV not found: {input_csv}",
        })

    output_csv = os.path.splitext(input_csv)[0] + '_predictions.csv'

    result = predict(                           # predict.py
        input_csv=input_csv,
        output_csv=output_csv,
        model_path=paths['model'],
        threshold=threshold,
        model_name=model_name,
    )
    return json.dumps({'status': 'success', **result})


# ── Tool 5 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def explain_model(data_dir: str = DEFAULT_DATA_DIR, model_name: str = 'xgboost', n_samples: int = 500) -> str:
    """
    Generate SHAP feature importance plots (bar chart + beeswarm) and return the
    top 10 features that most drive fraud predictions.
    Saves PNG plots to <data_dir>/shap_agent/.
    n_samples : number of test rows to use (fewer = faster). Default 500.
    Example: data_dir='...', model_name='xgboost', n_samples=500
    """
    paths = get_paths(data_dir, model_name)

    if not os.path.exists(paths['model']):
        return json.dumps({
            'status':  'error',
            'message': 'Model not found. Run train_model first.',
        })
    if not os.path.exists(paths['test']):
        return json.dumps({
            'status':  'error',
            'message': 'Test split not found. Run split_dataset first.',
        })

    test_df    = pd.read_csv(paths['test'], index_col=False)
    pipeline   = joblib.load(paths['model'])
    output_dir = os.path.join(data_dir, 'shap_agent')

    result = run_shap(                          # test.py
        pipeline=pipeline,
        X_test=test_df.drop('isFraud', axis=1),
        y_test=test_df['isFraud'],
        n_global=n_samples,
        output_dir=output_dir,
        model_name=model_name,
    )
    return json.dumps({'status': 'success', **result})


# ── Tool 6 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_feature_config() -> str:
    """
    Return the feature lists used by the model: selected, numerical, and categorical.
    Useful to understand what columns the model expects in a new transaction CSV.
    """
    return json.dumps({
        'total_features':       len(selected_features),
        'numerical_count':      len(numerical_features),
        'categorical_count':    len(categorical_features),
        'selected_features':    selected_features,
        'numerical_features':   numerical_features,
        'categorical_features': categorical_features,
    })


# ── Tool 7 ────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_models(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    List all saved fraud detection model .pkl files in data_dir.
    Example: data_dir='/mnt/c/Users/richa/Desktop/AI_Credit_Project/IEEEAICreditRiskProject'
    """
    found  = sorted(glob.glob(os.path.join(data_dir, 'fraud_pipeline_*.pkl')))
    models = [
        {
            'name':    os.path.basename(p),
            'path':    p,
            'size_mb': round(os.path.getsize(p) / 1_048_576, 2),
        }
        for p in found
    ]
    return json.dumps({'status': 'success', 'models_found': len(models), 'models': models})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mcp.run(transport='sse')   # runs on http://localhost:8000 — agent.py connects here
