import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from credit_risk.config.features import categorical_features, numerical_features, selected_features


def split_and_save(source_csv, paths):
    if not os.path.exists(source_csv):
        raise FileNotFoundError(f"Source dataset not found: {source_csv}")
    print("Loading raw dataset...")
    df = pd.read_csv(source_csv, index_col=False)
    if 'isFraud' not in df.columns:
        raise ValueError("Dataset missing required target column 'isFraud'")
    df = df[selected_features + ['isFraud']]

    # First cut: 70% train+val | 30% test
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df['isFraud'],
        random_state=42
    )

    # Second cut: 10/70 ≈ 0.1429 of pool → ~10% of total for val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1429,
        stratify=train_val_df['isFraud'],
        random_state=42
    )

    train_df.to_csv(paths['train'], index=False)
    val_df.to_csv(paths['val'],     index=False)
    test_df.to_csv(paths['test'],   index=False)

    total = len(df)
    print(f"Total   : {total:,} rows — deterministic split (random_state=42)")
    print(f"Train   : {len(train_df):,} rows ({len(train_df)/total*100:.1f}%) → {paths['train']}")
    print(f"Val     : {len(val_df):,} rows ({len(val_df)/total*100:.1f}%)  → {paths['val']}")
    print(f"Test    : {len(test_df):,} rows ({len(test_df)/total*100:.1f}%)  → {paths['test']}")
    print(
        f"Fraud rate — train: {train_df['isFraud'].mean():.4f} | "
        f"val: {val_df['isFraud'].mean():.4f} | "
        f"test: {test_df['isFraud'].mean():.4f}"
    )


def apply_log_transforms(df):
    """Apply log1p to TransactionAmt before the pipeline — called at data load time."""
    if 'TransactionAmt' not in df.columns:
        raise ValueError("Input data is missing required column: TransactionAmt")
    df = df.copy()
    df['TransactionAmt'] = np.log1p(df['TransactionAmt'].clip(lower=0))
    return df


def build_preprocessor():
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
    ])
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numerical_pipe, numerical_features),
            ('cat', categorical_pipe, categorical_features)
        ],
        remainder='drop'
    )
