# IEEE AI Credit Risk Project — Fraud Detection Pipeline

A machine learning system for credit card anomaly detection using the IEEE Fraud Detection Dataset. Supports training, evaluation, and scoring via CLI scripts or a natural language AI agent backed by a local Ollama model and MCP server. FN (top right) is the dangerous cell → only Recall sees it → Recall must be prioritised.

---

## Prerequisites

### 1. Install dependencies

**Option A — editable package install (recommended after refactor)**
```bash
pip install -e .
```
This installs the `credit_risk` package and registers five CLI entry points:
`credit-risk-train`, `credit-risk-evaluate`, `credit-risk-predict`, `credit-risk-server`, `credit-risk-agent`.

**Option B — plain requirements**
```bash
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model (Terminal 1)
```bash
ollama serve
ollama pull qwen2.5:7b
```

---

## Package Structure

The project is now a proper Python package (`credit_risk/`). All logic lives inside it; the root `.py` files are thin backward-compatible wrappers.

```
credit_risk/
├── config/
│   ├── features.py        # selected / numerical / categorical feature lists
│   └── constants.py       # file names, get_paths(), base_parser()
├── data/
│   └── preprocessing.py   # split_and_save, apply_log_transforms, build_preprocessor
├── models/
│   └── factory.py         # get_model_and_grid (XGBoost / RF)
├── explainability/
│   └── shap_utils.py      # _fraud_shap helper
├── pipeline/
│   ├── train.py           # train()  — CLI: credit-risk-train
│   ├── evaluate.py        # test(), run_shap()  — CLI: credit-risk-evaluate
│   └── predict.py         # predict()  — CLI: credit-risk-predict
├── server/
│   └── mcp_server.py      # FastMCP server  — CLI: credit-risk-server
└── agent/
    └── agent.py           # Ollama agent  — CLI: credit-risk-agent
```

Import anything directly:
```python
from credit_risk.pipeline.train import train
from credit_risk.config.features import selected_features
```

---

## MCP Server

The MCP server exposes all pipeline stages as tools that the AI agent calls. It must be running before you use the agent.

### Start the server (Terminal 2)
```bash
source myenv/bin/activate

# Using the installed entry point (after pip install -e .)
credit-risk-server

# Or using python -m
python -m credit_risk.server.mcp_server

# Or via the legacy root wrapper (still works)
python mcp_server.py

# Listening on http://localhost:8000 (SSE transport)
```

### Debug tools interactively in the browser
```bash
mcp dev credit_risk/server/mcp_server.py
```

### Open a terminal for queries (Terminal 3)
```bash
source myenv/bin/activate
```

---

## CLI Scripts

Use these to run each stage directly without the agent. Each stage can be run three equivalent ways — entry point, `python -m`, or legacy root wrapper.

### Stage 1 — Split + Train

```bash
# Using the installed entry point
credit-risk-train --data_dir .

# Using python -m
python -m credit_risk.pipeline.train --data_dir .

# Legacy root wrapper (still works)
python train.py --data_dir .
```

```bash
# Quick training run to verify everything works (~5 min)
credit-risk-train --data_dir . --fast_mode

# Train Random Forest instead (CPU, no GPU required)
credit-risk-train --data_dir . --model rf

# Retrain without re-splitting (reuse existing train/val/test CSVs)
credit-risk-train --data_dir . --skip_split

# Fast mode + skip split
credit-risk-train --data_dir . --fast_mode --skip_split
```

| Flag | Default | Effect |
|---|---|---|
| `--data_dir` | *(required)* | Folder containing CSVs and where models are saved |
| `--model` | `xgboost` | `xgboost` or `rf` |
| `--fast_mode` | off | One-combo grid search (~5 min) instead of full search |
| `--skip_split` | off | Reuse existing `train_split.csv` / `val_split.csv` / `test_split.csv` |

Training saves two files per model:
- `anomaly_pipeline_<model>.pkl` — the fitted pipeline
- `anomaly_pipeline_<model>.threshold` — optimal threshold (max-F2 on validation set)

---

### Stage 2 — Evaluate

```bash
# Using the installed entry point
credit-risk-evaluate --data_dir . --model xgboost

# Using python -m
python -m credit_risk.pipeline.evaluate --data_dir . --model xgboost

# Legacy root wrapper (still works)
python test.py --data_dir . --model xgboost
```

```bash
# Evaluate Random Forest
credit-risk-evaluate --data_dir . --model rf

# Generate SHAP global feature importance plots (saved to shap_test/)
credit-risk-evaluate --data_dir . --model xgboost --explain

# SHAP with more samples for smoother plots and custom output directory
credit-risk-evaluate --data_dir . --model xgboost --explain --n_global 5000 --output_dir ./my_plots
```

| Flag | Default | Effect |
|---|---|---|
| `--data_dir` | *(required)* | Folder containing `test_split.csv` and the model `.pkl` |
| `--model` | `xgboost` | `xgboost` or `rf` |
| `--explain` | off | Generate SHAP bar, beeswarm, and waterfall plots |
| `--n_global` | `2000` | Samples used for global SHAP plots (fewer = faster) |
| `--output_dir` | `<data_dir>/shap_test/` | Where SHAP PNGs are saved |

Outputs printed: PR-AUC, classification report (precision / recall / F1), confusion matrix.

---

### Stage 3 — Predict

```bash
# Using the installed entry point
credit-risk-predict \
  --model_path ./anomaly_pipeline_xgboost.pkl \
  --input      ./predict_without_target.csv \
  --output     predictions.csv

# Using python -m
python -m credit_risk.pipeline.predict \
  --model_path ./anomaly_pipeline_xgboost.pkl \
  --input      ./predict_without_target.csv \
  --output     predictions.csv

# Legacy root wrapper (still works)
python predict.py \
  --model_path ./anomaly_pipeline_xgboost.pkl \
  --input      ./predict_without_target.csv \
  --output     predictions.csv
```

```bash
# Override threshold (lower = catches more anomalies, higher false-positive rate)
credit-risk-predict \
  --model_path ./anomaly_pipeline_xgboost.pkl \
  --input      ./predict_without_target.csv \
  --output     predictions.csv \
  --threshold  0.3

# Score and generate SHAP waterfall plots for the top 5 flagged transactions
credit-risk-predict \
  --model_path ./anomaly_pipeline_xgboost.pkl \
  --input      ./predict_without_target.csv \
  --output     predictions.csv \
  --explain    --n_explain 5
```

| Flag | Default | Effect |
|---|---|---|
| `--model_path` | *(required)* | Path to the `.pkl` pipeline file |
| `--input` | *(required)* | CSV of new transactions to score |
| `--output` | `predictions.csv` | Path for the output CSV |
| `--threshold` | *(auto)* | Anomaly cutoff; if omitted, loads from `.threshold` file saved during training (falls back to `0.5`) |
| `--explain` | off | Generate SHAP waterfall plots for flagged rows |
| `--n_explain` | `5` | How many flagged rows get waterfall plots |

Output CSV adds two columns to the input: `anomaly_probability` and `anomaly_flag`.

---

## AI Agent Queries

The agent translates natural language into MCP tool calls. The MCP server must be running first.

Default flags:
- `--ollama_model qwen2.5:7b`
- `--mcp_url http://localhost:8000/sse`
- `--data_dir .`

Override any default by appending the relevant flag to the command.

```bash
# Using the installed entry point
credit-risk-agent --query "..."

# Using python -m
python -m credit_risk.agent.agent --query "..."

# Legacy root wrapper (still works)
python agent.py --query "..."
```

### Stage 1 — Train

```bash
# Split data then train XGBoost (full grid search, ~30-90 min)
credit-risk-agent --query "Split the dataset and train XGBoost"

# Quick training run (~5 min)
credit-risk-agent --query "Split the dataset and train XGBoost in fast mode"

# Train Random Forest instead
credit-risk-agent --query "Train a Random Forest model"
```

### Stage 2 — Evaluate

```bash
# Full evaluation report on the held-out test set
credit-risk-agent --query "Evaluate the XGBoost model on the test set"

# Specific metrics
credit-risk-agent --query "What is the ROC-AUC of the XGBoost model?"
credit-risk-agent --query "Show the confusion matrix and anomaly precision/recall for XGBoost"

# SHAP global feature importance (generates bar, beeswarm, waterfall PNGs)
credit-risk-agent --query "Explain the XGBoost model and show the top 10 most important features"
```

### Stage 3 — Predict

```bash
# Score the unlabeled dataset (threshold auto-loaded from training)
credit-risk-agent --query "Score transactions in predict_without_target.csv using XGBoost"

# Override threshold (lower = catches more anomalies)
credit-risk-agent --query "Score predict_without_target.csv with a threshold of 0.3"
```


---

## MCP Server Tools (7 available)

| Tool | Description | Key Parameters |
|---|---|---|
| `split_dataset` | Stratified 60/10/30 train/val/test split | `data_dir` |
| `train_model` | GridSearchCV + SMOTE training | `model_name`, `fast_mode` |
| `evaluate_model` | Full test-set evaluation report | `model_name` |
| `predict_transactions` | Score new transactions for anomalies | `input_csv`, `threshold` |
| `explain_model` | SHAP global feature importance | `model_name`, `n_samples` |
| `get_feature_config` | Returns the full feature schema | — |
| `list_models` | Lists saved `.pkl` model files | `data_dir` |

---

## Models

| Model | Backend | Best for |
|---|---|---|
| `xgboost` | GPU-accelerated (CUDA) | Best accuracy, faster training on GPU |
| `rf` | CPU (Random Forest) | No GPU required, balanced class weights |

---

## Output Files

| File / Directory | Created by | Contents |
|---|---|---|
| `anomaly_pipeline_xgboost.pkl` | `train.py` / `train_model` | Trained XGBoost pipeline |
| `anomaly_pipeline_rf.pkl` | `train.py` / `train_model` | Trained Random Forest pipeline |
| `anomaly_pipeline_<model>.threshold` | `train.py` / `train_model` | Optimal threshold JSON (max-F2 on val) |
| `train_split.csv` | `train.py` / `split_dataset` | 60% training rows |
| `val_split.csv` | `train.py` / `split_dataset` | 10% validation rows |
| `test_split.csv` | `train.py` / `split_dataset` | 30% held-out test rows |
| `predictions.csv` | `predict.py` / `predict_transactions` | Input rows + `anomaly_probability` + `anomaly_flag` |
| `shap_test/` | `test.py --explain` | Bar, beeswarm, waterfall PNGs |
| `shap_agent/` | `explain_model` tool | Bar, beeswarm, waterfall PNGs |
| `predictions_shap/` | `predict.py --explain` | Per-flagged-row waterfall PNGs |

---

## Planned Improvements

This section tracks every engineering and ML improvement needed to bring the project to production quality. Items are ordered by impact and grouped by category.

---

### ~~1. Project Structure — Refactor to a Python Package~~ ✅ Done

**Completed:** the flat script directory has been replaced with the `credit_risk` Python package.

**Implemented structure:**

```
credit_risk/
├── config/
│   ├── features.py        # selected / numerical / categorical feature lists
│   └── constants.py       # file names, get_paths(), base_parser()
├── data/
│   └── preprocessing.py   # split_and_save, apply_log_transforms, build_preprocessor
├── models/
│   └── factory.py         # get_model_and_grid (XGBoost / RF)
├── explainability/
│   └── shap_utils.py      # _fraud_shap helper
├── pipeline/
│   ├── train.py           # train() + main()
│   ├── evaluate.py        # test(), run_shap() + main()
│   └── predict.py         # predict() + main()
├── server/
│   └── mcp_server.py      # FastMCP server + main()
└── agent/
    └── agent.py           # Ollama agent + main()
```

`pyproject.toml` added — install with `pip install -e .` to get five CLI entry points (`credit-risk-train`, `credit-risk-evaluate`, `credit-risk-predict`, `credit-risk-server`, `credit-risk-agent`). Root `.py` files kept as thin backward-compatible wrappers so existing `python train.py` commands still work.

**Why:** enables `from credit_risk.pipeline.train import train`, proper dependency injection, and a clean separation between logic, serving, and agent layers.

---

### 2. Abstract Model Interface

**Current state:** `models.py` returns different objects (`XGBClassifier` vs `RandomForestClassifier`) with no shared contract — callers must know the type.

**Target:** add `credit_risk/models/base.py`:

```python
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class FraudModel(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]: ...
```

**Why:** adding a third model (e.g. LightGBM) currently requires touching `train.py`, `test.py`, `predict.py`, and `mcp_server.py`. With an interface it requires only a new file.

---

### 3. Configuration File

**Current state:** all paths, feature lists, hyperparameter grids, and model choices are hardcoded in `config.py` and `models.py`. Changing an experiment requires editing source code.

**Target:** `config/xgboost.yaml` and `config/rf.yaml`:

```yaml
model:
  type: xgboost
  n_estimators: 500
  max_depth: 6

data:
  source_path: data/train_with_features_target.csv
  test_size: 0.30
  val_size: 0.10
  random_state: 42

threshold:
  metric: f2
  beta: 2.0
```

**Why:** separates code from experiment config; enables running two experiments with different feature sets without touching source files.

---

### 4. Input Data Validation

**Current state:** input CSVs are consumed with no schema enforcement. Missing columns produce cryptic stack traces.

**Target:** add `credit_risk/data/schema.py` using `pandera`:

```python
import pandera as pa

TransactionSchema = pa.DataFrameSchema({
    "TransactionAmt": pa.Column(float, pa.Check.gt(0)),
    "card6": pa.Column(str, nullable=True),
    "P_emaildomain": pa.Column(str, nullable=True),
    # ... all 46 features
})
```

**Why:** surfaces data problems immediately with actionable error messages instead of silent wrong output or a crash 10 steps later.

---

### 5. Type Annotations

**Current state:** most functions have no parameter types or return types.

**Target:** annotate every function signature, e.g.:

```python
# Before
def split_and_save(df, output_dir='.'):
    ...

# After
def split_and_save(
    df: pd.DataFrame,
    output_dir: str | Path = '.',
    test_size: float = 0.30,
    val_size: float = 0.10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ...
```

**Why:** enables `mypy --strict` static analysis, documents contracts, catches bugs before runtime.

---

### 6. Pydantic Data Contracts for the Serving Layer

**Current state:** MCP tool inputs and outputs are untyped dicts.

**Target:** define request/response models:

```python
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    transaction_amt: float = Field(gt=0)
    card6: str
    p_emaildomain: str | None = None

class PredictionResult(BaseModel):
    fraud_probability: float = Field(ge=0.0, le=1.0)
    fraud_flag: bool
    threshold_used: float
```

**Why:** validates inputs at the API boundary, provides auto-generated docs, prevents silent wrong output.

---

### 7. Structured Logging

**Current state:** all status output uses bare `print()` calls.

**Target:** replace every `print()` with the `logging` module in JSON format:

```python
import logging
log = logging.getLogger(__name__)

log.info("training_started", extra={
    "model_type": model_type,
    "n_train_samples": len(X_train),
    "class_balance": float(y_train.mean()),
})
log.info("training_completed", extra={
    "best_params": best_params,
    "val_pr_auc": val_pr_auc,
    "optimal_threshold": threshold,
    "duration_seconds": elapsed,
})
```

**Why:** logs must be machine-readable to be ingested by monitoring systems (Datadog, Stackdriver, CloudWatch). `print()` is invisible to alerting pipelines.

---

### 8. Experiment Tracking (MLflow)

**Current state:** `anomaly_pipeline_xgboost.pkl` is silently overwritten on every training run. There is no record linking model artifacts to their hyperparameters or metrics.

**Target:** log every run with MLflow:

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("val_pr_auc", val_pr_auc)
    mlflow.log_metric("val_f2", val_f2)
    mlflow.log_metric("optimal_threshold", threshold)
    mlflow.sklearn.log_model(pipeline, "model")
```

**Why:** enables reproducibility, model comparison, and artifact lineage — answers "which hyperparameters produced this model?" with evidence.

---

### 9. Domain-Specific Exception Types

**Current state:** errors surface as raw `FileNotFoundError`, `KeyError`, etc. with no context.

**Target:** add `credit_risk/exceptions.py`:

```python
class ModelNotFoundError(Exception):
    """Raised when the trained model artifact cannot be located."""

class InvalidInputSchemaError(Exception):
    """Raised when input DataFrame is missing required columns."""

class ThresholdNotCalibratedError(Exception):
    """Raised when the threshold file is absent or corrupted."""
```

**Why:** callers can catch specific errors and respond appropriately; error messages are actionable instead of exposing internal file paths and class names.

---

### 10. Structured MCP Error Responses

**Current state:** all exceptions are caught and returned as `{"status": "error", "message": str(e)}`, which leaks internal implementation details.

**Target:**

```python
return {
    "status": "error",
    "code": "MODEL_NOT_FOUND",
    "message": "No trained model found. Run train_model first.",
}
```

**Why:** consistent error codes allow agent and client code to handle failures programmatically without parsing freeform strings.

---

### 11. Test Suite

**Current state:** `test.py` is a model evaluation script. There are zero unit or integration tests.

**Target:** a full `tests/` directory covering:

| Test file | What it covers |
|---|---|
| `unit/test_metrics.py` | F2 threshold logic, edge cases (all-negative, perfect model) |
| `unit/test_preprocessing.py` | log transform, imputation, unseen categories, missing columns |
| `unit/test_schema.py` | pandera schema rejects bad inputs, accepts valid inputs |
| `integration/test_train_pipeline.py` | end-to-end: split → train → threshold file written with valid values |
| `integration/test_predict_pipeline.py` | end-to-end: load model → score CSV → output has correct columns |
| `integration/test_mcp_server.py` | FastAPI TestClient: each endpoint returns expected status and keys |

Example:

```python
# tests/unit/test_metrics.py
def test_f2_threshold_prefers_recall():
    y_true = np.array([1, 1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.4, 0.3, 0.2])
    threshold = find_f2_threshold(y_true, y_prob)
    assert threshold <= 0.5  # F2 should pick a low threshold to catch all positives

def test_predict_raises_on_missing_column():
    bad_input = pd.DataFrame({"TransactionAmt": [100]})  # missing 45 features
    with pytest.raises(InvalidInputSchemaError):
        predict(bad_input)
```

**Run tests:**

```bash
pytest tests/ -v --cov=credit_risk --cov-report=term-missing
```

---

### 12. Cross-Validated Threshold

**Current state:** the F2-optimal threshold is computed once on a single fixed validation fold, which may be overfit to that specific split.

**Target:** bootstrap confidence interval across multiple folds:

```python
thresholds = []
for fold_idx in range(n_bootstrap):
    sample = val_df.sample(frac=0.8, replace=True, random_state=fold_idx)
    thresholds.append(find_f2_threshold(sample[target], model.predict_proba(sample)[:, 1]))

threshold_mean = np.mean(thresholds)
threshold_std  = np.std(thresholds)
```

**Why:** quantifies threshold stability and prevents presenting a threshold that happens to be lucky on one validation split.

---

### 13. Model Calibration Evaluation

**Current state:** evaluation reports PR-AUC and confusion matrix but never checks whether the predicted probabilities are calibrated — i.e., whether `fraud_probability = 0.8` actually means 80% of those transactions are fraudulent.

**Target:** add calibration curve to `test.py --explain` output:

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

fraction_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(mean_pred, fraction_pos)
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
```

**Why:** fraud risk scores used for business decisions (block vs. review vs. allow) require calibrated probabilities, not just ranked scores.

---

### 14. Makefile

**Current state:** no single command to run the full pipeline reproducibly from a clean clone.

**Target:**

```makefile
.PHONY: install test lint type-check train evaluate predict clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=credit_risk --cov-report=term-missing

lint:
	ruff check . && black --check .

type-check:
	mypy credit_risk/ --strict

train:
	python -m credit_risk.train --config config/xgboost.yaml

evaluate:
	python -m credit_risk.evaluate --model-dir outputs/latest

predict:
	python -m credit_risk.predict --input predict_without_target.csv --output predictions.csv

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
```

---

### 15. Dockerfile

**Current state:** no containerization. Requires the reviewer to manually match Python version, install CUDA drivers, and install packages.

**Target:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY credit_risk/ credit_risk/
COPY config/ config/

ENTRYPOINT ["python", "-m", "credit_risk.train"]
```

**Why:** guarantees the pipeline runs identically on any machine, removes environment mismatch as a failure mode.

---

### 16. CI Pipeline (GitHub Actions)

**Current state:** one git commit, no automated checks.

**Target:** `.github/workflows/ci.yml` that runs on every push and pull request:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: black --check .
      - run: mypy credit_risk/ --strict
      - run: pytest tests/ -v --cov=credit_risk
```

**Why:** every PR gets automatic lint, type-check, and test validation before merge. A green badge on the README is the first signal a recruiter reads.

---

### 17. `.gitignore`

**Current state:** `myenv/`, `__pycache__/`, and large CSV files are committed to the repository.

**Target:**

```gitignore
myenv/
__pycache__/
*.pyc
*.pkl
*.threshold
*.csv
outputs/
.env
```

**Why:** committing a virtual environment and compiled Python files signals the project has never been worked on collaboratively.

---

### Improvement Summary

| # | Improvement | Category | Effort |
|---|---|---|---|
| 1 | ~~Package structure~~ ✅ | Modularity | ~~Medium~~ |
| 2 | Abstract model interface | Modularity | Small |
| 3 | `config.yaml` | Modularity | Small |
| 4 | Input data validation (pandera) | Reliability | Small |
| 5 | Type annotations (mypy strict) | Code quality | Medium |
| 6 | Pydantic API contracts | Code quality | Medium |
| 7 | Structured logging | Observability | Small |
| 8 | MLflow experiment tracking | Observability | Small |
| 9 | Domain exception types | Error handling | Small |
| 10 | Structured MCP error codes | Error handling | Small |
| 11 | Full test suite (pytest) | Testing | Large |
| 12 | Cross-validated threshold | ML rigor | Medium |
| 13 | Calibration curve | ML rigor | Small |
| 14 | `Makefile` | Developer experience | Small |
| 15 | `Dockerfile` | Portability | Small |
| 16 | GitHub Actions CI | Engineering hygiene | Small |
| 17 | `.gitignore` | Engineering hygiene | Small |
