def _fraud_shap(explainer, X):
    """Return (shap_values, expected_value) for fraud class (class 1).

    TreeExplainer returns:
      XGBoost binary  → single ndarray + scalar expected_value
      RandomForest    → list of arrays + list of expected_values (one per class)
    """
    sv = explainer.shap_values(X)
    ev = explainer.expected_value
    if isinstance(sv, list):        # RandomForest
        return sv[1], float(ev[1])
    return sv, float(ev)            # XGBoost binary
