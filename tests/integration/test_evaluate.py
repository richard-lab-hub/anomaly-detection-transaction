from credit_risk.pipeline.evaluate import test as evaluate_test

_EXPECTED_KEYS = {
    'pr_auc', 'accuracy', 'fraud_metrics', 'legit_metrics',
    'confusion_matrix', 'test_samples', 'fraud_rate',
}

_CM_KEYS = {'tn', 'fp', 'fn', 'tp'}


def test_evaluate_returns_expected_keys(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert _EXPECTED_KEYS <= set(result.keys())


def test_evaluate_pr_auc_in_range(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert 0.0 <= result['pr_auc'] <= 1.0


def test_evaluate_accuracy_in_range(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert 0.0 <= result['accuracy'] <= 1.0


def test_evaluate_confusion_matrix_keys(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert set(result['confusion_matrix'].keys()) == _CM_KEYS


def test_evaluate_confusion_matrix_sums_to_test_samples(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    cm = result['confusion_matrix']
    total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
    assert total == result['test_samples']


def test_evaluate_positive_sample_count(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert result['test_samples'] > 0


def test_evaluate_fraud_rate_in_range(trained_paths_module):
    result = evaluate_test(paths=trained_paths_module, model_name='xgboost')
    assert 0.0 <= result['fraud_rate'] <= 1.0
