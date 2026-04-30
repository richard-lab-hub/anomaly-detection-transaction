from credit_risk.config.features import selected_features, numerical_features, categorical_features


def test_selected_features_count():
    assert len(selected_features) == 66


def test_numerical_features_count():
    assert len(numerical_features) == 53


def test_categorical_features_count():
    assert len(categorical_features) == 13


def test_no_duplicates_in_selected():
    assert len(selected_features) == len(set(selected_features))


def test_no_duplicates_in_numerical():
    assert len(numerical_features) == len(set(numerical_features))


def test_no_duplicates_in_categorical():
    assert len(categorical_features) == len(set(categorical_features))


def test_numerical_and_categorical_disjoint():
    assert set(numerical_features).isdisjoint(set(categorical_features))


def test_numerical_and_categorical_cover_selected():
    assert set(numerical_features) | set(categorical_features) == set(selected_features)


def test_transaction_amt_is_numerical():
    assert 'TransactionAmt' in numerical_features


def test_all_features_are_strings():
    for lst in (selected_features, numerical_features, categorical_features):
        assert all(isinstance(f, str) for f in lst)
