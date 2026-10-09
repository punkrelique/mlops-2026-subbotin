import pandas as pd

from src.config import TARGET, feature_columns
from src.data import clean, generate, split


def test_schema_is_complete(raw_df, params):
    """Все колонки из params.yaml есть в данных."""
    missing = set(feature_columns(params) + [TARGET]) - set(raw_df.columns)
    assert not missing, f"в датасете нет колонок: {missing}"


def test_target_is_not_degenerate(raw_df):
    """Защита от 'все нули': такой датасет даст accuracy 0.9 и бесполезную модель."""
    rate = raw_df[TARGET].mean()
    assert 0.05 < rate < 0.60, f"подозрительный churn rate: {rate:.3f}"


def test_generation_is_reproducible():
    a, b = generate(n=500, seed=123), generate(n=500, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_clean_removes_missing_values(raw_df):
    assert raw_df["total_charges"].isna().any(), "в сырых данных ожидаются пропуски"
    assert not clean(raw_df)["total_charges"].isna().any()


def test_no_duplicate_customer_ids(raw_df):
    """Дубликат customer_id означает нарушенную уникальность ключа:
    один и тот же клиент не должен попасть в выборку дважды."""
    assert not raw_df["customer_id"].duplicated().any()


def test_split_is_deterministic(raw_df, params):
    df = clean(raw_df)
    train_a, val_a, test_a = split(df, params)
    train_b, val_b, test_b = split(df, params)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(val_a, val_b)
    pd.testing.assert_frame_equal(test_a, test_b)
