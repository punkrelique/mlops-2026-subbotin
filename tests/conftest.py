import pytest

from src.config import feature_columns, load_params
from src.data import clean, generate
from src.train import build_pipeline


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def cols(params):
    return feature_columns(params)


@pytest.fixture(scope="session")
def raw_df():
    return generate(n=2000, seed=7)


@pytest.fixture(scope="session")
def trained_pipeline(params, raw_df, cols):
    """scope='session' обязателен: обучение медленное, а тестов много."""
    df = clean(raw_df)
    pipe = build_pipeline(params)
    pipe.fit(df[cols], df["churn"])
    return pipe
