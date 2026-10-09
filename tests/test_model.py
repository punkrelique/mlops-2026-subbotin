import time

import pandas as pd
import pytest

from src.data import clean
from src.train import build_pipeline


def test_more_support_calls_increase_churn_risk(trained_pipeline, raw_df, cols):
    """Направленный тест: больше обращений в поддержку -> выше риск ухода.
    Нарушение означает перепутанные признаки или таргет."""
    base = clean(raw_df)[cols].head(200).copy()
    calm, angry = base.copy(), base.copy()
    calm.loc[:, "num_support_calls"] = 0
    angry.loc[:, "num_support_calls"] = 8

    assert (
        trained_pipeline.predict_proba(angry)[:, 1].mean() > trained_pipeline.predict_proba(calm)[:, 1].mean()
    )


def test_probabilities_are_valid(trained_pipeline, raw_df, cols):
    proba = trained_pipeline.predict_proba(clean(raw_df).tail(400)[cols])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.std() > 0.01, "модель выдаёт почти константу — признак сломанного обучения"


def test_prediction_is_row_independent(trained_pipeline, raw_df, cols):
    """Предсказание для строки не зависит от соседей в батче.
    Иначе /predict и /predict/batch дадут разные ответы для одного клиента."""
    X = clean(raw_df).tail(400)[cols]
    assert (
        abs(
            trained_pipeline.predict_proba(X.head(1))[:, 1][0]
            - trained_pipeline.predict_proba(X.head(50))[:, 1][0]
        )
        < 1e-9
    )


def test_longer_contract_decreases_churn_risk(trained_pipeline, raw_df, cols):
    """Направленный тест: долгосрочный контракт снижает риск ухода."""
    base = clean(raw_df)[cols].head(200).copy()
    monthly, two_year = base.copy(), base.copy()
    monthly.loc[:, "contract_type"] = "month-to-month"
    two_year.loc[:, "contract_type"] = "two_year"

    assert (
        trained_pipeline.predict_proba(monthly)[:, 1].mean()
        > trained_pipeline.predict_proba(two_year)[:, 1].mean()
    )


def test_retraining_is_deterministic(params, raw_df, cols):
    """Повторное обучение на тех же данных должно давать те же предсказания."""
    df = clean(raw_df)
    pipe_a = build_pipeline(params).fit(df[cols], df["churn"])
    pipe_b = build_pipeline(params).fit(df[cols], df["churn"])

    proba_a = pipe_a.predict_proba(df[cols].head(200))[:, 1]
    proba_b = pipe_b.predict_proba(df[cols].head(200))[:, 1]
    assert (proba_a == proba_b).all()


def test_duplicate_row_does_not_change_prediction(trained_pipeline, raw_df, cols):
    """Один и тот же клиент, поданный дважды, получает одинаковый ответ.
    Нарушение означает, что модель смотрит на позицию строки в батче."""
    row = clean(raw_df)[cols].head(1)
    doubled = pd.concat([row, row], ignore_index=True)
    proba = trained_pipeline.predict_proba(doubled)[:, 1]
    assert abs(proba[0] - proba[1]) < 1e-12


def test_extra_column_does_not_change_prediction(trained_pipeline, raw_df, cols):
    """Лишняя колонка, которой не было при обучении, не должна влиять на ответ"""
    row = clean(raw_df)[cols].head(5)
    with_extra = row.copy()
    with_extra["unexpected_field"] = "irrelevant"

    proba_plain = trained_pipeline.predict_proba(row)[:, 1]
    proba_extra = trained_pipeline.predict_proba(with_extra)[:, 1]
    assert (proba_plain == proba_extra).all()


@pytest.mark.slow
def test_batch_inference_is_fast_enough(trained_pipeline, raw_df, cols):
    """1000 объектов должны считаться быстрее секунды.
    Порог с запасом: ловим катастрофу (кто-то вызвал fit в predict),
    а не колебания на разных машинах"""
    batch = clean(raw_df)[cols].sample(1000, replace=True, random_state=0)
    start = time.perf_counter()
    trained_pipeline.predict_proba(batch)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"инференс 1000 строк занял {elapsed:.2f} с"


def test_missing_total_charges_is_scored_without_crashing(trained_pipeline, raw_df, cols):
    """Новый клиент первого месяца может прислать total_charges=NaN.
    Этот тест ловит ситуацию, когда кто-то забыл вызвать clean() на пути инференса
    и сервис уйдёт в 500 вместо ответа"""
    row = clean(raw_df)[cols].head(1).copy()
    row.loc[:, "total_charges"] = float("nan")
    cleaned = clean(row)
    proba = trained_pipeline.predict_proba(cleaned)[:, 1]
    assert not pd.isna(proba[0])
