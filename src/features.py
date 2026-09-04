"""Построение препроцессора признаков.

TODO (занятие 1): собрать здесь ColumnTransformer.

Требования:
  * числовые признаки: заполнение пропусков + масштабирование;
  * категориальные: заполнение пропусков + OneHotEncoder;
  * бинарные: без изменений;
  * списки колонок берутся из params.yaml, а не пишутся в коде.

Подсказка: почему препроцессор обязан ехать в одном Pipeline с моделью,
разбирается на паре. Если сделать иначе — сервис на занятии 10 сломается.
"""
from __future__ import annotations

from typing import Any


def build_preprocessor(params: dict[str, Any]):
    """Собирает ColumnTransformer для разных типов признаков."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    f = params["features"]

    # Для числовых: заполнение пропусков медианой + масштабирование
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Для категориальных: заполнение пропусков модой + OneHot
    # handle_unknown="ignore" важен: если на сервере появится новая категория,
    # сервис не упадёт, а вернёт нули
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Собираем всё вместе
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, f["numeric"]),
        ("cat", categorical_pipe, f["categorical"]),
        ("bin", "passthrough", f["binary"]),  # бинарные не обрабатываем
    ])

    return preprocessor
