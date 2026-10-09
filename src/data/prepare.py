"""Стадия prepare: сырой CSV -> train/val/test."""

from __future__ import annotations

from pandas import DataFrame

from src.config import load_params
from src.logging_setup import setup_logging

log = setup_logging()


def clean(df: DataFrame) -> DataFrame:
    """Заполняет пропуски в total_charges: monthly_charges * tenure_months."""
    df = df.copy()
    if df["total_charges"].isna().any():
        mask = df["total_charges"].isna()
        df.loc[mask, "total_charges"] = df.loc[mask, "monthly_charges"] * df.loc[mask, "tenure_months"]
        log.info(f"Заполнены пропуски в total_charges: {mask.sum()} строк")
    return df


def split(df: DataFrame, params: dict) -> tuple[DataFrame, DataFrame, DataFrame]:
    from sklearn.model_selection import train_test_split

    d = params["data"]
    seed = params["seed"]

    # Сначала отделяем тест
    train_val, test = train_test_split(df, test_size=d["test_size"], random_state=seed, stratify=df["churn"])

    # Потом отделяем валидацию от train
    # val_size задан в долях от всего датасета, а отделяем от остатка
    val_ratio = d["val_size"] / (1.0 - d["test_size"])
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=seed, stratify=train_val["churn"]
    )
    return train, val, test


def main() -> None:
    params = load_params()
    d = params["data"]

    # 1. Читаем raw данные
    log.info(f"Загружаю данные из {d['raw_path']}")
    import pandas as pd

    from src.config import resolve

    raw = pd.read_csv(resolve(d["raw_path"]))
    log.info(f"Загружено {len(raw)} строк, {len(raw.columns)} колонок")

    # 2. Обработка пропусков в total_charges
    raw = clean(raw)

    # 3. Разбиваем на train/val/test
    train, val, test = split(raw, params)

    # 4. Логируем размеры и долю оттока
    for name, df in [("train", train), ("val", val), ("test", test)]:
        churn_rate = (df["churn"] == 1).mean()
        log.info(f"{name:5s}: {len(df):6d} строк, {churn_rate:.1%} оттока")

    # 5. Сохраняем
    from pathlib import Path

    from src.config import resolve

    processed_dir = Path(resolve(d["processed_dir"]))
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    log.info(f"Сохранены в {processed_dir}/")
    log.info("✅ Подготовка завершена")


if __name__ == "__main__":
    main()
