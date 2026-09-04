"""Стадия prepare: сырой CSV -> train/val/test.

TODO (занятие 1):
  1. прочитать data/raw/churn.csv;
  2. обработать пропуски в total_charges осмысленно (не dropna!);
  3. разбить на train/val/test со stratify по churn и random_state из params;
  4. сохранить три CSV в data/processed/.

Проверка: два запуска подряд должны дать одинаковые файлы.
"""
from __future__ import annotations

from src.config import load_params
from src.logging_setup import setup_logging

log = setup_logging()


def main() -> None:
    params = load_params()
    d = params["data"]
    seed = params["seed"]

    # 1. Читаем raw данные
    log.info(f"Загружаю данные из {d['raw_path']}")
    from src.config import resolve
    import pandas as pd
    from sklearn.model_selection import train_test_split

    raw = pd.read_csv(resolve(d["raw_path"]))
    log.info(f"Загружено {len(raw)} строк, {len(raw.columns)} колонок")

    # 2. Обработка пропусков в total_charges
    # Пропуски у клиентов первого месяца — заполняем как monthly_charges * tenure_months
    if raw["total_charges"].isna().any():
        mask = raw["total_charges"].isna()
        raw.loc[mask, "total_charges"] = raw.loc[mask, "monthly_charges"] * raw.loc[mask, "tenure_months"]
        log.info(f"Заполнены пропуски в total_charges: {mask.sum()} строк")

    # 3. Разбиваем на train/val/test
    # Сначала отделяем тест
    train_val, test = train_test_split(
        raw,
        test_size=d["test_size"],
        random_state=seed,
        stratify=raw["churn"]
    )

    # Потом отделяем валидацию от train
    # val_size задан в долях от всего датасета, а отделяем от остатка
    val_ratio = d["val_size"] / (1.0 - d["test_size"])
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_val["churn"]
    )

    # 4. Логируем размеры и долю оттока
    for name, df in [("train", train), ("val", val), ("test", test)]:
        churn_rate = (df["churn"] == 1).mean()
        log.info(f"{name:5s}: {len(df):6d} строк, {churn_rate:.1%} оттока")

    # 5. Сохраняем
    from src.config import resolve
    from pathlib import Path

    processed_dir = Path(resolve(d["processed_dir"]))
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(processed_dir / "train.csv", index=False)
    val.to_csv(processed_dir / "val.csv", index=False)
    test.to_csv(processed_dir / "test.csv", index=False)

    log.info(f"Сохранены в {processed_dir}/")
    log.info("✅ Подготовка завершена")


if __name__ == "__main__":
    main()
