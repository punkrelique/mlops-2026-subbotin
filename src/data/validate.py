"""Стадия validate.
Запуск: python -m src.validate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from src.config import TARGET, load_params, resolve
from src.logging_setup import setup_logging

log = setup_logging()
params = load_params()

CHECKS = {
    "строк не меньше минимума": lambda df, p: len(df) >= p["min_rows"],
    "доля пропусков в норме": lambda df, p: df.isna().mean().max() <= p["max_missing_share"],
    "доля оттока осмысленна": lambda df, p: p["target_rate"][0] < df[TARGET].mean() < p["target_rate"][1],
    "нет дублей по клиенту": lambda df, p: not df["customer_id"].duplicated().any(),
    "стаж в допустимом диапазоне": lambda df, p: df["tenure_months"].between(0, 200).all(),
}


def main() -> None:
    df = load_train_df()
    validate_result = validate(df)
    write_report(validate_result)

    if not all(validate_result.values()):
        sys.exit(1)


def load_train_df() -> DataFrame:
    log.info("Загружаю данные...")
    d = params["data"]
    processed_dir = resolve(d["processed_dir"])
    train_df = pd.read_csv(processed_dir / "train.csv")
    return train_df


def validate(df: DataFrame) -> dict[str, bool]:
    log.info("Валидация данных..")
    validate_opts = params["validate"]
    log.info(f"Параметры валидации: {validate_opts}")

    results: dict[str, bool] = {}
    for name, check in CHECKS.items():
        try:
            ok = bool(check(df, validate_opts))
        except Exception as e:
            ok = False
            log.error(f"{name}: raised {type(e).__name__}: {e}")
        else:
            log.info(f"[{'OK' if ok else 'FAIL'}] {name}")
        results[name] = ok

    return results


def write_report(validate_result: dict[str, bool]) -> None:
    report_dir = Path(resolve("reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation.json"

    with open(report_path, "w") as f:
        json.dump(validate_result, f, indent=2)

    log.info(f"Отчёт записан: {report_path}")


if __name__ == "__main__":
    main()
