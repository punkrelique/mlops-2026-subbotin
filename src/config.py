"""Загрузка params.yaml и доступ к настройкам.

Один модуль-точка входа для конфигурации: скрипты не читают YAML сами
и не хранят пути внутри себя. Это то, ради чего на занятии 3 всё выносилось
из кода — здесь видно результат.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = Path(os.getenv("PARAMS_PATH", PROJECT_ROOT / "params.yaml"))


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Читает params.yaml. Кэш не нужен: файл маленький, а неявный кэш путает."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(relative: str) -> Path:
    """Путь из конфига -> абсолютный путь от корня проекта.

    Без этого проект работает только если запускать его из корня —
    классическая причина 'у меня локально работало'.
    """
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def feature_columns(params: dict[str, Any]) -> list:
    f = params["features"]
    return list(f["numeric"]) + list(f["categorical"]) + list(f["binary"])


def data_md5(raw_path: str) -> str:
    """Хеш версии данных из .dvc файла, привязывает метрики к версии данных."""
    dvc_path = resolve(f"{raw_path}.dvc")
    with open(dvc_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]


def update_data_stats(section: str, data: dict[str, Any]) -> Path:
    """Мерджит метрики стадии в общий reports/data_stats.json по ключу."""
    stats_path = resolve("reports/data_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    if stats_path.exists():
        with open(stats_path, encoding="utf-8") as f:
            stats = json.load(f)
    stats[section] = data
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats_path


TARGET = "churn"
