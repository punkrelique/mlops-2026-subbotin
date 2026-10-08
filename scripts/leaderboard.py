"""Собирает топ прогонов из MLFlow и пишет reports/LEADERBOARD.md.

Запуск: python -m scripts.leaderboard
"""

from pathlib import Path

import mlflow
from mlflow.entities import Run
from pandas import DataFrame

from src.config import load_params, resolve
from src.logging_setup import setup_logging

FILENAME = "LEADERBOARD.md"

COLS = {
    "params.model": "model",
    "metrics.roc_auc": "roc_auc",
    "metrics.pr_auc": "pr_auc",
    "tags.git_sha": "git_sha",
    "start_time": "date",
}

log = setup_logging()
params = load_params()


def main() -> None:
    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    runs = mlflow.search_runs(
        experiment_names=[cfg["experiment_name"]],
        order_by=["metrics.roc_auc DESC"],
        max_results=5,
        output_format="pandas",
    )

    table = (
        runs.reindex(columns=list(COLS)).rename(columns=COLS).fillna("—").replace({"unknown": "—", "": "—"})
    )

    dump_to_markdown(table)


def dump_to_markdown(runs: list[Run] | DataFrame) -> None:
    reports_dir = Path(resolve("reports"))
    leaderboard_path = reports_dir / FILENAME
    md = runs.to_markdown()

    with open(leaderboard_path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
