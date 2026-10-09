"""
Cравнивает кандидата с текущей продовой версией на тестовой выборке и переводит в Production,
только если ROC-AUC выше хотя бы на 0.005.

Запуск: python -m scripts.promote
"""

import sys

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score

from src.config import TARGET, feature_columns, load_params, resolve
from src.logging_setup import setup_logging

MIN_IMPROVEMENT = 0.005
CHAMPION_ALIAS = "champion"
CANDIDATE_ALIAS = "candidate"

log = setup_logging()
params = load_params()


def main() -> None:
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    client = mlflow.MlflowClient()
    name = params["mlflow"]["registered_model_name"]

    try:
        candidate = client.get_model_version_by_alias(name, CANDIDATE_ALIAS)
    except MlflowException:
        log.error(f"У модели {name} нет версии с алиасом '{CANDIDATE_ALIAS}' - нечего промоутить")
        sys.exit(1)

    X_test, y_test = load_test_set()
    candidate_auc = evaluate_version(name, candidate.version, X_test, y_test)
    log.info(f"Candidate (версия {candidate.version}) ROC-AUC: {candidate_auc:.4f}")

    try:
        champion = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None

    if champion is None:
        log.info("Текущей продовой версии нет - переводим кандидата в Production")
        promote(client, name, candidate.version)
        return

    champion_auc = evaluate_version(name, champion.version, X_test, y_test)
    log.info(f"Champion (версия {champion.version}) ROC-AUC: {champion_auc:.4f}")

    delta = candidate_auc - champion_auc
    if delta >= MIN_IMPROVEMENT:
        log.info(f"Прирост {delta:.4f} >= {MIN_IMPROVEMENT} - переводим кандидата в Production")
        promote(client, name, candidate.version)
    else:
        log.info(f"Прирост {delta:.4f} меньше порога {MIN_IMPROVEMENT} - кандидат остаётся без изменений")


def load_test_set() -> tuple[pd.DataFrame, pd.Series]:
    processed_dir = resolve(params["data"]["processed_dir"])
    test_df = pd.read_csv(processed_dir / "test.csv")
    cols = feature_columns(params)
    return test_df[cols], test_df[TARGET]


def evaluate_version(name: str, version: str, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_pred_proba)


def promote(client: mlflow.MlflowClient, name: str, version: str) -> None:
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    client.delete_registered_model_alias(name, CANDIDATE_ALIAS)


if __name__ == "__main__":
    main()
