"""Стадия train.
Запуск: python -m src.train
"""

from __future__ import annotations

import argparse

from src.config import load_params
from src.logging_setup import setup_logging

parser = argparse.ArgumentParser()
parser.add_argument("--no-mlflow", action="store_true")
args = parser.parse_args()

log = setup_logging()


def main() -> None:
    params = load_params()

    # Импорты
    import json
    from pathlib import Path

    import joblib
    import pandas as pd
    from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
    from sklearn.pipeline import Pipeline

    from src.config import TARGET, data_md5, feature_columns, resolve, update_data_stats
    from src.features import build_preprocessor

    # 1. Загружаем данные
    log.info("Загружаю данные...")
    d = params["data"]
    processed_dir = resolve(d["processed_dir"])

    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")

    log.info(f"train: {len(train_df)} строк, {(train_df[TARGET] == 1).mean():.1%} оттока")
    log.info(f"val  : {len(val_df)} строк, {(val_df[TARGET] == 1).mean():.1%} оттока")

    # 2. Признаки и целевая переменная
    cols = feature_columns(params)
    X_train = train_df[cols]
    y_train = train_df[TARGET]
    X_val = val_df[cols]
    y_val = val_df[TARGET]

    # 3. Строим Pipeline (препроцессор + модель)
    log.info("Построение Pipeline...")
    log.info(f"Модель {params['train']['model']}")
    pipe = Pipeline(
        [
            ("preprocess", build_preprocessor(params)),
            ("model", build_model(params)),
        ]
    )

    # 4. Обучаем
    log.info("Обучение...")
    pipe.fit(X_train, y_train)

    # 5. Предсказываем на валидации
    y_pred_proba = pipe.predict_proba(X_val)[:, 1]
    y_pred = pipe.predict(X_val)

    # 6. Считаем метрики
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    f1 = f1_score(y_val, y_pred)

    # Считаем PR-AUC
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)

    log.info(f"✅ ROC-AUC: {roc_auc:.4f}")
    log.info(f"✅ PR-AUC:  {pr_auc:.4f}")
    log.info(f"✅ F1:      {f1:.4f}")

    # 7. Сохраняем модель
    models_dir = Path(resolve("models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    joblib.dump(pipe, model_path)
    log.info(f"Модель сохранена в {model_path}")

    meta_path = models_dir / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"model": params["train"]["model"], "features": cols}, f, indent=2)

    # 8. Сохраняем метрики в JSON
    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
    }

    reports_dir = Path(resolve("reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "train_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    stats_path = update_data_stats("train", {"data_md5": data_md5(d["raw_path"])})
    log.info(f"Метрики сохранены в {metrics_path} и {stats_path}")

    if params["mlflow"]["enabled"] and not args.no_mlflow:
        log_to_mlflow(params, pipe, metrics, train_df[cols].head(5))


def build_model(params: dict) -> any:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    model = params["train"]["model"]
    cfg = params["train"].get(model, {})
    if model == "logreg":
        return LogisticRegression(**cfg)
    elif model == "random_forest":
        return RandomForestClassifier(**cfg)
    elif model == "gradient_boosting":
        return GradientBoostingClassifier(**cfg)
    else:
        raise ValueError("Model is not specified.")


def log_to_mlflow(params, pipe, metrics, input_example) -> None:
    import mlflow
    import mlflow.sklearn

    log.info("Запись в mlflow")

    cfg = params["mlflow"]
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])

    name = params["train"]["model"]
    with mlflow.start_run():
        mlflow.log_params({"model": name, "seed": params["seed"]})
        mlflow.log_params({f"{name}.{k}": v for k, v in params["train"][name].items()})
        mlflow.log_metrics(metrics)
        mlflow.set_tag("git_sha", git_sha())
        mlflow.log_artifact("params.yaml")
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=input_example)


def git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
