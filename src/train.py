"""Стадия train.
Запуск: python -m src.train
"""

from __future__ import annotations

from src.config import load_params
from src.logging_setup import setup_logging

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

    from src.config import TARGET, feature_columns, resolve
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

    log.info(f"Метрики сохранены в {metrics_path}")


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


if __name__ == "__main__":
    main()
