"""Стадия evaluate: метрики на отложенном тесте и quality gate.
Запуск: python -m src.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

from src.config import TARGET, data_md5, feature_columns, load_params, resolve, update_data_stats
from src.logging_setup import setup_logging

log = setup_logging()


def main() -> None:
    params = load_params()
    d = params["data"]
    gate = params["evaluate"]["min_roc_auc"]
    threshold = params["evaluate"]["threshold"]
    model_path = resolve("models/model.joblib")

    log.info(f"Загружаю модель из {model_path}")
    model = joblib.load(model_path)
    processed_dir = resolve(d["processed_dir"])
    test_df = pd.read_csv(processed_dir / "test.csv")

    log.info(f"test: {len(test_df)} строк, {(test_df[TARGET] == 1).mean():.1%} оттока")
    cols = feature_columns(params)

    X_test = test_df[cols]
    y_test = test_df[TARGET]
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    log.info(f"✅ ROC-AUC: {roc_auc:.4f}")
    log.info(f"✅ PR-AUC:  {pr_auc:.4f}")
    log.info(f"✅ F1:      {f1:.4f}")

    metrics = {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1),
    }
    reports_dir = Path(resolve("reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "eval_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    stats_path = update_data_stats("eval", {"data_md5": data_md5(d["raw_path"])})
    log.info(f"Метрики сохранены в {metrics_path} и {stats_path}")

    if metrics["roc_auc"] < gate:
        log.error("ROC-AUC %.4f ниже порога %.4f", metrics["roc_auc"], gate)
        sys.exit(1)


if __name__ == "__main__":
    main()
