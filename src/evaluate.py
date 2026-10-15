"""Стадия evaluate: метрики на отложенном тесте и quality gate.
Запуск: python -m src.evaluate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score, roc_curve

from src.config import TARGET, data_md5, feature_columns, load_params, resolve, update_data_stats
from src.logging_setup import setup_logging

log = setup_logging()

SLICES = ["contract_type", "internet_service", "tenure_bucket"]
TENURE_BINS = [0, 12, 24, float("inf")]
TENURE_LABELS = ["первый год (1-12)", "второй год (13-24)", "старожилы (25+)"]


def segment_quality(model, df: pd.DataFrame, cols: list) -> tuple:
    df["tenure_bucket"] = pd.cut(df["tenure_months"], bins=TENURE_BINS, labels=TENURE_LABELS)
    df["_proba"] = model.predict_proba(df[cols])[:, 1]
    overall = roc_auc_score(df[TARGET], df["_proba"])
    rows = []
    for slice_col in SLICES:
        for segment, group in df.groupby(slice_col, observed=True):
            score = roc_auc_score(group[TARGET], group["_proba"]) if group[TARGET].nunique() > 1 else None
            rows.append(
                {
                    "slice": slice_col,
                    "segment": str(segment),
                    "n": int(len(group)),
                    "roc_auc": score,
                    "delta": None if score is None else score - overall,
                }
            )
    return overall, rows


def segment_gate_failures(rows: list, min_auc: float, min_size: int) -> list:
    return [r for r in rows if r["n"] > min_size and r["roc_auc"] is not None and r["roc_auc"] < min_auc]


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

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    step = max(1, len(fpr) // 200)
    roc_path = reports_dir / "roc.json"
    with open(roc_path, "w", encoding="utf-8") as f:
        json.dump([{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr[::step], tpr[::step])], f)

    _, segments = segment_quality(model, test_df, cols)
    segments_path = reports_dir / "segment_metrics.json"
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(
            {f"{r['slice']}={r['segment']}": {"n": r["n"], "roc_auc": r["roc_auc"]} for r in segments},
            f,
            indent=2,
            ensure_ascii=False,
        )

    stats_path = update_data_stats("eval", {"data_md5": data_md5(d["raw_path"])})
    log.info(f"Метрики сохранены в {metrics_path}, {segments_path} и {stats_path}")

    failed = False
    if metrics["roc_auc"] < gate:
        log.error("ROC-AUC %.4f ниже порога %.4f", metrics["roc_auc"], gate)
        failed = True

    ev = params["evaluate"]
    for r in segment_gate_failures(segments, ev["min_roc_auc_segment"], ev["min_segment_size"]):
        log.error(
            "ROC-AUC %.4f в сегменте %s=%s (n=%d) ниже порога %.4f",
            r["roc_auc"],
            r["slice"],
            r["segment"],
            r["n"],
            ev["min_roc_auc_segment"],
        )
        failed = True

    if failed:
        log.error("Хотя бы в одном сегменте размером больше 200 объектов ROC-AUC ниже min_roc_auc_segment")
        sys.exit(1)


if __name__ == "__main__":
    main()
