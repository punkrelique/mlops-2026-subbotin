import argparse
import json

import pandas as pd

from src.config import TARGET, load_params, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", type=str)
    args = parser.parse_args()
    params = load_params()

    currentPath = resolve(args.current)
    currentDf = pd.read_csv(currentPath)

    referencePath = resolve(params["data"]["raw_path"])
    referenceDf = pd.read_csv(referencePath)

    features = (
        params["features"]["numeric"] + params["features"]["categorical"] + params["features"]["binary"]
    )

    report = build_report(referenceDf, currentDf, features)
    summarize(report)


def build_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list):
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    presets = [DataDriftPreset()]
    if TARGET in reference.columns and TARGET in current.columns:
        presets.append(TargetDriftPreset())

    keep = [c for c in columns + [TARGET] if c in reference.columns and c in current.columns]
    report = Report(metrics=presets)
    report.run(reference_data=reference[keep], current_data=current[keep])
    report.save_html("reports/drift.html")  # для человека: посмотреть, разобраться

    summary = summarize(report)  # для автоматики: принять решение
    json.dump(summary, resolve("reports/drift.json", "w"), indent=2)

    return report


def summarize(report) -> dict:
    """Достаём из отчёта то немногое, по чему принимается решение."""
    payload = report.as_dict()
    result = payload["metrics"][0]["result"]
    by_column = result.get("drift_by_columns", {})
    return {
        "dataset_drift": bool(result.get("dataset_drift", False)),
        "share_of_drifted_columns": float(result.get("share_of_drifted_columns", 0.0)),
        "n_drifted_columns": int(result.get("number_of_drifted_columns", 0)),
        "drifted_columns": sorted(n for n, i in by_column.items() if i.get("drift_detected")),
    }


if __name__ == "__main__":
    main()
