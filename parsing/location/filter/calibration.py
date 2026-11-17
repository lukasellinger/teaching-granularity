"""Select the filter threshold on development scores, never held-out labels."""

import math

from .metrics import classification_metrics


def select_threshold(rows, minimum_multiple_recall=0.975):
    if not 0 < minimum_multiple_recall <= 1:
        raise ValueError("minimum_multiple_recall must be in (0, 1]")
    if not rows or any(row.get("filter_status", "ok") != "ok" or not math.isfinite(row["multiple_log_odds"]) for row in rows):
        raise ValueError("Calibration needs finite, successful scores for every dev case")
    if {row["expected_output"] for row in rows} != {"Single", "Multiple"}:
        raise ValueError("Calibration needs both dev classes")

    def metrics(threshold):
        return classification_metrics([{**row, "prediction": "Multiple" if row["multiple_log_odds"] >= threshold else "Single"} for row in rows])
    eligible = [(threshold, metrics(threshold)) for threshold in sorted({row["multiple_log_odds"] for row in rows})]
    eligible = [(threshold, result) for threshold, result in eligible if result["multiple_recall"] >= minimum_multiple_recall]
    threshold, _ = max(eligible, key=lambda pair: (pair[1]["single_recall"], pair[1]["balanced_accuracy"], pair[0]))
    frozen = round(threshold, 6)
    # Preserve safety at a boundary if rounding moves it above a tied score.
    if metrics(frozen)["multiple_recall"] < minimum_multiple_recall:
        frozen = threshold
    return frozen, metrics(frozen)
