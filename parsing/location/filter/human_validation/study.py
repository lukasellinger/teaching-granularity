"""Evaluate the frozen safety threshold without fitting it on reference labels."""

import math
from pathlib import Path

from parsing.location._validation import aligned, run
from parsing.location.filter.inference import SETTINGS
from parsing.location.filter.metrics import classification_metrics


def evaluate(references, predictions):
    thresholds = {pred.get("filter", pred)["multiple_log_odds_threshold"] for pred in predictions if "multiple_log_odds_threshold" in pred.get("filter", pred)}
    if not thresholds:
        thresholds = {SETTINGS["multiple_log_odds_threshold"]}
    if len(thresholds) != 1 or not math.isfinite(next(iter(thresholds))):
        raise ValueError("Predictions must use one finite frozen threshold")
    threshold = next(iter(thresholds))
    rows = []
    for ref, pred in aligned(references, predictions, ("question", "answer", "event_extraction")):
        if ref["expected_output"] not in {"Single", "Multiple"}:
            raise ValueError("Every reference needs a Single or Multiple label")
        result = pred.get("filter", pred)
        odds = result.get("multiple_log_odds")
        valid = result.get("filter_status", "ok") == "ok" and isinstance(odds, (float, int)) and math.isfinite(odds)
        prediction = ("Multiple" if odds >= threshold else "Single") if valid else "Invalid"
        rows.append({**ref, "prediction": prediction})
    return {"multiple_log_odds_threshold": threshold,
            "metrics": classification_metrics(rows),
            "mismatches": [row for row in rows if row["prediction"] != row["expected_output"]]}


if __name__ == "__main__":
    run(Path(__file__).resolve().parent, evaluate)
