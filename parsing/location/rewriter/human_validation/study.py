"""Evaluate saved rewrites with the selected reference/formatting policy."""

from pathlib import Path

from parsing.location._validation import aligned, run
from parsing.location.rewriter.metrics import metrics


def evaluate(references, predictions):
    rows = []
    for ref, pred in aligned(references, predictions, ("question", "answer", "original_location", "replacement_location")):
        if not isinstance(ref.get("expected_output"), str) or not ref["expected_output"].strip():
            raise ValueError("Every case requires a non-empty reference rewrite")
        result = pred.get("rewrite", pred)
        rows.append({**ref, **{key: result[key] for key in ("prediction", "status", "validation_errors") if key in result}})
    return {"metrics": metrics(rows), "mismatches": [row for row in rows if row["status"] != "ok" or row["prediction"] != row["expected_output"]]}


if __name__ == "__main__":
    run(Path(__file__).resolve().parent, evaluate)
