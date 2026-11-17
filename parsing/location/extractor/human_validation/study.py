"""Compare extractor predictions against the reference annotations."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from .sample import DIRECTORY, read_rows


def normalize(text):
    return " ".join(text.split()).casefold()


def tolerant(text):
    text = re.sub(r"\b(?:region|area)\b", "", text, flags=re.IGNORECASE)
    return normalize(re.sub(r"\s*([,;])\s*", r"\1", text))


def wilson(matches, total):
    z = 1.959963984540054
    p = matches / total
    center = (p + z*z / (2*total)) / (1 + z*z / total)
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total**2)) / (1 + z*z/total)
    return {"lower": max(0.0, center-margin), "upper": min(1.0, center+margin)}


def unique(rows):
    result = {}
    for row in rows:
        key = row.get("item_id")
        if not key or key in result:
            raise ValueError(f"Missing or duplicate item_id: {key}")
        result[key] = row
    return result


def evaluate(annotations, predictions):
    references, predictions = unique(annotations), unique(predictions)
    if not references or references.keys() != predictions.keys():
        raise ValueError("Predictions must cover exactly the reference sample")
    scores = {"strict_exact_match": [], "normalized_exact_match": [], "descriptor_tolerant_match": []}
    statuses, mismatches = Counter(), []
    for item_id, row in references.items():
        expected = row.get("expected_output")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"Missing reference label: {item_id}")
        prediction = predictions[item_id]
        for key in ("question", "answer"):
            if prediction.get(key) != row[key]:
                raise ValueError(f"Prediction input differs in {key}: {item_id}")
        result = prediction.get("extraction", prediction)
        status, actual = result["status"], result.get("location")
        statuses[status] += 1
        for name, normalizer in (("strict_exact_match", str.strip), ("normalized_exact_match", normalize), ("descriptor_tolerant_match", tolerant)):
            matches = status == "ok" and isinstance(actual, str) and normalizer(actual) == normalizer(expected)
            if name == "descriptor_tolerant_match" and not tolerant(expected).strip(" ,;"):
                matches = False
            scores[name].append(matches)
        if not scores["strict_exact_match"][-1]:
            mismatches.append({"item_id": item_id, "question": row["question"], "answer": row["answer"],
                               "expected_output": expected, "actual_output": actual, "status": status})
    total = len(references)
    return {
        "items": total,
        "by_model": dict(Counter(row["source_model"] for row in annotations)),
        "reviewed_flags": sum(row.get("reviewed") is True for row in annotations),
        "metrics": {name: {"matches": sum(values), "total": total, "accuracy": sum(values)/total,
                           "wilson_95": wilson(sum(values), total)} for name, values in scores.items()},
        "statuses": dict(statuses), "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DIRECTORY / "annotations.jsonl")
    parser.add_argument("--predictions", type=Path, required=True, help="Extractor JSONL output or a JSON report with cases")
    parser.add_argument("--output", type=Path, default=DIRECTORY / "rescored.json")
    args = parser.parse_args()
    predictions = json.loads(args.predictions.read_text())["cases"] if args.predictions.suffix == ".json" else read_rows(args.predictions)
    result = evaluate(read_rows(args.annotations), predictions)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "items": result["items"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
