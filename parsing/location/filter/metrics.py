"""Single/Multiple classification metrics, including unsafe admissions."""

import math

LABELS = ("Single", "Multiple")

def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None

def wilson(successes: int, total: int, z: float = 1.959963984540054) -> dict | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"lower": max(0.0, center - spread), "upper": min(1.0, center + spread)}

def classification_metrics(rows: list[dict]) -> dict:
    predictions = ("Single", "Multiple", "Invalid")
    confusion = {
        gold: {prediction: 0 for prediction in predictions}
        for gold in LABELS
    }
    for row in rows:
        confusion[row["expected_output"]][row["prediction"]] += 1
    total = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    valid = sum(confusion[gold][prediction] for gold in LABELS for prediction in LABELS)
    predicted_single = sum(confusion[gold]["Single"] for gold in LABELS)
    gold_single = sum(confusion["Single"].values())
    gold_multiple = sum(confusion["Multiple"].values())
    true_single = confusion["Single"]["Single"]
    true_multiple = confusion["Multiple"]["Multiple"]
    false_single = confusion["Multiple"]["Single"]
    false_multiple = confusion["Single"]["Multiple"]
    single_precision = rate(true_single, predicted_single)
    single_recall = rate(true_single, gold_single)
    multiple_precision = rate(true_multiple, sum(confusion[g]["Multiple"] for g in LABELS))
    multiple_recall = rate(true_multiple, gold_multiple)

    def f1(precision: float | None, recall: float | None) -> float | None:
        if recall is None:
            return None
        if precision is None:
            return 0.0
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    class_f1 = {
        "Single": f1(single_precision, single_recall),
        "Multiple": f1(multiple_precision, multiple_recall),
    }
    defined_f1 = [value for value in class_f1.values() if value is not None]
    return {
        "items": total,
        "accuracy": rate(correct, total),
        "valid_output_rate": rate(valid, total),
        "balanced_accuracy": (
            (single_recall + multiple_recall) / 2
            if single_recall is not None and multiple_recall is not None
            else None
        ),
        "macro_f1": sum(defined_f1) / len(defined_f1) if defined_f1 else None,
        "single_precision": single_precision,
        "single_precision_wilson_95": wilson(true_single, predicted_single),
        "single_recall": single_recall,
        "multiple_precision": multiple_precision,
        "multiple_recall": multiple_recall,
        "unsafe_false_single": {
            "count": false_single,
            "rate_among_gold_multiple": rate(false_single, gold_multiple),
            "wilson_95": wilson(false_single, gold_multiple),
            "meaning": "Multiple text incorrectly admitted as Single.",
        },
        "conservative_false_multiple": {
            "count": false_multiple,
            "rate_among_gold_single": rate(false_multiple, gold_single),
            "meaning": "Safe Single text rejected; loses training data but does not contaminate it.",
        },
        "class_f1": class_f1,
        "confusion": confusion,
    }
