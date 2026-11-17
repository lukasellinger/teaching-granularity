"""Analyze the temporal component audit and replay its parser predictions."""

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from parsing.temporal import parse_temporal_response

DIRECTORY = Path(__file__).resolve().parent
REVIEWS = (
    "span_review", "completeness_review", "event_anchoring_review",
    "normalization_review", "granularity_review", "multiple_values_review",
    "overall_pipeline_review", "wikidata_entity_review", "wikidata_completion_review",
)
LABELS = {"Correct", "Incorrect", "Not applicable", "Cannot determine"}
ANSWER_STATUSES = {
    "One clear requested value", "Multiple requested values", "No requested value",
    "Abstention", "Cannot determine",
}
JSON_FIELDS = (
    "raw_extracted_mentions", "normalized_relevant_candidates", "final_prediction",
)


def index_unique(rows, key):
    result = {}
    for row in rows:
        identity = row.get(key)
        if not identity or identity in result:
            raise ValueError(f"Missing or duplicate {key}: {identity!r}")
        result[identity] = row
    return result


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def review_summary(labels):
    counts = Counter(labels)
    total = counts["Correct"] + counts["Incorrect"]
    return {
        "correct": counts["Correct"], "incorrect": counts["Incorrect"],
        "not_applicable": counts["Not applicable"],
        "cannot_determine": counts["Cannot determine"],
        "rate": counts["Correct"] / total if total else None,
        "wilson_95": wilson(counts["Correct"], total),
    }


def analyze(tasks, cases, annotations):
    tasks = index_unique([task["data"] for task in tasks], "item_id")
    cases = index_unique(cases, "item_id")
    rows = index_unique(annotations, "item_id")
    index_unique(tasks.values(), "task_uid")
    index_unique(rows.values(), "task_uid")
    if not tasks or tasks.keys() != cases.keys() or tasks.keys() != rows.keys():
        raise ValueError("Tasks, cases and annotations must cover exactly the same non-empty sample")
    annotators = {row.get("annotator") for row in rows.values()}
    if len(annotators) != 1 or not next(iter(annotators)):
        raise ValueError("This audit requires one annotator with one annotation per task")

    changed = []
    for item_id, task in tasks.items():
        row = rows[item_id]
        for field in ("task_uid", "prompt", "answer", "modality", "automatic_category", "automatic_parse_status"):
            if row.get(field) != task[field]:
                raise ValueError(f"{item_id}: annotation/task mismatch in {field}")
        if task["modality"] != "temporal":
            raise ValueError(f"{item_id}: expected temporal task")
        for field in JSON_FIELDS:
            if json.loads(row[field]) != json.loads(task[field]):
                raise ValueError(f"{item_id}: annotation/task mismatch in {field}")
        for field in REVIEWS:
            if row.get(field) not in LABELS:
                raise ValueError(f"{item_id}: invalid or missing {field}")
        if row.get("answer_status") not in ANSWER_STATUSES:
            raise ValueError(f"{item_id}: invalid or missing answer_status")

        # Human reviews describe the displayed snapshot. Replay checks applicability
        # to the current parser without changing that snapshot or the human labels.
        result = json.loads(json.dumps(asdict(parse_temporal_response(task["answer"], cases[item_id]["relation"]))))
        expected = {
            "status": task["automatic_parse_status"],
            "value": json.loads(task["final_prediction"]),
            "matched_values": json.loads(task["normalized_relevant_candidates"]),
        }
        actual = {key: result[key] for key in expected}
        if actual != expected:
            changed.append({"item_id": item_id, "audited": expected, "current": actual})

    return {
        "items": len(rows), "annotators": len(annotators),
        "by_model": dict(sorted(Counter(case["model"] for case in cases.values()).items())),
        "answer_status": dict(sorted(Counter(row["answer_status"] for row in rows.values()).items())),
        "reviews": {field: review_summary(row[field] for row in rows.values()) for field in REVIEWS},
        "parser_replay": {"matching": len(rows) - len(changed), "changed": changed},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=DIRECTORY)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.annotations = args.annotations or args.study_dir / "annotations.csv"
    args.output = args.output or args.study_dir / "results.json"
    tasks = json.loads((args.study_dir / "tasks.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in (args.study_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with args.annotations.open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))
    result = analyze(tasks, cases, annotations)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    overall = result["reviews"]["overall_pipeline_review"]
    print(json.dumps({"report": str(args.output), "overall_pipeline": overall, "parser_replay": result["parser_replay"]}, indent=2))


if __name__ == "__main__":
    main()
