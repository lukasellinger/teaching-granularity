"""Read saved predictions and require exact reference-input coverage."""

import argparse
from shared.io import read_rows as read_jsonl_rows
import json
from pathlib import Path


def read_rows(path):
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))["cases"]
    return read_jsonl_rows(path)


def aligned(references, predictions, fields):
    def index(rows):
        result = {}
        for row in rows:
            key = row.get("item_id")
            if not key or key in result:
                raise ValueError("Missing or duplicate item_id")
            result[key] = row
        return result
    refs, preds = index(references), index(predictions)
    if not refs or refs.keys() != preds.keys():
        raise ValueError("Predictions must cover exactly the reference sample")
    for key, ref in refs.items():
        pred = preds[key]
        for field in fields:
            if ref[field] != pred.get(field):
                raise ValueError(f"Prediction input differs in {field}: {key}")
        yield ref, pred


def run(directory, evaluate):
    parser = argparse.ArgumentParser(description="Score saved component predictions against reference annotations")
    parser.add_argument("--annotations", type=Path, default=directory / "annotations.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=directory / "rescored.json")
    args = parser.parse_args()
    result = evaluate(read_rows(args.annotations), read_rows(args.predictions))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "metrics": result["metrics"]}, indent=2))
