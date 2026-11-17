"""Reference coverage, sampling and scoring tests without model inference."""

import unittest
import json
import hashlib
from collections import Counter

from .sample import DIRECTORY, REPO, read_rows, sample, select
from .study import evaluate


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_rows(DIRECTORY / "annotations.jsonl")
        self.predictions = [
            {"item_id": row["item_id"], "question": row["question"], "answer": row["answer"],
             "extraction": {"status": "ok", "location": row["expected_output"]}}
            for row in self.rows
        ]

    def test_reference_coverage(self):
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(len({row["qid"] for row in self.rows}), 100)
        self.assertEqual(Counter(row["source_model"] for row in self.rows),
                         {"LFM2_5_1_2B": 33, "GEMMA4_31B": 33, "QWEN3_5_9B": 34})
        self.assertTrue(all(isinstance(row["expected_output"], str) and row["expected_output"] for row in self.rows))
        self.assertEqual(evaluate(self.rows, self.predictions)["reviewed_flags"], 0)

    def test_perfect_synthetic_predictions(self):
        result = evaluate(self.rows, self.predictions)
        self.assertEqual(result["metrics"]["strict_exact_match"]["matches"], 100)
        self.assertAlmostEqual(result["metrics"]["strict_exact_match"]["wilson_95"]["lower"], 0.9630065017930143)

    def test_failures_stay_in_denominator(self):
        self.predictions[0]["extraction"]["status"] = "generation_truncated"
        result = evaluate(self.rows, self.predictions)
        self.assertEqual(result["metrics"]["strict_exact_match"]["accuracy"], 0.99)
        self.assertEqual(len(result["mismatches"]), 1)

    def test_missing_duplicate_and_changed_inputs_rejected(self):
        for predictions in (self.predictions[:-1], self.predictions + [self.predictions[0]]):
            with self.assertRaises(ValueError):
                evaluate(self.rows, predictions)
        self.predictions[0]["answer"] += " changed"
        with self.assertRaises(ValueError):
            evaluate(self.rows, self.predictions)

    def test_blank_reference_rejected(self):
        self.rows[0]["expected_output"] = None
        with self.assertRaises(ValueError):
            evaluate(self.rows, self.predictions)

    def test_tolerance_and_component_order(self):
        row = {**self.rows[0], "expected_output": "Cambridge; England"}
        prediction = {**self.predictions[0], "extraction": {"status": "ok", "location": "Cambridge area ; England"}}
        metrics = evaluate([row], [prediction])["metrics"]
        self.assertEqual(metrics["strict_exact_match"]["matches"], 0)
        self.assertEqual(metrics["descriptor_tolerant_match"]["matches"], 1)
        prediction["extraction"]["location"] = "England; Cambridge"
        self.assertEqual(evaluate([row], [prediction])["metrics"]["descriptor_tolerant_match"]["matches"], 0)

    def test_sampling_reproduces_supplied_reference_inputs(self):
        excluded = set((DIRECTORY / "excluded_people.txt").read_text().split())
        rows = sample(REPO / "data_preparation/vital_persons/samples",
                      REPO / "data_preparation/vital_persons/metadata/gold_adapted.jsonl", excluded)
        keys = ("item_id", "qid", "source_model", "sample_index", "question", "answer")
        self.assertEqual([{key: row[key] for key in keys} for row in rows],
                         [{key: row[key] for key in keys} for row in self.rows])
        self.assertFalse({row["qid"] for row in rows} & excluded)
        self.assertTrue(all(row["expected_output"] is None for row in rows))

    def test_original_predictions_and_verified_current_scores(self):
        path = DIRECTORY / "predictions.json"
        verification = json.loads((DIRECTORY / "verification.json").read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         verification["original_predictions_sha256"])
        self.assertEqual(hashlib.sha256((DIRECTORY / "annotations.jsonl").read_bytes()).hexdigest(),
                         verification["current_annotations_sha256"])
        original = json.loads(path.read_text())
        result = evaluate(self.rows, original["cases"])
        self.assertEqual(result, json.loads((DIRECTORY / "results.json").read_text()))
        self.assertEqual(result["metrics"]["strict_exact_match"]["matches"], 91)
        self.assertEqual(verification["exact_raw_prediction_matches"], 100)
        model = json.loads((DIRECTORY.parent / "model.json").read_text())
        for key in ("adapter_repo", "adapter_revision", "adapter_subfolder", "base_model", "base_revision", "prompt_sha256"):
            self.assertEqual(verification["generation"][key], model[key])
        by_id = {r["item_id"]: r for r in original["cases"]}
        updates = [{"item_id": r["item_id"], "original_expected_output": by_id[r["item_id"]]["expected_output"],
                    "current_expected_output": r["expected_output"]} for r in self.rows
                   if r["expected_output"] != by_id[r["item_id"]]["expected_output"]]
        self.assertEqual(updates, verification["reference_updates"])
        self.assertEqual(len(updates), 2)
        old_references = [{**r, "expected_output": by_id[r["item_id"]]["expected_output"]} for r in self.rows]
        old_result = evaluate(old_references, original["cases"])
        for metric, scores in original["metrics"].items():
            if isinstance(scores, dict):
                for key, value in scores.items():
                    self.assertEqual(old_result["metrics"][metric][key], value)

    def test_unachievable_sampling_rejected(self):
        with self.assertRaises(ValueError):
            select({("lfm", "birthplace"): {}}, set(), size=1)


if __name__ == "__main__":
    unittest.main()
