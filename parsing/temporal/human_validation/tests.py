"""Audit coverage, scoring and snapshot replay tests (standard library only)."""

import copy
import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from .study import DIRECTORY, analyze, review_summary
from .sample import balanced_sample, load_population, make_tasks


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tasks = json.loads((DIRECTORY / "tasks.json").read_text(encoding="utf-8"))
        self.cases = [json.loads(line) for line in (DIRECTORY / "cases.jsonl").read_text().splitlines()]
        with (DIRECTORY / "annotations.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))

    def analyze(self):
        return analyze(self.tasks, self.cases, self.rows)

    def test_actual_audit_and_current_parser(self):
        result = self.analyze()
        self.assertEqual(result["items"], 25)
        self.assertEqual(result["by_model"], {"LFM2_5_1_2B": 9, "GEMMA4_31B": 8, "QWEN3_5_9B": 8})
        self.assertEqual(result["reviews"]["overall_pipeline_review"]["correct"], 25)
        self.assertAlmostEqual(result["reviews"]["overall_pipeline_review"]["wilson_95"]["lower"], 0.8668077490609515)
        self.assertEqual(result["reviews"]["multiple_values_review"]["correct"], 6)
        self.assertEqual(result["reviews"]["multiple_values_review"]["not_applicable"], 19)
        self.assertEqual(result["parser_replay"], {"matching": 25, "changed": []})

    def test_excluded_labels_are_not_errors(self):
        result = review_summary(["Correct", "Incorrect", "Not applicable", "Cannot determine"])
        self.assertEqual(result["rate"], 0.5)
        self.assertEqual(result["not_applicable"], 1)
        self.assertIsNone(review_summary(["Not applicable"])["rate"])
        self.assertIsNone(review_summary(["Cannot determine"])["wilson_95"])

    def test_missing_annotation_rejected(self):
        self.rows.pop()
        with self.assertRaises(ValueError):
            self.analyze()

    def test_duplicate_annotation_rejected(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        with self.assertRaises(ValueError):
            self.analyze()

    def test_changed_answer_rejected(self):
        self.rows[0]["answer"] += " changed"
        with self.assertRaisesRegex(ValueError, "mismatch in answer"):
            self.analyze()

    def test_missing_review_rejected(self):
        self.rows[0]["span_review"] = ""
        with self.assertRaisesRegex(ValueError, "missing span_review"):
            self.analyze()

    def test_replay_difference_does_not_change_human_review(self):
        item_id = self.tasks[0]["data"]["item_id"]
        task = self.tasks[0]["data"]
        row = next(row for row in self.rows if row["item_id"] == item_id)
        for value in (task, row):
            value["automatic_parse_status"] = "unparsed"
            value["final_prediction"] = "null"
            value["normalized_relevant_candidates"] = "[]"
        result = self.analyze()
        self.assertEqual(result["parser_replay"]["matching"], 24)
        self.assertEqual(result["parser_replay"]["changed"][0]["item_id"], item_id)
        self.assertEqual(result["reviews"]["overall_pipeline_review"]["correct"], 25)


class SamplingTests(unittest.TestCase):
    def test_balanced_without_replacement_and_deterministic(self):
        population = [{"model": model, "item_id": f"{model}-{i}"} for model in ("lfm", "qwen", "gemma") for i in range(328)]
        selected = balanced_sample(population)
        self.assertEqual(selected, balanced_sample(population))
        self.assertEqual(len({row["item_id"] for row in selected}), 25)
        self.assertEqual(Counter(row["model"] for row in selected), {"lfm": 9, "qwen": 8, "gemma": 8})

    def test_small_samples_allow_zero_allocation(self):
        selected = balanced_sample([{"model": str(i)} for i in range(4)], size=1)
        self.assertEqual(len(selected), 1)

    def test_impossible_sizes_and_balance_rejected(self):
        population = [{"model": "a"}] + [{"model": "b"}] * 5
        for size in (0, 7, 6):
            with self.assertRaises(ValueError):
                balanced_sample(population, size=size)

    def test_evaluation_loading_and_task_contents(self):
        row = {
            "case_id": "LFM2_5_1_2B:Q1:birth", "prompt": "When was A born?",
            "answer": "A was born in 1900.", "modality": "temporal",
            "variant": "base_plain", "split": "test", "relation": "birth", "qid": "Q1",
            "prediction_category": "YEAR", "prediction": {"year": 1900},
            "matched_values": [{"year": 1900}], "parse_status": "parsed",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluated_generations.jsonl"
            path.write_text(json.dumps(row) + "\n")
            population, study_id = load_population([f"LFM2_5_1_2B={path}"])
            tasks, cases = make_tasks(balanced_sample(population, size=1), study_id)
            self.assertEqual(tasks[0]["data"]["answer"], row["answer"])
            self.assertNotIn("model", tasks[0]["data"])
            self.assertEqual(cases[0]["qid"], "Q1")
            self.assertEqual(json.loads(tasks[0]["data"]["final_prediction"]), row["prediction"])
            with self.assertRaises(ValueError):
                load_population([f"lfm={path}", f"lfm={path}"])
            path.write_text((json.dumps(row) + "\n") * 2)
            with self.assertRaises(ValueError):
                load_population([f"lfm={path}"])
            path.write_text(json.dumps({**row, "split": "dev"}) + "\n")
            with self.assertRaises(ValueError):
                load_population([f"lfm={path}"])


if __name__ == "__main__":
    unittest.main()
