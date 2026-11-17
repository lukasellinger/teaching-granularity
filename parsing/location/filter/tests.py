"""Frozen-threshold and safety-policy tests without model inference."""

import json
import math
import unittest
from pathlib import Path

from parsing.location._inference import padded
from .inference import SETTINGS, classify, format_prompt, input_location
from .human_validation.study import evaluate
from .calibration import select_threshold
from .prepare import assemble, read_rows, sample

DIRECTORY = Path(__file__).resolve().parent


class FilterTests(unittest.TestCase):
    def test_threshold_direction_and_boundary(self):
        self.assertEqual(SETTINGS["multiple_log_odds_threshold"], -3.375)
        self.assertEqual(classify({"Single": -1, "Multiple": -4.375})["filter_prediction"], "Multiple")
        self.assertEqual(classify({"Single": -1, "Multiple": -4.376})["filter_prediction"], "Single")
        self.assertIsNone(classify({"Single": math.nan, "Multiple": -1})["filter_prediction"])

    def test_upstream_failure_and_idk_never_admitted(self):
        for extraction in ({"status": "generation_truncated", "location": "York"},
                           {"status": "ok", "location": "IDK"}, {"location": ""}):
            location, reason = input_location({"extraction": extraction})
            self.assertIsNone(location)
            self.assertIsNotNone(reason)
        self.assertEqual(input_location({"event_extraction": {"location": "York; England"}}), ("York; England", None))

    def test_exact_prompt_and_question_extraction_answer(self):
        text = format_prompt({"question": "Where was A born?", "answer": "A was born in York."}, "York")
        self.assertIn("Question: Where was A born?", text)
        self.assertIn("Extracted event location: York", text)
        self.assertIn("Complete answer: A was born in York.", text)
        self.assertIn("Repeating the event", text)

    def test_left_padding(self):
        self.assertEqual(padded([[1], [1, 2]], 0), ([[0, 1], [1, 2]], [[0, 1], [1, 1]]))

    def test_scoring_uses_a_new_adapters_declared_threshold(self):
        row = {"item_id": "one", "question": "Where was A born?", "answer": "A was born in York.",
               "event_extraction": {"location": "York"}, "expected_output": "Single"}
        prediction = {**row, "filter": {"filter_status": "ok", "multiple_log_odds": -2,
                                       "multiple_log_odds_threshold": 0}}
        self.assertEqual(evaluate([row], [prediction])["metrics"]["accuracy"], 1)
        self.assertEqual(evaluate([row], [prediction])["multiple_log_odds_threshold"], 0)

    def test_reference_provenance_and_saved_summary(self):
        rows = [json.loads(line) for line in (DIRECTORY / "human_validation/annotations.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 200)
        self.assertTrue(all(row["annotation_source"] == "codex_teacher_with_selective_human_review" for row in rows))
        self.assertTrue(all(row["human_reviewed"] is None for row in rows))
        self.assertTrue(all("reviewed" not in row for row in rows))
        self.assertEqual(len({row["qid"] for row in rows}), 200)
        single = [row for row in rows if row["expected_output"] == "Single"]
        multiple = [row for row in rows if row["expected_output"] == "Multiple"]
        self.assertEqual((len(single), len(multiple)), (136, 64))
        predictions = json.loads((DIRECTORY / "human_validation/predictions.json").read_text())["cases"]
        actual = evaluate(rows, predictions)["metrics"]
        expected = json.loads((DIRECTORY / "human_validation/results.json").read_text())["metrics"]
        self.assertEqual(actual, expected)
        with self.assertRaises(ValueError):
            evaluate(rows, predictions[:-1])

    @unittest.skipUnless((DIRECTORY / "training/dev_predictions.json").exists(),
                         "Original development predictions are local-only; regenerate them to replay calibration")
    def test_original_dev_scores_reproduce_frozen_threshold(self):
        predictions = json.loads((DIRECTORY / "training/dev_predictions.json").read_text())["cases"]
        references = read_rows(DIRECTORY / "training/dev.jsonl")
        # Evaluate checks exact coverage and input identity before calibration.
        evaluate(references, predictions)
        scores = {row["item_id"]: row for row in predictions}
        self.assertTrue(all(row["expected_output"] == scores[row["item_id"]]["expected_output"] for row in references))
        threshold, _ = select_threshold([{**row, "multiple_log_odds": scores[row["item_id"]]["multiple_log_odds"]} for row in references])
        self.assertEqual(threshold, SETTINGS["multiple_log_odds_threshold"])

    def test_preparation_rebuilds_exact_released_records_and_order(self):
        rebuilt = assemble(read_rows(DIRECTORY / "preparation/annotations.jsonl"))
        for split, path in (("train", "training/train.jsonl"), ("dev", "training/dev.jsonl"),
                            ("heldout", "human_validation/annotations.jsonl")):
            self.assertEqual(rebuilt[split], read_rows(DIRECTORY / path))

    @unittest.skipUnless((DIRECTORY / "preparation/extractor_population.jsonl").exists(),
                         "Original extractor population is a local-only preparation trace")
    def test_original_reference_extractor_sampling_and_exclusions(self):
        population = read_rows(DIRECTORY / "preparation/extractor_population.jsonl")
        records = read_rows(DIRECTORY / "preparation/annotations.jsonl")
        excluded = []
        for batch, seed in (("development-150-v1", 20261008), ("development-150-v2", 20261009)):
            chosen = sample(population, source="extractor", size=150, seed=seed, excluded=excluded)
            expected = [record["row"]["item_id"] for record in records if record["batch"] == batch]
            self.assertEqual([row["item_id"] for row in chosen], expected)
            self.assertFalse({row["qid"] for row in chosen} & {row["qid"] for row in excluded})
            excluded.extend(chosen)



if __name__ == "__main__":
    unittest.main()
