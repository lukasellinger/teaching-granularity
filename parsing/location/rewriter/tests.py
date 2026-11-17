"""Reference and output-policy tests without loading model weights."""

import json
import hashlib
import unittest
from pathlib import Path

from .inference import format_prompt, postprocess, validate
from .scoring import formatting_match
from .validation import replacement_expression_count
from .human_validation.study import evaluate
from .prepare import read_rows, sample, assemble

DIRECTORY = Path(__file__).resolve().parent


def example(replacement="his home, Los Angeles"):
    return {"question": "Where did Billy Wilder die?", "answer": "Billy Wilder died at his home in Los Angeles, California.",
            "original_location": "his home; Los Angeles; California", "replacement_location": replacement}


class RewriterTests(unittest.TestCase):
    def test_original_population_and_released_splits_reproduce(self):
        preparation = DIRECTORY / "preparation"
        provenance = json.loads((preparation / "provenance.json").read_text())
        for name, expected in provenance["files"].items():
            self.assertEqual(hashlib.sha256((preparation / name).read_bytes()).hexdigest(), expected)
        population = read_rows(preparation / "population.jsonl")
        annotations = read_rows(preparation / "annotations.jsonl")
        self.assertEqual((len(population), len({r["qid"] for r in population})), (1879, 471))
        splits = assemble(population, annotations)
        for split, rows in splits.items():
            path = DIRECTORY / ("human_validation/annotations.jsonl" if split == "heldout" else f"training/{split}.jsonl")
            self.assertEqual(rows, read_rows(path), split)
        cohorts = sample(population)
        self.assertEqual(cohorts, sample(list(reversed(population))))
        counts = {}
        people = set()
        for sources in cohorts.values():
            for row in sources:
                self.assertNotIn(row["qid"], people)
                people.add(row["qid"])
                counts[row["source_model"]] = counts.get(row["source_model"], 0) + 1
        self.assertEqual(counts, {"LFM2_5_1_2B": 100, "QWEN3_5_9B": 100})
        with self.assertRaises(ValueError):
            assemble(population, annotations[:-1])
        with self.assertRaises(ValueError):
            assemble(population, annotations + [annotations[0]])
        contaminated = [{**population[0], "rewrites": [{**r, "expected_output": "leaked label"}
                                                    for r in population[0]["rewrites"]]}, *population[1:]]
        with self.assertRaises(ValueError):
            sample(contaminated)

    def test_original_heldout_predictions_reproduce_recorded_results(self):
        directory = DIRECTORY / "human_validation"
        preparation = DIRECTORY / "preparation"
        provenance = json.loads((preparation / "provenance.json").read_text())
        self.assertTrue((preparation / provenance["preparation_script"]).is_file())
        path = preparation / provenance["heldout_prediction_report"]["path"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         provenance["heldout_prediction_report"]["sha256"])
        references = read_rows(directory / "annotations.jsonl")
        predictions = json.loads(path.read_text())["cases"]
        actual = evaluate(references, predictions)
        expected = json.loads((directory / "results.json").read_text())["metrics"]
        self.assertEqual(actual["metrics"], expected)
        self.assertEqual(len(actual["mismatches"]), 2)
        self.assertTrue(all(not validate(row, row["prediction"]) for row in predictions))

    def test_exact_prompt(self):
        text = format_prompt(example())
        self.assertIn("Replacement location: his home, Los Angeles", text)
        self.assertIn("Original text:\nBilly Wilder died", text)
        self.assertIn("Return only the complete rewritten event sentence", text)

    def test_natural_connectors_and_markdown(self):
        for text in ("Billy Wilder died at his home in **Los Angeles**.", "Billy Wilder died at his home, **Los Angeles**."):
            self.assertEqual(validate(example(), text), [])
        self.assertEqual(validate(example("his home, 13 High Street, London"),
                                  "Billy Wilder died at his home at **13 High Street**, in **London**."), [])

    def test_missing_repeated_reordered_disconnected_or_qualified_parts_fail(self):
        for text in ("Billy Wilder died at his home.",
                     "Billy Wilder died at his home in Los Angeles. He passed away at his home in Los Angeles.",
                     "Billy Wilder died at his home. He died in Los Angeles.",
                     "Billy Wilder died in Los Angeles at his home.",
                     "Billy Wilder died at his home near Los Angeles.",
                     "Billy Wilder died at his home in Los Angelesville."):
            self.assertIn("replacement_count", validate(example(), text))

    def test_internal_commas_and_person_name_collision(self):
        self.assertEqual(replacement_expression_count(example("his home, Washington, D.C."),
                                                      "Billy Wilder died at his home in Washington, D.C."), 1)
        self.assertEqual(replacement_expression_count(example("his home, Washington, D.C."),
                                                      "Billy Wilder died at his home in Washington in D.C."), 0)
        row = {"question": "Where did Cyril of Alexandria die?", "answer": "Cyril of Alexandria died in Alexandria.", "replacement_location": "Alexandria"}
        self.assertEqual(validate(row, row["answer"]), [])

    def test_trim_fragment_but_never_accept_truncated_generation(self):
        raw = "Billy Wilder died at his home in Los Angeles. Later he"
        result = postprocess(example(), {"status": "ok", "raw_prediction": raw})
        self.assertTrue(result["output_tail_trimmed"])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["prediction"], "Billy Wilder died at his home in Los Angeles.")
        self.assertEqual(postprocess(example(), {"status": "generation_truncated", "raw_prediction": raw})["status"], "generation_truncated")

    def test_name_preservation_and_empty_output(self):
        self.assertIn("person_name_changed", validate(example(), "Someone died at his home in Los Angeles."))
        self.assertIn("incomplete_output", validate(example(), ""))

    def test_formatting_tolerance_keeps_words(self):
        case = {**example("England"), "status": "ok", "prediction": "Billy Wilder died in **England**.", "expected_output": "Billy Wilder died in England."}
        self.assertTrue(formatting_match(case))
        case["prediction"] = "Billy Wilder was born in England."
        self.assertFalse(formatting_match(case))

    def test_reference_coverage_and_synthetic_scores(self):
        rows = [json.loads(line) for line in (DIRECTORY / "human_validation/annotations.jsonl").read_text().splitlines()]
        self.assertEqual((len(rows), len({row["qid"] for row in rows})), (154, 40))
        predictions = [{**row, "rewrite": {"status": "ok", "prediction": row["expected_output"], "validation_errors": []}} for row in rows]
        result = evaluate(rows, predictions)
        self.assertEqual(result["metrics"]["strict_exact_match"], 1)
        with self.assertRaises(ValueError):
            evaluate(rows, predictions[:-1])
        self.assertEqual(json.loads((DIRECTORY / "human_validation/results.json").read_text())["metrics"]["items"], 154)


if __name__ == "__main__":
    unittest.main()
