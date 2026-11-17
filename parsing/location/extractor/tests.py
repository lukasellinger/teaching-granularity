"""Offline extractor contract tests; no weights, GPU or network required."""

import hashlib
import json
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from .inference import DIRECTORY, SETTINGS, LocationExtractor, decode_output, format_prompt, parse_output


class PreparationTests(unittest.TestCase):
    def test_curated_pool_reproduces_all_released_records(self):
        from .prepare import prepare, read_rows, person_registry, REPO
        source = DIRECTORY / "preparation"
        provenance = json.loads((source / "provenance.json").read_text())
        for name, expected in provenance["files"].items():
            self.assertEqual(hashlib.sha256((source / name).read_bytes()).hexdigest(), expected)
        annotations = read_rows(source / "annotations.jsonl")
        references = read_rows(DIRECTORY / "human_validation/annotations.jsonl")
        names = person_registry(read_rows(REPO / "data_preparation/vital_persons/metadata/gold_adapted.jsonl"))
        splits = prepare(annotations, references, names)
        for split, rows in splits.items():
            path = DIRECTORY / ("human_validation/annotations.jsonl" if split == "heldout" else f"training/{split}.jsonl")
            self.assertEqual(rows, read_rows(path), split)
        self.assertEqual(len(annotations), 668)
        self.assertEqual(len({r["person_group"] for r in annotations}), 665)
        with self.assertRaises(ValueError):
            prepare(annotations + [annotations[0]], references, names)
        with self.assertRaises(ValueError):
            prepare([{**annotations[0], "expected_output": None}, *annotations[1:]], references, names)
        with self.assertRaises(ValueError):
            prepare(annotations, references + [annotations[0]], names)
        with self.assertRaises(ValueError):
            prepare([{**annotations[0], "difficulty_categories": ["unknown"]}, *annotations[1:]], references, names)


class OutputTests(unittest.TestCase):
    def test_exact_training_prompt(self):
        content = (DIRECTORY / "prompt.txt").read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), SETTINGS["prompt_sha256"])
        answer = "Born in **Candia** (now Heraklion).\nLater detail {untouched}."
        prompt = format_prompt("Where was A born?", answer)
        self.assertTrue(prompt.endswith("Text: " + answer))
        self.assertTrue(prompt.startswith("Where was A born?\n"))

    def test_empty_inputs_rejected(self):
        for question, answer in [("", "Born in York."), ("Where?", ""), (None, "text")]:
            with self.assertRaises(ValueError):
                format_prompt(question, answer)

    def test_semicolons_not_commas(self):
        result = parse_output("Washington, D.C.; United States")
        self.assertEqual(result["components"], ["Washington, D.C.", "United States"])

    def test_identical_repetitions_only(self):
        result = parse_output("Valencia; Spain; valencia; province of Valencia")
        self.assertEqual(result["location"], "Valencia; Spain; province of Valencia")
        self.assertEqual(result["raw_location"], "Valencia; Spain; valencia; province of Valencia")
        self.assertEqual(parse_output("New York; NEW   YORK")["components"], ["New York"])

    def test_order_and_spatial_qualifiers_preserved(self):
        self.assertEqual(parse_output("his home; near Cambridge; England")["components"], ["his home", "near Cambridge", "England"])

    def test_alternatives_are_not_abstention(self):
        result = parse_output("York; Bath")
        self.assertEqual(result["components"], ["York", "Bath"])
        self.assertEqual(result["status"], "ok")

    def test_idk_is_success_but_failures_are_not_idk(self):
        result = parse_output(" IDK\n")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["location"], "IDK")
        self.assertEqual(result["components"], [])
        for raw, stopped, status in [("", True, "empty_output"), ("York;", False, "generation_truncated"), ("IDK", False, "generation_truncated")]:
            result = parse_output(raw, stopped)
            self.assertEqual(result["status"], status)
            self.assertIsNone(result["location"])
            self.assertEqual(result["components"], [])

    def test_decode_stops_before_padding(self):
        decode = lambda tokens, **kwargs: "; ".join(str(token) for token in tokens)
        result = decode_output([12, 34, 99, 0, 0], {99}, decode)
        self.assertEqual(result["location"], "12; 34")
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(decode_output([12, 34], {99}, decode)["status"], "generation_truncated")


class BatchTests(unittest.TestCase):
    def test_batch_padding_order_and_overlong_input(self):
        extractor = object.__new__(LocationExtractor)
        extractor.batch_size = 2
        tokenizer_calls, generate_calls = [], []

        def template(messages, **kwargs):
            tokenizer_calls.append(kwargs)
            text = messages[0]["content"]
            return [1] * (SETTINGS["max_input_tokens"] + 1) if "TOO LONG" in text else ([1, 2] if "LONGER" in text else [1])

        extractor.tokenizer = SimpleNamespace(
            pad_token_id=0, eos_token_id=99, apply_chat_template=template,
            decode=lambda tokens, **kwargs: "IDK" if tokens == [22] else "York; England",
        )

        class Generated:
            def __getitem__(self, key):
                return SimpleNamespace(tolist=lambda: [[11, 99], [22, 99]])

        def generate(**kwargs):
            generate_calls.append(kwargs)
            return Generated()

        extractor.model = SimpleNamespace(
            device="cpu", generation_config=SimpleNamespace(eos_token_id=99), generate=generate,
        )
        torch = SimpleNamespace(tensor=lambda values, **kwargs: values, inference_mode=nullcontext)
        rows = [{"question": "Where was A born?", "answer": answer} for answer in ("Short", "TOO LONG", "LONGER")]
        with patch.dict("sys.modules", {"torch": torch}):
            results = extractor.extract_many(rows)
            self.assertEqual(extractor.extract_many([]), [])
        self.assertEqual([result["status"] for result in results], ["ok", "input_too_long", "ok"])
        self.assertEqual(results[2]["location"], "IDK")
        self.assertEqual(generate_calls[0]["input_ids"], [[0, 1], [1, 2]])
        self.assertEqual(generate_calls[0]["attention_mask"], [[0, 1], [1, 1]])
        self.assertFalse(generate_calls[0]["do_sample"])
        self.assertEqual(generate_calls[0]["max_new_tokens"], 256)
        self.assertTrue(all(not call["enable_thinking"] for call in tokenizer_calls))


if __name__ == "__main__":
    unittest.main()
