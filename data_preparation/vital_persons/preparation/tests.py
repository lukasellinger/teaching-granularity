"""Offline preparation tests; model calls are replaced by deterministic rewrites."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from shared.io import write
from . import build
from .targets import summarize, replacements, target_aware_split, RELATIONS
from .realize import temporal_example, temporal_spans, location_candidates, validate_example
from parsing.location.normalization.values import LocationValue, LocationLevel
from parsing.temporal import TemporalValue, TemporalLevel


def source(qid="Q1", relation="birthplace", index=0, *, model="LFM2_5_1_2B", city="Berlin", country="Germany"):
    name = f"Person {qid}"
    born = relation in {"birth", "birthplace"}
    question = f"Where {'was' if born else 'did'} {name} {'born' if born else 'die'}?"
    temporal = relation in {"birth", "death"}
    answer = (f"{name} {'was born' if born else 'died'} on March 24, 1962." if temporal else
              f"{name} {'was born' if born else 'died'} in **{city}, {country}**.")
    value = LocationValue(level=LocationLevel.CITY, city=city, region=("Region",), country=country, continent="Europe")
    return {"case_id": f"{model}:{qid}:{relation}:{index}", "qid": qid, "source_model": model,
            "source_relation": relation, "sample_index": index, "question": question, "answer": answer,
            "extraction": {"status": "ok", "location": "IDK" if temporal else f"{city}; {country}", "components": [] if temporal else [city, country]},
            "normalization": {"status": "idk" if temporal else "normalized", "consensus_value": None if temporal else asdict(value)},
            "filter": {"filter_status": "ok", "filter_prediction": "Single"}}


def summary(rows, **kwargs):
    row = rows[0]
    return summarize(rows, model=row["source_model"], qid=row["qid"], relation=row["source_relation"], **kwargs)


class ConsensusTests(unittest.TestCase):
    def test_original_denominator_not_available_or_filtered_subset(self):
        rows = [source(index=i) for i in range(20)]
        for row in rows[15:]:
            row["normalization"]["consensus_value"] = None
        self.assertEqual(summary(rows)["target_status"], "unavailable")
        rows[15] = source(index=15)
        self.assertEqual(summary(rows)["target_status"], "known")
        for row in rows:
            row["filter"]["filter_prediction"] = "Multiple"
        self.assertEqual(summary(rows)["target_level"], "CITY")

    def test_enough_but_incompatible_evidence_is_idk(self):
        rows = [source(index=i, city=f"City {i}", country=f"Country {i}") for i in range(20)]
        for i, row in enumerate(rows):
            row["normalization"]["consensus_value"]["continent"] = f"Continent {i}"
            row["normalization"]["consensus_value"]["region"] = [f"Region {i}"]
        self.assertEqual(summary(rows)["target_status"], "idk")

    def test_operational_value_never_substitutes_for_safe_consensus(self):
        rows = [source(index=i) for i in range(20)]
        for row in rows:
            row["normalization"]["value"] = row["normalization"]["consensus_value"]
            row["normalization"]["consensus_value"] = None
        self.assertEqual(summary(rows)["available_primary_values"], 0)

    def test_place_subcity_project_before_consensus(self):
        rows = [source(index=i) for i in range(20)]
        for row in rows:
            value = row["normalization"]["consensus_value"]
            value.update(level="PLACE", place=[f"Hospital {row['sample_index']}"])
        self.assertEqual(summary(rows)["target_level"], "CITY")
        for row in rows:
            row["normalization"]["consensus_value"].update(level="SUBCITY", subcity=["District"])
        self.assertEqual(summary(rows)["target_level"], "CITY")

    def test_finer_negatives_from_one_compatible_sample(self):
        rows = [source(index=i, city=f"City {i}") for i in range(20)]
        for row in rows:
            row["normalization"]["consensus_value"]["region"] = [f"Region {row['sample_index']}"]
        target = summary(rows)
        self.assertEqual(target["target_level"], "COUNTRY")
        variants = replacements(target, 42)
        self.assertEqual([v["target_level"] for v in variants], ["CITY", "REGION", "COUNTRY", "CONTINENT"])
        finer = [v for v in variants if v["replacement_origin"] == "single_sample"]
        self.assertEqual(len({v["replacement_source_case_id"] for v in finer}), 1)
        self.assertEqual(variants, replacements(target, 42))

    def test_temporal_target_not_gold_and_five_levels(self):
        rows = [source(relation="birth", index=i) for i in range(20)]
        for row in rows:
            row["answer"] = f"Person Q1 was born in {'March' if row['sample_index'] < 10 else 'April'} 1962."
        self.assertEqual(summary(rows)["target_level"], "YEAR")
        self.assertEqual(summary(rows)["target"]["year"], 1962)

    def test_every_slot_votes_once(self):
        rows = [source(index=i) for i in range(20)]
        with self.assertRaises(ValueError):
            summary(rows[:-1])
        rows[-1]["sample_index"] = 0
        with self.assertRaises(ValueError):
            summary(rows)

    def test_split_person_grouped_deterministic_and_test_independent(self):
        targets = [{"qid": f"Q{i}", "relation": relation, "target_level": "YEAR", "target_status": "known"}
                   for i in range(12) for relation in RELATIONS]
        first, _ = target_aware_split([f"Q{i}" for i in range(10)], ["Q10", "Q11"], targets, fraction=.2)
        targets[-1]["target_level"] = "CENTURY"
        second, _ = target_aware_split([f"Q{i}" for i in reversed(range(10))], ["Q11", "Q10"], targets, fraction=.2)
        self.assertEqual(first, second)
        self.assertEqual(list(first.values()).count("dev"), 2)
        self.assertEqual(first["Q10"], "test")


class RealizationTests(unittest.TestCase):
    def test_temporal_rendering_score_spans_and_levels(self):
        row = source(relation="birth")
        common = TemporalValue(level=TemporalLevel.YEAR, year=1962)
        result = temporal_example(row, common, "train", "Person Q1", 42, True)
        self.assertEqual(result["positive_answer"], "Person Q1 was born in 1962.")
        validate_example(result)
        self.assertEqual({c["level"] for c in result["contrastive_groups"][0]["candidates"]}, {"DAY", "MONTH", "YEAR", "DECADE", "CENTURY", "IDK"})

    def test_secondary_and_multiple_dates_rejected(self):
        row = source(relation="birth")
        common = TemporalValue(level=TemporalLevel.YEAR, year=1962)
        row["extraction"]["components"] = ["Berlin"]
        with self.assertRaises(ValueError):
            temporal_example(row, common, "train", "Person Q1", 42)
        row["extraction"]["components"] = []
        row["answer"] += " He died in 2000."
        with self.assertRaises(ValueError):
            temporal_example(row, common, "train", "Person Q1", 42)

    def test_address_number_not_year_but_bad_dates_are_rejected(self):
        self.assertFalse(temporal_spans("He died at 1900 Market Street.", ["1900 Market Street"]))
        self.assertTrue(temporal_spans("He died on March 24, 19621.", []))

    def test_location_grammar_score_windows_and_unrelated_changes(self):
        rows = [{"target_level": "CITY", "replacement_location": "Berlin"}, {"target_level": "COUNTRY", "replacement_location": "Germany"}]
        answers = ["A was born in **Berlin**.", "A was born in **Germany**."]
        candidates = location_candidates(rows, answers)
        self.assertEqual(candidates[0]["answer"][candidates[0]["score_start"]:candidates[0]["score_end"]], "Berlin")
        with self.assertRaises(ValueError):
            location_candidates(rows, [answers[0], "Another person was born in **Germany**."])


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.metadata = self.directory / "metadata"
        write(self.metadata / "development.jsonl", [{"qid": f"Q{i}", "name": f"Person Q{i}", "birth_year": 1} for i in range(9)], rows=True)
        write(self.metadata / "test.jsonl", [{"qid": "Q9", "name": "Person Q9"}], rows=True)
        self.processed = self.directory / "processed"
        write(self.processed / "lfm.jsonl", [source(f"Q{i}", relation, index) for i in range(10) for relation in RELATIONS for index in range(2)], rows=True)
        self.args = SimpleNamespace(metadata_dir=self.metadata, processed_dir=self.processed, models=["lfm"],
            threshold=.8, expected_samples=2, dev_fraction=1/9, seed=42, include_idk_negatives=False,
            work_dir=self.directory / "plan", output_dir=self.directory / "datasets", batch_size=8,
            dtype="float32", device="cpu", attention_implementation="sdpa")

    def test_plan_rewrite_assemble_resume_no_test_leakage(self):
        first = build.plan(self.args)
        self.assertEqual(first, build.plan(self.args))
        tasks = json.loads((self.args.work_dir / "tasks.json").read_text())
        self.assertTrue(tasks)
        self.assertFalse(any("Person Q9 " in task["answer"] for task in tasks.values()))
        class Rewriter:
            def __init__(self, **kwargs):
                pass
            def process_many(self, rows):
                output = []
                for row in rows:
                    person = row["answer"].split(" was born")[0] if " was born" in row["answer"] else row["answer"].split(" died")[0]
                    event = "was born" if " was born" in row["answer"] else "died"
                    output.append({"status": "ok", "prediction": f"{person} {event} in **{row['replacement_location']}**."})
                return output
        with patch.object(build, "LocationRewriter", Rewriter):
            build.rewrite(self.args)
        with patch.object(build, "LocationRewriter", side_effect=AssertionError("Should reuse rewrites")):
            build.rewrite(self.args)
        outputs = build.assemble(self.args)
        self.assertGreater(outputs["lfm"]["examples"], 0)
        directory = self.args.output_dir / "lfm"
        self.assertEqual((directory / "test.jsonl").read_text(), "")
        train, dev = (list(build.read_rows(directory / f"{split}.jsonl")) for split in ("train", "dev"))
        self.assertFalse({r["qid"] for r in train} & {r["qid"] for r in dev})
        self.assertNotIn("Q9", {r["qid"] for r in train + dev})
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["case_targets_by_split"]["test"], 4)
        self.assertEqual(manifest["accepted_examples"], len(train) + len(dev))
        self.assertEqual(build.assemble(self.args)["lfm"], manifest["accepted_examples"])
        (directory / "train.jsonl").write_text("changed")
        with self.assertRaises(ValueError):
            build.assemble(self.args)

    def test_gold_changes_do_not_change_targets_or_plan(self):
        first = build.plan(self.args)
        rows = list(build.read_rows(self.metadata / "development.jsonl"))
        for row in rows:
            row.update(birth_year=9999, birth_city="Wrong gold")
        write(self.metadata / "development.jsonl", rows, rows=True)
        self.assertEqual(build.plan(self.args)["signature"], first["signature"])

    def test_stale_processed_input_rejected(self):
        build.plan(self.args)
        path = self.processed / "lfm.jsonl"
        with path.open("a") as target:
            target.write("\n")
        with self.assertRaises(ValueError):
            build.plan(self.args)

    def test_flatten_uses_location_questions_for_secondary_extraction(self):
        from ..generate import PROMPTS
        samples = self.directory / "samples"
        for relation in RELATIONS:
            rows = [{"qid": f"Q{i}", "samples": [source(f"Q{i}", relation, j)["answer"] for j in range(2)]}
                    for i in range(10)]
            write(samples / "lfm" / f"{relation}_samples.jsonl", rows, rows=True)
        args = SimpleNamespace(metadata_dir=self.metadata, samples_dir=samples, models=["lfm"],
                               expected_samples=2, output_dir=self.directory / "flat")
        build.flatten(args)
        rows = list(build.read_rows(args.output_dir / "lfm.jsonl"))
        self.assertEqual(len(rows), 80)
        self.assertTrue(all(row["question"].startswith("Where") for row in rows))
        birth = next(row for row in rows if row["source_relation"] == "birth")
        self.assertEqual(birth["generation_question"], PROMPTS["birth"].format(person="Person Q0"))
        self.assertTrue(birth["case_id"].startswith("LFM2_5_1_2B:"))

    def test_four_models_have_separate_targets_and_outputs(self):
        self.args.models = list(build.MODEL_NAMES)
        for model, name in build.MODEL_NAMES.items():
            write(self.processed / f"{model}.jsonl", [source(f"Q{i}", relation, index, model=name)
                for i in range(10) for relation in RELATIONS for index in range(2)], rows=True)
        build.plan(self.args)
        for model, name in build.MODEL_NAMES.items():
            planned = json.loads((self.args.work_dir / f"{model}.json").read_text())
            self.assertEqual({r["source_model"] for r in planned["case_targets"]}, {name})
            self.assertEqual(planned["split_report"]["people"], {"train": 8, "dev": 1, "test": 1})

    def test_missing_preprocessing_is_not_silently_unavailable(self):
        rows = list(build.read_rows(self.processed / "lfm.jsonl"))
        row = next(r for r in rows if r["source_relation"] == "birthplace")
        del row["normalization"]
        write(self.processed / "lfm.jsonl", rows, rows=True)
        with self.assertRaisesRegex(ValueError, "normalization and filter"):
            build.plan(self.args)

    def test_identical_surface_levels_keep_consensus_positive_label(self):
        rows = list(build.read_rows(self.processed / "lfm.jsonl"))
        for row in rows:
            if row["source_relation"] in {"birthplace", "deathplace"}:
                value = row["normalization"]["consensus_value"]
                value["region"] = ["Berlin"]
                if row["sample_index"] == 0:
                    value["level"] = "REGION"
        write(self.processed / "lfm.jsonl", rows, rows=True)
        build.plan(self.args)
        tasks = json.loads((self.args.work_dir / "tasks.json").read_text())
        import sqlite3
        from contextlib import closing
        with closing(sqlite3.connect(self.args.work_dir / "rewrites.sqlite3")) as cache:
            cache.execute("CREATE TABLE rewrites (cache_key TEXT PRIMARY KEY, result TEXT NOT NULL)")
            for key, task in tasks.items():
                event = "was born" if "was born" in task["answer"] else "died"
                prefix = task["answer"].split(event)[0] + event
                cache.execute("INSERT INTO rewrites VALUES (?,?)", (key, json.dumps({"status": "ok", "prediction": f"{prefix} in **{task['replacement_location']}**."})))
            cache.commit()
        build.assemble(self.args)
        examples = [row for split in ("train", "dev") for row in build.read_rows(self.args.output_dir / "lfm" / f"{split}.jsonl")]
        location = [row for row in examples if row["relation"] in {"birthplace", "deathplace"}]
        self.assertTrue(location)
        self.assertEqual({row["primary_target_level"] for row in location}, {"REGION"})
        self.assertTrue(all(row["contrastive_groups"][0]["candidates"][0]["level"] == "REGION" for row in location))


class SuppliedDatasetTests(unittest.TestCase):
    def test_supplied_dataset_checksums_splits_and_counts(self):
        metadata_test = {row["qid"] for row in build.read_rows(build.VITAL / "metadata/test.jsonl")}
        expected = {"lfm": (10545, 1359), "qwen": (8139, 1051),
                    "gemma": (7351, 884), "ministral": (9191, 1096)}
        for model, (train_count, dev_count) in expected.items():
            with self.subTest(model=model):
                directory = build.VITAL / "datasets" / model
                manifest = json.loads((directory / "manifest.json").read_text())
                for record in manifest["output_files"]:
                    path = directory / record["path"]
                    self.assertEqual(path.stat().st_size, record["size_bytes"])
                    self.assertEqual(build.file_hash(path), record["sha256"])
                assignments = {}
                for target in build.read_rows(directory / "case_targets.jsonl"):
                    self.assertEqual(assignments.setdefault(target["qid"], target["split"]), target["split"])
                self.assertEqual({qid for qid, split in assignments.items() if split == "test"}, metadata_test)
                for split, count in (("train", train_count), ("dev", dev_count)):
                    rows = list(build.read_rows(directory / f"{split}.jsonl"))
                    self.assertEqual(len(rows), count)
                    self.assertEqual(len(rows), manifest["accepted_by_split"][split])
                    self.assertTrue(all(assignments[row["qid"]] == split for row in rows))


if __name__ == "__main__":
    unittest.main()
