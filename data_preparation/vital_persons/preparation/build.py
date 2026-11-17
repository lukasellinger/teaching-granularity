"""Freeze model targets, cache rewrites, and assemble validated JSONL datasets."""
from shared.constants import MODEL_NAMES

from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unicodedata

from shared.io import write, iter_rows, digest, file_hash
from parsing.location.rewriter.inference import SETTINGS, LocationRewriter, format_prompt, validate
from .targets import RELATIONS, summarize, target_aware_split, replacements, temporal_value
from .realize import (IDK_ANSWER, example, temporal_example, location_eligible,
                      location_candidates, balanced_markdown, validate_example)

ROOT = Path(__file__).resolve().parent
VITAL = ROOT.parent
VERSION = "vital-primary-five-temporal-four-location-v1"


def read_rows(path):
    yield from iter_rows(path)


def people(metadata_dir):
    pools = {}
    names = {}
    for split in ("development", "test"):
        rows = list(read_rows(metadata_dir / f"{split}.jsonl"))
        pools[split] = {row["qid"] for row in rows}
        if not rows or len(pools[split]) != len(rows):
            raise ValueError("Empty pool or duplicate person in split metadata")
        for row in rows:
            if row["qid"] in names:
                raise ValueError("Development/test person overlap")
            # Deliberately do not access gold dates or geographic values.
            names[row["qid"]] = row["name"]
    return pools, names


def flatten(args):
    from ..generate import PROMPTS, validate_pool
    pools, names = people(args.metadata_dir)
    identities = [{"qid": qid, "name": name} for qid, name in names.items()]
    if args.output_dir.exists():
        raise ValueError("Use a new flatten output directory")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        stage = Path(temporary) / "inputs"
        stage.mkdir()
        for model in args.models:
            def rows():
                for relation in RELATIONS:
                    path = args.samples_dir / model / f"{relation}_samples.jsonl"
                    records = list(read_rows(path))
                    if validate_pool(records, identities, relation, args.expected_samples) != set(names):
                        raise ValueError(f"Incomplete generation pool: {model}/{relation}")
                    for record in records:
                        qid = record["qid"]
                        event = "birthplace" if relation in {"birth", "birthplace"} else "deathplace"
                        for index, answer in enumerate(record["samples"]):
                            yield {"case_id": f"{MODEL_NAMES[model]}:{qid}:{relation}:{index}", "qid": qid,
                                   "source_model": MODEL_NAMES[model], "source_relation": relation, "sample_index": index,
                                   "generation_question": PROMPTS[relation].format(person=names[qid]),
                                   "question": PROMPTS[event].format(person=names[qid]), "answer": answer}
            write(stage / f"{model}.jsonl", rows(), rows=True)
        stage.rename(args.output_dir)


def compact(row):
    required = ("case_id", "qid", "source_model", "source_relation", "sample_index", "question", "answer")
    result = {field: row[field] for field in required}
    if not isinstance(row.get("extraction"), dict):
        raise ValueError("Processed rows must contain extractor output")
    if row["source_relation"] in {"birthplace", "deathplace"} and not all(isinstance(row.get(field), dict) for field in ("normalization", "filter")):
        raise ValueError("Location rows require normalization and filter outputs before planning")
    result["extraction"] = {field: row["extraction"].get(field) for field in ("status", "location", "components")}
    resolution = row.get("normalization", {})
    result["normalization"] = {field: resolution.get(field) for field in ("status", "consensus_value")}
    result["filter"] = row.get("filter", {})
    return result


def reject(row, reason, details):
    return {"case_id": row["case_id"], "qid": row["qid"], "source_model": row["source_model"],
            "relation": row["source_relation"], "original_answer": row["answer"], "reason": reason,
            "details": details, "source_spans": []}


def plan(args):
    pools, names = people(args.metadata_dir)
    inputs = {model: file_hash(args.processed_dir / f"{model}.jsonl") for model in args.models}
    repo = VITAL.parents[1]
    shared = [path for path in (repo / "parsing/temporal").glob("*.py") if path.name != "tests.py"]
    shared.append(repo / "parsing/location/normalization/values.py")
    shared.extend(repo/'shared'/f'{name}.py' for name in ('text', 'geography', 'io', 'constants'))
    identity = {"version": VERSION, "models": args.models, "inputs": inputs,
                "input_sizes": {model: (args.processed_dir / f"{model}.jsonl").stat().st_size for model in args.models},
                "shared_code": {str(path.relative_to(repo)): file_hash(path) for path in shared},
                "people": [[qid, name] for qid, name in sorted(names.items())], "test_people": sorted(pools["test"]),
                "threshold": args.threshold, "expected_samples": args.expected_samples,
                "dev_fraction": args.dev_fraction, "seed": args.seed,
                "include_idk_negatives": args.include_idk_negatives,
                "rewriter": SETTINGS,
                "code": {path.name: file_hash(path) for path in ROOT.glob("*") if path.suffix == ".py" and path.name != "tests.py"},
                "question_templates": file_hash(ROOT / "prompts.json")}
    signature = digest(identity)
    if args.work_dir.exists():
        manifest = read_plan(args.work_dir)
        if manifest["signature"] != signature:
            raise ValueError("Plan inputs/settings changed; use a new work directory")
        return manifest
    tasks, plans = {}, {}
    from tqdm.auto import tqdm
    for model in args.models:
        groups = defaultdict(list)
        for raw in read_rows(args.processed_dir / f"{model}.jsonl"):
            row = compact(raw)
            if row["source_model"] != MODEL_NAMES[model] or row["qid"] not in names or row["source_relation"] not in RELATIONS:
                raise ValueError("Unexpected processed source identity")
            expected_id = f"{row['source_model']}:{row['qid']}:{row['source_relation']}:{row['sample_index']}"
            if row["case_id"] != expected_id:
                raise ValueError("Case ID does not match provenance")
            groups[row["qid"], row["source_relation"]].append(row)
        targets = [summarize(groups[qid, relation], model=MODEL_NAMES[model], qid=qid, relation=relation,
                             threshold=args.threshold, expected_samples=args.expected_samples)
                   for qid in sorted(names) for relation in RELATIONS]
        split, split_report = target_aware_split(pools["development"], pools["test"], targets,
                                                fraction=args.dev_fraction, seed=args.seed)
        examples, rejects, sources = [], [], []
        for target in tqdm(targets, desc=f"Plan {model}", unit="case"):
            target["split"] = split[target["qid"]]
            if target["split"] == "test":
                continue
            rows = sorted(groups[target["qid"], target["relation"]], key=lambda row: row["sample_index"])
            if target["target_status"] == "idk":
                row = {**rows[0], "case_id": target["case_id"] + ":idk"}
                examples.append(example(row, target["split"], names[target["qid"]], None, IDK_ANSWER, [], args.seed, idk=True))
                continue
            for row in rows:
                if target["target_status"] == "unavailable":
                    rejects.append(reject(row, "insufficient_primary_values", "Insufficient parsed/normalized evidence; not IDK"))
                    continue
                if target["relation"] in {"birth", "death"}:
                    try:
                        examples.append(temporal_example(row, temporal_value(target["target"]), target["split"], names[row["qid"]], args.seed, args.include_idk_negatives))
                    except ValueError as error:
                        rejects.append(reject(row, "malformed_source_answer", str(error)))
                    continue
                reason = location_eligible(row)
                if reason:
                    rejects.append(reject(row, "malformed_source_answer", reason))
                    continue
                variants = []
                for replacement in replacements(target, args.seed):
                    task = {"question": row["question"], "answer": row["answer"],
                            "original_location": row["extraction"]["location"],
                            "original_components": row["extraction"]["components"],
                            "normalized_value": replacement.get("replacement_source_value", target["target"]), **replacement}
                    key = digest({"rewriter": SETTINGS, "prompt": format_prompt(task)})
                    tasks.setdefault(key, task)
                    variants.append({"key": key, "target_level": replacement["target_level"], "normalized_value": task["normalized_value"]})
                if target["target_level"] not in {v["target_level"] for v in variants}:
                    raise ValueError("Consensus positive is missing from rewrite tasks")
                sources.append({"row": {field: row[field] for field in ("case_id", "qid", "source_model", "source_relation", "sample_index", "answer")},
                                "split": target["split"], "name": names[row["qid"]], "target_level": target["target_level"], "variants": variants})
        plans[model] = {"examples": examples, "rejects": rejects, "case_targets": targets,
                        "location_sources": sources, "split_report": split_report}
    args.work_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir.parent) as temporary:
        stage = Path(temporary) / "plan"
        stage.mkdir()
        write(stage / "tasks.json", tasks)
        for model, value in plans.items():
            write(stage / f"{model}.json", value)
        manifest = {"signature": signature, "identity": identity, "unique_rewrite_tasks": len(tasks),
                    "files": {path.name: file_hash(path) for path in stage.iterdir()}}
        write(stage / "manifest.json", manifest)
        stage.rename(args.work_dir)
    return manifest


def read_plan(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(manifest["identity"]) != manifest["signature"]:
        raise ValueError("Frozen plan identity changed")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Frozen plan changed: {name}")
    if manifest["identity"]["rewriter"] != SETTINGS:
        raise ValueError("Rewriter settings differ from frozen plan")
    for name, expected in manifest["identity"]["code"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Preparation code changed; use a new work directory")
    for name, expected in manifest["identity"]["shared_code"].items():
        if file_hash(VITAL.parents[1] / name) != expected:
            raise ValueError("Shared parsing/value code changed; use a new work directory")
    if file_hash(ROOT / "prompts.json") != manifest["identity"]["question_templates"]:
        raise ValueError("Question templates changed")
    return manifest


def rewrite(args):
    read_plan(args.work_dir)
    tasks = json.loads((args.work_dir / "tasks.json").read_text())
    rewrite_tasks(args, tasks)


def rewrite_tasks(args, tasks):
    """Run supplied location tasks through the shared resumable rewrite cache."""
    with closing(sqlite3.connect(args.work_dir / "rewrites.sqlite3")) as cache:
        cache.execute("PRAGMA journal_mode=WAL")
        cache.execute("CREATE TABLE IF NOT EXISTS rewrites (cache_key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        cache.commit()
        existing = {key for key, result in cache.execute("SELECT cache_key,result FROM rewrites") if json.loads(result)["status"] == "ok"}
        pending = [(key, task) for key, task in tasks.items() if key not in existing]
        print(f"{len(existing & tasks.keys())} cached; {len(pending)} rewrites pending")
        if not pending:
            return
        runtime = {field: getattr(args, field) for field in ("dtype", "attention_implementation", "device")}
        runtime_path = args.work_dir / "rewrite_runtime.json"
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
            raise ValueError("Use the original rewrite runtime settings or a new work directory")
        write(runtime_path, runtime)
        model = LocationRewriter(batch_size=args.batch_size, **runtime)
        from tqdm.auto import tqdm
        with tqdm(total=len(pending), desc="Location rewriting", unit="rewrite") as progress:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                results = model.process_many([task for _, task in batch])
                for (key, _), result in zip(batch, results, strict=True):
                    cache.execute("INSERT OR REPLACE INTO rewrites VALUES (?,?)", (key, json.dumps(result, ensure_ascii=False)))
                cache.commit()
                progress.update(len(batch))


def assembly_manifest(stage, examples, rejects, targets, model, plan_manifest, split_report, duplicates):
    groups = [group for row in examples for group in row["contrastive_groups"]]
    count = lambda rows, field: dict(Counter(row[field] for row in rows))
    files = [{"path": path.name, "sha256": file_hash(path), "size_bytes": path.stat().st_size} for path in sorted(stage.iterdir())]
    return {"created_at": datetime.now(timezone.utc).isoformat(), "random_seed": plan_manifest["identity"]["seed"],
            "config": {"source_model": MODEL_NAMES[model], "dataset_pipeline": VERSION, "target_mode": "primary-only",
                       "temporal_taxonomy": "five-level", "location_taxonomy": "four-level",
                       "consensus_threshold": plan_manifest["identity"]["threshold"],
                       "expected_samples": plan_manifest["identity"]["expected_samples"],
                       "dev_fraction_of_development": plan_manifest["identity"]["dev_fraction"],
                       "component_plan_signature": plan_manifest["signature"], "split_report": split_report,
                       "duplicates_removed": duplicates},
            "input_files": [{"path": f"{model}.jsonl", "sha256": plan_manifest["identity"]["inputs"][model],
                             "size_bytes": plan_manifest["identity"]["input_sizes"][model]}],
            "output_files": files, "accepted_examples": len(examples), "rejected_examples": len(rejects),
            "accepted_by_split": count(examples, "split"), "accepted_by_source_model": {MODEL_NAMES[model]: len(examples)},
            "accepted_by_relation": count(examples, "relation"), "rejected_by_reason": count(rejects, "reason"),
            "contrastive_groups": len(groups), "contrastive_candidates": sum(len(g["candidates"]) for g in groups),
            "contrastive_groups_by_modality": count(groups, "modality"), "contrastive_groups_by_target_level": count(groups, "target_level"),
            "positive_primary_targets_by_level": dict(Counter(r["primary_target_level"] for r in examples if r["primary_target_level"])),
            "positive_secondary_targets_by_level": {}, "idk_examples": sum(r["is_idk"] for r in examples),
            "case_targets": len(targets), "case_targets_by_split": count(targets, "split"),
            "case_targets_by_relation": count(targets, "relation"), "case_targets_by_status": count(targets, "target_status"),
            "case_targets_by_target_level": dict(Counter(r["target_level"] for r in targets if r["target_level"]))}


def assemble(args):
    manifest = read_plan(args.work_dir)
    tasks = json.loads((args.work_dir / "tasks.json").read_text())
    cache_path = args.work_dir / "rewrites.sqlite3"
    results = {}
    if cache_path.exists():
        with closing(sqlite3.connect(cache_path.resolve().as_uri() + "?mode=ro", uri=True)) as cache:
            results = {key: json.loads(value) for key, value in cache.execute("SELECT cache_key,result FROM rewrites")}
    if tasks.keys() - results.keys():
        raise ValueError("Rewrites are pending; run the rewrite stage first")
    outputs = {}
    for model in manifest["identity"]["models"]:
        output = args.output_dir / model
        if output.exists():
            saved = json.loads((output / "manifest.json").read_text())
            if saved["config"]["component_plan_signature"] != manifest["signature"]:
                raise ValueError("Output belongs to a different plan")
            for record in saved["output_files"]:
                if file_hash(output / record["path"]) != record["sha256"]:
                    raise ValueError("Existing dataset file changed")
            outputs[model] = saved["accepted_examples"]
            continue
        planned = json.loads((args.work_dir / f"{model}.json").read_text())
        examples, rejects = planned["examples"], planned["rejects"]
        for source in planned["location_sources"]:
            specs, predictions, seen = [], [], set()
            try:
                for variant in sorted(source["variants"], key=lambda item: item["target_level"] != source["target_level"]):
                    task = {**tasks[variant["key"]], "target_level": variant["target_level"], "normalized_value": variant["normalized_value"]}
                    result = results[variant["key"]]
                    text = result["prediction"]
                    if result["status"] != "ok" or validate(task, text) or not balanced_markdown(text):
                        raise ValueError(f"Invalid rewrite for {variant['target_level']}")
                    if text not in seen:
                        specs.append(task)
                        predictions.append(text)
                        seen.add(text)
                candidates = location_candidates(specs, predictions)
                if manifest["identity"]["include_idk_negatives"]:
                    candidates.append({"answer": IDK_ANSWER, "score_start": 0, "score_end": len(IDK_ANSWER), "level": "IDK"})
                groups = [{"modality": "location", "target_level": source["target_level"], "target_index": 0,
                           "candidates": candidates}] if len(candidates) > 1 else []
                examples.append(example(source["row"], source["split"], source["name"], source["target_level"], predictions[0], groups, manifest["identity"]["seed"]))
            except ValueError as error:
                rejects.append(reject(source["row"], "rewriter_invalid_output", str(error)))
        unique = {}
        for row in examples:
            validate_example(row)
            key = (row["qid"], row["relation"], " ".join(unicodedata.normalize("NFKC", row["positive_answer"]).casefold().split()))
            richness = lambda r: (len(r["contrastive_groups"]), sum(len(g["candidates"]) for g in r["contrastive_groups"]))
            if key not in unique or richness(row) > richness(unique[key]):
                unique[key] = row
        duplicates = len(examples) - len(unique)
        examples = sorted(unique.values(), key=lambda row: row["case_id"])
        targets = [{k: v for k, v in row.items() if k not in {"target", "evidence"}} for row in planned["case_targets"]]
        membership = {row["qid"]: row["split"] for row in targets}
        if any(row["split"] != membership[row["qid"]] for row in examples):
            raise ValueError("Person split leakage")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir) as temporary:
            stage = Path(temporary) / model
            stage.mkdir()
            for split in ("train", "dev", "test"):
                write(stage / f"{split}.jsonl", (row for row in examples if row["split"] == split), rows=True)
            write(stage / "case_targets.jsonl", targets, rows=True)
            write(stage / "consensus.jsonl", planned["case_targets"], rows=True)
            write(stage / "rejects.jsonl", rejects, rows=True)
            record = assembly_manifest(stage, examples, rejects, targets, model, manifest, planned["split_report"], duplicates)
            write(stage / "manifest.json", record)
            stage.rename(output)
        outputs[model] = {"examples": len(examples), "duplicates_removed": duplicates,
                          "rewriter_rejections": sum(r["reason"] == "rewriter_invalid_output" for r in rejects)}
    return outputs
