"""Sample unlabelled filter cases and rebuild the released person-disjoint splits.

Sampling never uses filter labels or predictions. Assembly consumes saved teacher
annotations and selective human corrections; it does not generate new labels.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

DIRECTORY = Path(__file__).resolve().parent
MODELS = ("GEMMA4_31B", "LFM2_5_1_2B", "QWEN3_5_9B")
RELATIONS = ("birth", "death", "birthplace", "deathplace")
STRATA = (("Single", "birthplace"), ("Single", "deathplace"),
          ("Multiple", "birthplace"), ("Multiple", "deathplace"),
          ("Single", "birth"), ("Multiple", "birth"),
          ("Multiple", "death"), ("Single", "death"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def answer_key(row):
    return " ".join(row["answer"].split()).casefold()


def sample(rows, *, source, size, seed, excluded=()):
    """Reproduce the original seeded selection from a complete ordered population.

    Extractor-pool rows carry reference event_extraction. Generation rows must
    carry extraction (including failures) from the complete extractor run.
    """
    if size < (2 if source == "extractor" else 12):
        raise ValueError("Sample size must cover every sampling stratum")
    rng = random.Random(seed)
    used_people = {row["qid"] for row in excluded}
    used_answers = {answer_key(row) for row in excluded}
    selected = []
    if source == "extractor":
        strata = ["birthplace", "deathplace"]
    else:
        strata = [(model, relation) for model in MODELS for relation in RELATIONS]
        random.Random(seed).shuffle(strata)
    quotas = {key: size // len(strata) + (i < size % len(strata)) for i, key in enumerate(strata)}
    for stratum in (strata if source == "extractor" else sorted(strata)):
        candidates = []
        for row in rows:
            if source == "extractor":
                if row["source_relation"] == stratum:
                    candidates.append(row)
            elif (row["source_model"], row["source_relation"]) == stratum:
                result = row["extraction"]
                if result.get("status") == "ok" and result.get("location") not in (None, "", "IDK") and result.get("components"):
                    candidates.append(row)
        if source == "extractor":
            rng.shuffle(candidates)
            groups = [[row] for row in candidates]
        else:
            by_person = {}
            for row in candidates:
                by_person.setdefault(row["qid"], []).append(row)
            people = sorted(by_person)
            rng.shuffle(people)
            groups = [by_person[person] for person in people]
        taken = 0
        for group in groups:
            if group[0]["qid"] in used_people:
                continue
            options = list(group)
            if source != "extractor":
                rng.shuffle(options)
            row = next((row for row in options if answer_key(row) not in used_answers), None)
            if row is None:
                continue
            if source == "extractor":
                identity = {key: row[key] for key in ("source_item_id", "qid", "source_relation", "question", "answer")}
                extraction = row["event_extraction"]
            else:
                identity = {key: row[key] for key in ("source_model", "source_relation", "qid", "sample_index", "question", "answer")}
                extraction = {key: row["extraction"][key] for key in ("location", "components")}
            if not extraction.get("location") or extraction["location"] == "IDK" or not extraction.get("components"):
                raise ValueError("Extractor-pool candidates must have a non-IDK reference extraction")
            fields = ("source_item_id", "qid", "name", "source_model", "source_relation", "event_relation", "sample_index", "question", "answer")
            selected.append({**{key: row[key] for key in fields if key in row},
                             "item_id": digest(identity)[:24], "event_extraction": extraction})
            used_people.add(row["qid"])
            used_answers.add(answer_key(row))
            taken += 1
            if taken == quotas[stratum]:
                break
        if taken != quotas[stratum]:
            raise ValueError(f"Only {taken}/{quotas[stratum]} eligible people for {stratum}")
    rng.shuffle(selected)
    return selected


def assemble(records):
    """Split development annotations 80/20; keep preassigned held-out people aside."""
    development, heldout = [], []
    people, ids, answers = set(), set(), set()
    for record in records:
        row = record["row"]
        if row["expected_output"] not in ("Single", "Multiple"):
            raise ValueError("Every annotation needs an accepted Single or Multiple label")
        if row["qid"] in people or row["item_id"] in ids or answer_key(row) in answers:
            raise ValueError("People, item IDs, and normalized answer texts must be unique")
        people.add(row["qid"]); ids.add(row["item_id"]); answers.add(answer_key(row))
        if record["partition"] == "development":
            development.append(row)
        elif record["partition"] == "heldout":
            heldout.append(row)
        else:
            raise ValueError("Unknown source partition")
    groups = {key: [] for key in STRATA}
    for row in development:
        groups[row.get("split_label", row["expected_output"]), row["source_relation"]].append(row["qid"])
    rng, dev_people = random.Random(42), set()
    for group in groups.values():
        group.sort(); rng.shuffle(group)
        count = round(len(group) * 0.2)
        if len(group) >= 2:
            count = max(1, min(len(group)-1, count))
        dev_people.update(group[:count])
    return {"train": [row for row in development if row["qid"] not in dev_people],
            "dev": [row for row in development if row["qid"] in dev_people], "heldout": heldout}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("assemble", help="Rebuild splits from the released annotated source batches")
    build.add_argument("--annotations", type=Path, default=DIRECTORY / "preparation/annotations.jsonl")
    build.add_argument("--output-dir", type=Path, required=True)
    draw = commands.add_parser("sample", help="Prepare unlabelled cases for teacher annotation and human review")
    draw.add_argument("--population", type=Path, required=True)
    draw.add_argument("--source", choices=("extractor", "generations"), required=True)
    draw.add_argument("--size", type=int, required=True)
    draw.add_argument("--seed", type=int, required=True)
    draw.add_argument("--exclude-items", type=Path, action="append", default=[])
    draw.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Output exists; refusing to overwrite annotations or datasets")
    if args.command == "assemble":
        splits = assemble(read_rows(args.annotations))
        args.output_dir.mkdir(parents=True)
        for split, rows in splits.items():
            write_rows(args.output_dir / f"{split}.jsonl", rows)
        manifest = {"source_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                    "seed": 42, "counts": {key: len(rows) for key, rows in splits.items()},
                    "heldout_used_for_split_selection": False}
    else:
        excluded = [row for path in args.exclude_items for row in read_rows(path)]
        rows = sample(read_rows(args.population), source=args.source, size=args.size, seed=args.seed, excluded=excluded)
        args.output_dir.mkdir(parents=True)
        write_rows(args.output_dir / "items.jsonl", rows)
        tasks = [{"data": {"item_id": row["item_id"], "question": row["question"], "answer": row["answer"],
                            "event_location": row["event_extraction"]["location"]}} for row in rows]
        (args.output_dir / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
        write_rows(args.output_dir / "labels.jsonl", [{"item_id": row["item_id"], "expected_output": None,
                   "annotation_source": None, "human_reviewed": None} for row in rows])
        manifest = {"source": args.source, "population_sha256": hashlib.sha256(args.population.read_bytes()).hexdigest(),
                    "seed": args.seed, "size": args.size, "excluded_people": len({row["qid"] for row in excluded}),
                    "filter_labels_or_predictions_used_for_sampling": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
