"""Sample person-disjoint location references from twenty-answer generation pools."""

import argparse
from shared.io import read_rows as read_jsonl_rows
import hashlib
import json
import random
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[3]
MODELS = {"lfm": "LFM2_5_1_2B", "qwen": "QWEN3_5_9B", "gemma": "GEMMA4_31B", "ministral": "MINISTRAL3_3B"}
RELATIONS = ("birthplace", "deathplace")


def read_rows(path):
    return read_jsonl_rows(path)


def select(population, excluded_people, *, size=100, seed=20261009):
    strata = list(population)
    if size < len(strata) or not strata:
        raise ValueError("Request at least one sample per model/event stratum")
    rng = random.Random(seed)
    rng.shuffle(strata)
    quotas = {key: size // len(strata) + (index < size % len(strata)) for index, key in enumerate(strata)}
    selected, used = [], set(excluded_people)
    for stratum in strata:
        groups = population[stratum]
        candidates = sorted(groups)
        rng.shuffle(candidates)
        taken = 0
        for pair in candidates:
            if pair[0] in used:
                continue
            selected.append(rng.choice(groups[pair]))
            used.add(pair[0])
            taken += 1
            if taken == quotas[stratum]:
                break
        if taken != quotas[stratum]:
            raise ValueError(f"Insufficient person-disjoint sources for {stratum}")
    rng.shuffle(selected)
    return selected


def sample(samples_dir, metadata, excluded_people, *, models=("lfm", "qwen", "gemma"), size=100, seed=20261009):
    names = {}
    for row in read_rows(metadata):
        if row["qid"] in names or not row["name"].strip():
            raise ValueError("Metadata must have unique qids and non-empty names")
        names[row["qid"]] = row["name"]
    population = {}
    for alias in models:
        model = MODELS[alias]
        for relation in RELATIONS:
            groups = {}
            for row in read_rows(samples_dir / alias / f"{relation}_samples.jsonl"):
                qid = row["qid"]
                if qid not in names or (qid, relation) in groups:
                    raise ValueError("Unknown or duplicate person in generation pool")
                if len(row["samples"]) != 20 or any(not isinstance(answer, str) or not answer.strip() for answer in row["samples"]):
                    raise ValueError("Expected twenty non-empty answers per question")
                question = f"Where was {names[qid]} born?" if relation == "birthplace" else f"Where did {names[qid]} die?"
                groups[qid, relation] = [
                    {"source_model": model, "qid": qid, "relation": relation,
                     "sample_index": index, "answer": answer, "question": question}
                    for index, answer in enumerate(row["samples"])
                ]
            population[model, relation] = groups
    rows = select(population, excluded_people, size=size, seed=seed)
    result = []
    for row in rows:
        identity = {key: row[key] for key in ("source_model", "qid", "relation", "sample_index", "answer")}
        item_id = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
        result.append({"item_id": item_id, **row, "expected_output": None, "reviewed": False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, default=REPO / "data_preparation/vital_persons/samples")
    parser.add_argument("--metadata", type=Path, default=REPO / "data_preparation/vital_persons/metadata/gold_adapted.jsonl")
    parser.add_argument("--excluded-people", type=Path, default=DIRECTORY / "excluded_people.txt")
    parser.add_argument("--models", choices=list(MODELS), nargs="+", default=["lfm", "qwen", "gemma"])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL annotation file")
    args = parser.parse_args()
    if args.output.exists() or len(set(args.models)) != len(args.models):
        parser.error("Output must be new and model names distinct")
    excluded = set(args.excluded_people.read_text().split())
    rows = sample(args.samples_dir, args.metadata, excluded, models=args.models, size=args.sample_size, seed=args.seed)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} blank references written to {args.output}; no model calls made")


if __name__ == "__main__":
    main()
