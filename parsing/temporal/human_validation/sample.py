"""Sample base/test temporal evaluations and create a Label Studio audit."""

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .study import DIRECTORY, index_unique


def balanced_sample(population, size=25, seed=42):
    if not 1 <= size <= len(population):
        raise ValueError("Sample size must be positive and no larger than the population")
    groups = defaultdict(list)
    for row in population:
        groups[row["model"]].append(row)
    rng = random.Random(seed)
    # Preserve population/model insertion order: it determines RNG consumption.
    for rows in groups.values():
        rng.shuffle(rows)
    strata = sorted(groups)
    counts = dict.fromkeys(strata, 0)
    for _ in range(size):
        available = [key for key in strata if counts[key] < len(groups[key])]
        minimum = min(counts[key] for key in available)
        candidates = [key for key in available if counts[key] == minimum]
        counts[candidates[rng.randrange(len(candidates))]] += 1
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("Insufficient population for balanced model sampling")
    selected = [row for key in strata for row in groups[key][:counts[key]]]
    rng.shuffle(selected)
    random.Random(seed).shuffle(selected)
    return selected


def load_population(sources):
    population, digest = [], hashlib.sha256()
    seen_models = set()
    for source in sources:
        model, separator, filename = source.partition("=")
        if not separator or not model or not filename or model in seen_models:
            raise ValueError("Use a distinct MODEL=EVALUATED_GENERATIONS.jsonl per model")
        seen_models.add(model)
        path = Path(filename)
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        digest.update(f"{model}\0temporal\0{file_hash}\n".encode())
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"Empty evaluation: {path}")
        for row in rows:
            if (row.get("modality"), row.get("variant"), row.get("split")) != ("temporal", "base_plain", "test"):
                raise ValueError(f"Expected temporal/base_plain/test rows in {path}")
            for field in ("case_id", "prompt", "answer", "prediction_category", "prediction", "matched_values", "parse_status", "relation", "qid"):
                if field not in row:
                    raise ValueError(f"Missing {field} in {path}")
            if row["relation"] not in {"birth", "death"}:
                raise ValueError(f"Unexpected relation in {path}: {row['relation']}")
            key = f"{model}\0temporal\0{row['case_id']}"
            population.append({**row, "model": model, "item_id": hashlib.sha256(key.encode()).hexdigest()[:20]})
    index_unique(population, "item_id")
    return population, digest.hexdigest()[:10]


def make_tasks(selected, study_id):
    tasks, cases = [], []
    for row in selected:
        cases.append({key: row[key] for key in ("item_id", "model", "qid", "relation")})
        pretty = lambda value: json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        tasks.append({"data": {
            "task_uid": f"extract-{study_id}-{row['item_id']}",
            "item_id": row["item_id"], "modality": "temporal",
            "prompt": row["prompt"], "answer": row["answer"],
            "automatic_category": row["prediction_category"],
            "automatic_parse_status": row["parse_status"],
            "raw_extracted_mentions": pretty(row["matched_values"]),
            "normalized_relevant_candidates": pretty(row["matched_values"]),
            "final_prediction": pretty(row["prediction"]),
        }})
    return tasks, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-evaluation", action="append", required=True, metavar="MODEL=FILE")
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True, help="New audit directory; existing directories are not overwritten")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    population, study_id = load_population(args.model_evaluation)
    selected = balanced_sample(population, args.sample_size, args.seed)
    tasks, cases = make_tasks(selected, study_id)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    shutil.copyfile(DIRECTORY / "label_config.xml", args.output_dir / "label_config.xml")
    print(json.dumps({"output_dir": str(args.output_dir), "population": len(population), "sample": len(selected), "by_model": dict(Counter(row["model"] for row in selected)), "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
