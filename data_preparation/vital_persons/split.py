"""Create the shared person-level development/test split from gold metadata."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

from shared.io import read_rows, write
from .metadata import ROOT, QID


def present(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return False
    return bool(value) if isinstance(value, (list, tuple, dict, str)) else True


def strata(person):
    labels = []
    for event in ('birth', 'death'):
        precision = person.get(event+'_precision')
        precision = str(int(precision)) if present(precision) and float(precision).is_integer() else str(precision)
        labels.append(f'{event}|precision={precision}' if present(person.get(event+'_date')) else f'{event}|not_present')
        available = any(present(person.get(event+'_'+field)) for field in ('place', 'city', 'region', 'country', 'continent'))
        ambiguous = person.get(event+'_needs_disambiguation')
        ambiguous = bool(ambiguous) if present(ambiguous) else False
        labels.append(f'{event}place|needs_disambiguation={str(ambiguous).lower()}' if available else f'{event}place|not_present')
    return labels


def test_people(people, *, fraction=.1, trials=5000, seed=42):
    """Balanced subset search with hash-dependent swap-candidate ordering."""
    groups = [p['qid'] for p in people]
    if len(groups) < 2 or len(set(groups)) != len(groups) or any(not QID.fullmatch(q) for q in groups):
        raise ValueError('Need at least two people with unique valid QIDs')
    if not 0 < fraction < 1 or trials < 1:
        raise ValueError('Test fraction must be between 0 and 1; trials must be positive')
    labels = {p['qid']: strata(p) for p in people}
    totals = Counter(label for q in groups for label in labels[q])
    size = min(max(round(fraction*len(groups)), 1), len(groups)-1)

    def score(counts):
        return sum((counts[label]-fraction*total)**2 / max(fraction*total*(1-fraction), 1.)
                   for label,total in totals.items())

    rng, best, best_score = random.Random(seed), None, float('inf')
    for _ in range(trials):
        candidate = set(rng.sample(groups, size))
        counts = Counter(label for q in candidate for label in labels[q])
        loss = score(counts)
        if loss < best_score:
            best, best_score = candidate, loss
            if loss == 0:
                break
    # Seed alone does not fix set iteration across Python processes.
    selected = list(best)
    remaining = [q for q in groups if q not in best]
    counts = Counter(label for q in selected for label in labels[q])
    for _ in range(trials*20):
        i, j = rng.randrange(len(selected)), rng.randrange(len(remaining))
        removed, added = selected[i], remaining[j]
        candidate_counts = counts.copy()
        candidate_counts.subtract(labels[removed])
        candidate_counts.update(labels[added])
        loss = score(candidate_counts)
        if loss < best_score:
            best_score, counts = loss, candidate_counts
            selected[i], remaining[j] = added, removed
    return set(selected)


def prepare(metadata_path, output_dir, *, fraction=.1, trials=5000, seed=42, overwrite=False):
    people = read_rows(metadata_path)
    selected = test_people(people, fraction=fraction, trials=trials, seed=seed)
    outputs = {'development.jsonl': [{k:p[k] for k in ('qid', 'name', 'wiki_title')} for p in people if p['qid'] not in selected],
               'test.jsonl': [{k:p[k] for k in ('qid', 'name', 'wiki_title')} for p in people if p['qid'] in selected]}
    for name, rows in outputs.items():
        path = output_dir/name
        if path.exists() and not overwrite and read_rows(path) != rows:
            raise ValueError(f'{path} differs; choose a new --output-dir or explicitly --overwrite')
    for name, rows in outputs.items():
        write(output_dir/name, rows, rows=True)
    summary = {'people': len(people), 'development': len(people)-len(selected), 'test': len(selected),
               'seed': seed, 'trials': trials, 'test_fraction': fraction}
    print(json.dumps(summary, indent=2))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, default=ROOT/'metadata/gold_adapted.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'metadata')
    parser.add_argument('--test-fraction', type=float, default=.1)
    parser.add_argument('--trials', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.metadata, args.output_dir, fraction=args.test_fraction, trials=args.trials,
            seed=args.seed, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
