"""Replay the original source sampling and build person-disjoint rewriter data.

The population freezes eligible answers and supplied replacement values from
completed normalization/consensus groups. Sampling never uses reference rewrites.
Assembly joins separately saved teacher annotations after selecting the people.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

DIRECTORY = Path(__file__).resolve().parent
MODELS = ('LFM2_5_1_2B', 'QWEN3_5_9B')
LEVELS = ('PLACE', 'CITY', 'REGION', 'COUNTRY', 'CONTINENT')
HELDOUT_FIELDS = ('item_id', 'question', 'answer', 'expected_output',
                  'original_location', 'original_components', 'replacement_location',
                  'target_level', 'qid', 'output_scope')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows), encoding='utf-8')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample(population, *, seed=43, people_per_model=100):
    """Sort source IDs, shuffle once, then enforce unique people and model quotas."""
    if people_per_model < 1:
        raise ValueError('people_per_model must be positive')
    ids = [row['parent_item_id'] for row in population]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source-answer identity')
    for row in population:
        if row['source_model'] not in MODELS or row['person_group'] != row['qid']:
            raise ValueError('Unexpected source model or person identity')
        if row['relation'] not in ('birthplace', 'deathplace') or not row['answer'].strip():
            raise ValueError('Missing source answer or wrong relation')
        levels = [target['target_level'] for target in row['rewrites']]
        if not levels or len(set(levels)) != len(levels) or any(level not in LEVELS for level in levels):
            raise ValueError('Missing, duplicate or unexpected replacement level')
        if any('expected_output' in target for target in row['rewrites']):
            raise ValueError('The sampling population must not contain reference rewrites')
    candidates = sorted(population, key=lambda row: row['parent_item_id'])
    random.Random(seed).shuffle(candidates)
    selected, people, counts = [], set(), Counter()
    for row in candidates:
        if row['qid'] in people or counts[row['source_model']] >= people_per_model:
            continue
        selected.append(row)
        people.add(row['qid'])
        counts[row['source_model']] += 1
        if len(selected) == people_per_model*len(MODELS):
            break
    if len(selected) != people_per_model*len(MODELS):
        raise ValueError(f'Insufficient person-disjoint sources for model quotas: {dict(counts)}')
    size = max(1, round(len(selected)*0.2))
    if 2*size >= len(selected):
        raise ValueError('Too few people for three nonempty partitions')
    return {'heldout': selected[:size], 'dev': selected[size:2*size], 'train': selected[2*size:]}


def assemble(population, annotations, *, seed=43, people_per_model=100):
    cohorts = sample(population, seed=seed, people_per_model=people_per_model)
    labels = {row['parent_item_id']: row for row in annotations}
    if len(labels) != len(annotations):
        raise ValueError('Duplicate annotated source identity')
    if set(labels) != {row['parent_item_id'] for rows in cohorts.values() for row in rows}:
        raise ValueError('Annotations must cover exactly the sampled source answers')
    splits, seen_inputs, seen_ids = {}, {}, set()
    for split in ('train', 'dev', 'heldout'):
        expanded = []
        for source in cohorts[split]:
            expected = labels[source['parent_item_id']]['rewrites']
            if set(expected) != {target['target_level'] for target in source['rewrites']}:
                raise ValueError('Reference annotations do not cover the supplied replacement levels')
            for target in source['rewrites']:
                answer = expected[target['target_level']]
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError('Each replacement requires a nonempty teacher reference')
                row = {**{key: value for key, value in source.items() if key != 'rewrites'},
                       **target, 'expected_output': answer,
                       'item_id': source['parent_item_id']+':'+target['target_level'],
                       'output_scope': 'complete-sentence-output-v1',
                       'upstream_outputs_taken_as_given': True}
                if row['item_id'] in seen_ids:
                    raise ValueError('Duplicate task identity')
                seen_ids.add(row['item_id'])
                identity = row['question'], row['answer'], row['replacement_location']
                previous = seen_inputs.setdefault(identity, (split, answer))
                if previous != (split, answer):
                    raise ValueError('Identical rewrite input has conflicting references or crosses splits')
                expanded.append({key: row[key] for key in HELDOUT_FIELDS} if split == 'heldout' else row)
        splits[split] = expanded
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sample', 'assemble'))
    parser.add_argument('--population', type=Path, default=DIRECTORY/'preparation/population.jsonl')
    parser.add_argument('--annotations', type=Path, default=DIRECTORY/'preparation/annotations.jsonl')
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--people-per-model', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Output exists; refusing to overwrite annotations or datasets')
    population = read_rows(args.population)
    settings = {'seed': args.seed, 'people_per_model': args.people_per_model}
    if args.command == 'assemble':
        splits = assemble(population, read_rows(args.annotations), **settings)
        args.output_dir.mkdir(parents=True)
        for split, rows in splits.items():
            write_rows(args.output_dir/f'{split}.jsonl', rows)
        annotation_hash = file_hash(args.annotations)
    else:
        splits = sample(population, **settings)
        args.output_dir.mkdir(parents=True)
        items, labels = [], []
        for split, sources in splits.items():
            for source in sources:
                items.append({'partition': split, 'row': source})
                labels.append({'parent_item_id': source['parent_item_id'],
                               'rewrites': {target['target_level']: None for target in source['rewrites']},
                               'annotation_source': None, 'human_reviewed': None})
        write_rows(args.output_dir/'items.jsonl', items)
        write_rows(args.output_dir/'annotations.jsonl', labels)
        annotation_hash = None
    manifest = {**settings, 'population_sha256': file_hash(args.population),
                'annotations_sha256': annotation_hash,
                'reference_labels_used_for_sampling': False,
                'counts': {split: len(rows) for split, rows in splits.items()},
                'people': {split: len({row['qid'] for row in rows}) for split, rows in splits.items()}}
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
