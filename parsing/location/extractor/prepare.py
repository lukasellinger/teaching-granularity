"""Prepare person-disjoint extractor data from a labelled annotation pool."""

import argparse
import json
import random
import re
from pathlib import Path

from shared.io import read_rows, write, file_hash
from .training_features import CATEGORIES, categorize

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[2]


def person_registry(rows):
    names = {}
    for row in rows:
        for name in (row.get('name'), (row.get('wiki_title') or '').replace('_', ' ')):
            if name:
                key = ' '.join(name.split()).casefold()
                if key in names and names[key] != row['qid']:
                    raise ValueError('Ambiguous person name in metadata')
                names[key] = row['qid']
    return names


def person(row, names):
    patterns = {'birthplace': r'Where was (.+) born\?', 'deathplace': r'Where did (.+) die\?'}
    match = re.fullmatch(patterns.get(row['relation'], r'(?!)'), row['question'].strip(), re.I)
    if not match:
        raise ValueError('Expected a birth/death location question identifying the person')
    name = ' '.join(match[1].split()).casefold()
    group = row.get('qid') or row.get('person_group') or names.get(name, 'name:' + name)
    if not isinstance(group, str) or not group.strip():
        raise ValueError('Missing person identity')
    known = names.get(name)
    if known and group != known:
        raise ValueError('Person group conflicts with the question identity')
    return group


def prepare(annotations, heldout, names=None, *, seed=42, dev_fraction=.2):
    if not 0 < dev_fraction < 1:
        raise ValueError('Development fraction must be between zero and one')
    names = names or {}
    seen_ids, seen_inputs = set(), set()
    prepared = []
    for row in annotations:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ('item_id', 'question', 'answer', 'expected_output')):
            raise ValueError('Every annotation needs an identity, full input and nonempty target')
        categories = row.get('difficulty_categories', categorize(row))
        if not isinstance(categories, list) or len(set(categories)) != len(categories) or set(categories) - set(CATEGORIES):
            raise ValueError('Invalid difficulty categories')
        group = person(row, names)
        identity = row['question'], row['answer']
        if row['item_id'] in seen_ids or identity in seen_inputs:
            raise ValueError('Duplicate annotated identity or input')
        seen_ids.add(row['item_id']); seen_inputs.add(identity)
        prepared.append({**{key: row[key] for key in ('answer', 'expected_output', 'item_id', 'question', 'relation')},
                         'difficulty_categories': categories, 'person_group': group,
                         'reviewed': row.get('reviewed') is True})
    people = sorted({row['person_group'] for row in prepared})
    if len(people) < 2:
        raise ValueError('Need at least two source people')
    random.Random(seed).shuffle(people)
    size = max(1, min(len(people)-1, round(len(people)*dev_fraction)))
    dev_people = set(people[:size])
    used = set(people)
    for row in heldout:
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for key in ('item_id', 'question', 'answer', 'expected_output')):
            raise ValueError('Every held-out reference needs complete inputs and a target')
        group = person(row, names)
        identity = row['question'], row['answer']
        if group in used or row['item_id'] in seen_ids or identity in seen_inputs:
            raise ValueError('Held-out references overlap source people, identities or inputs')
        used.add(group); seen_ids.add(row['item_id']); seen_inputs.add(identity)
    if not heldout:
        raise ValueError('Held-out references must be supplied separately')
    return {'train': [row for row in prepared if row['person_group'] not in dev_people],
            'dev': [row for row in prepared if row['person_group'] in dev_people],
            'heldout': list(heldout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=DIRECTORY/'preparation/annotations.jsonl')
    parser.add_argument('--heldout-annotations', type=Path, default=DIRECTORY/'human_validation/annotations.jsonl')
    parser.add_argument('--metadata', type=Path, default=REPO/'data_preparation/vital_persons/metadata/gold_adapted.jsonl')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-fraction', type=float, default=.2)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Output exists; refusing to overwrite data or annotations')
    splits = prepare(read_rows(args.annotations), read_rows(args.heldout_annotations),
                     person_registry(read_rows(args.metadata)), seed=args.seed, dev_fraction=args.dev_fraction)
    args.output_dir.mkdir(parents=True)
    for split, rows in splits.items():
        write(args.output_dir/f'{split}.jsonl', rows, rows=True)
    manifest = {'seed': args.seed, 'dev_fraction_of_people': args.dev_fraction,
                'annotations_sha256': file_hash(args.annotations),
                'heldout_annotations_sha256': file_hash(args.heldout_annotations),
                'metadata_sha256': file_hash(args.metadata),
                'heldout_used_for_partition_selection': False,
                'counts': {split: len(rows) for split, rows in splits.items()},
                'people': {split: len({person(row, {}) for row in rows}) for split, rows in splits.items()}}
    write(args.output_dir/'manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
