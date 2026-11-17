"""Load the complete split grid, including unavailable consensus targets."""
import json
from pathlib import Path

from shared.io import read_rows
from .cases import (
    VITAL, SOURCE_MODELS, QUESTIONS, Split, Relation, TemporalLevel, LocationLevel,
    TemporalTaxonomy, TemporalEvaluationCase, LocationEvaluationCase,
    parse_gold_temporal, parse_gold_location,
)
from shared.geography import FOUR_LEVEL, FOUR_LEVELS, project_location


def load_cases(model, split='test', *, data_dir=None, metadata=None):
    split = Split(split)
    if split not in {Split.DEV, Split.TEST}:
        raise ValueError('Evaluate dev or test, not training people')
    directory = Path(data_dir) if data_dir else VITAL/'datasets'/model
    metadata = Path(metadata) if metadata else VITAL/'metadata/gold_adapted.jsonl'
    manifest = json.loads((directory/'manifest.json').read_text())
    configuration = manifest.get('config', {})
    if configuration.get('temporal_taxonomy') != 'five-level' or configuration.get('location_taxonomy') != FOUR_LEVEL:
        raise ValueError('This evaluation uses five temporal and four location levels')
    people = {}
    for row in read_rows(metadata):
        if not isinstance(row.get('qid'), str) or not isinstance(row.get('name'), str) or not row['name'].strip() or row['qid'] in people:
            raise ValueError('Gold metadata requires unique people and non-empty names')
        people[row['qid']] = row
    targets, assignments = {}, {}
    for row in read_rows(directory/'case_targets.jsonl'):
        if row['source_model'] != SOURCE_MODELS[model]:
            raise ValueError('Target source model differs from selected model')
        relation = Relation(row['relation'])
        assigned = Split(row['split'])
        if assignments.setdefault(row['qid'], assigned) != assigned:
            raise ValueError('Person appears in different target splits')
        key = row['qid'], relation
        if key in targets:
            raise ValueError('Duplicate person/event target')
        if row['case_id'] != f"{SOURCE_MODELS[model]}:{row['qid']}:{relation.value}":
            raise ValueError('Target case ID does not match its person/event')
        if row['target_status'] not in {'known', 'idk', 'unavailable'}:
            raise ValueError('Unknown consensus target status')
        level = row['target_level']
        if row['target_status'] == 'known':
            allowed = {'DAY', 'MONTH', 'YEAR', 'DECADE', 'CENTURY'} if relation.value in {'birth', 'death'} else {v.value for v in FOUR_LEVELS}
            if level not in allowed:
                raise ValueError('Target level is outside the selected taxonomy')
        elif level is not None:
            raise ValueError('IDK/unavailable targets must not have a target level')
        targets[key] = row
    selected = {qid for qid, assigned in assignments.items() if assigned is split}
    if not selected or selected - people.keys():
        raise ValueError('Empty split or targets missing gold metadata')
    result = {'temporal': [], 'location': []}
    for qid in people:
        if qid not in selected:
            continue
        row = people[qid]
        for relation in Relation:
            target = targets.get((qid, relation))
            if target is None:
                raise ValueError(f'Missing event target: {qid}/{relation.value}')
            common = dict(case_id=target['case_id'], source_model=target['source_model'], split=split,
                          qid=qid, name=row['name'], relation=relation,
                          prompt=QUESTIONS[relation.value].format(person=row['name']), target_status=target['target_status'])
            prefix = 'birth' if relation.value.startswith('birth') else 'death'
            if relation.value in {'birth', 'death'}:
                gold = parse_gold_temporal(row[prefix+'_date'], row[prefix+'_precision']) if row.get(prefix+'_date') else None
                result['temporal'].append(TemporalEvaluationCase(**common, gold=gold,
                    target_level=TemporalLevel(target['target_level']) if target['target_level'] else None,
                    temporal_taxonomy=TemporalTaxonomy.FIVE_LEVEL))
            else:
                gold = project_location(parse_gold_location(row, prefix), FOUR_LEVEL)
                result['location'].append(LocationEvaluationCase(**common, gold=gold,
                    target_level=LocationLevel(target['target_level']) if target['target_level'] else None,
                    location_taxonomy=FOUR_LEVEL))
    return result
