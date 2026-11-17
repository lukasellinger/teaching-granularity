"""Read annotations and score processing judgments for the final Base/FT audit."""

import csv
import json
from collections import Counter

from parsing.location.extractor.human_validation.study import wilson
from .audit import CHECKS


def choice(value):
    if isinstance(value, str):
        value = value.strip()
        if value.startswith(('[', '{')):
            value = json.loads(value)
    if isinstance(value, dict):
        value = value.get('choices')
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError('Each judgment requires exactly one choice')
        value = value[0]
    if not isinstance(value, str) or not value:
        raise ValueError('Missing judgment')
    return value


def read_annotations(path):
    if path.suffix.lower() == '.csv':
        with path.open(newline='', encoding='utf-8-sig') as source:
            return list(csv.DictReader(source))
    tasks = json.loads(path.read_text())
    if not isinstance(tasks, list):
        raise ValueError('Expected a Label Studio JSON task export')
    rows = []
    for task in tasks:
        completed = [a for a in task.get('annotations', []) if a.get('result') and not a.get('was_cancelled')]
        if len(completed) != 1:
            raise ValueError('Require one completed annotation per task; adjudicate duplicates first')
        annotation = completed[0]
        row = {**task['data'], 'annotation_id': annotation.get('id'), 'annotator': annotation.get('completed_by')}
        for result in annotation['result']:
            key = result['from_name']
            if key in row:
                raise ValueError('Duplicate judgment field')
            row[key] = result['value'].get('choices', result['value'].get('text', ''))
        rows.append(row)
    return rows


def evaluate(cases, annotations):
    expected = {row['item_id']: row for row in cases}
    if len(expected) != len(cases) or not expected:
        raise ValueError('Missing or duplicate frozen case identity')
    index = {}
    for row in annotations:
        item_id = row.get('item_id') or row.get('task_uid', '').removeprefix('location-audit-')
        if not item_id or item_id in index:
            raise ValueError('Missing or duplicate annotated case identity')
        index[item_id] = row
    if index.keys() != expected.keys():
        raise ValueError('Annotations must cover exactly the frozen audit; no missing or extra cases')
    judgments = []
    for item_id, case in expected.items():
        row = index[item_id]
        if any(row.get(key) != case[key] for key in ('question', 'answer')):
            raise ValueError('Annotation input differs from the frozen question/answer')
        outcome = choice(row.get('review_outcome'))
        labels = {}
        if outcome == 'All checks correct':
            for field in CHECKS:
                if row.get(field) and choice(row[field]) not in ('Correct', 'Not applicable'):
                    raise ValueError('Overall shortcut conflicts with a detailed judgment')
                labels[field] = ('Not applicable' if field != 'extraction_correct'
                                 and case['normalization']['status'] in ('idk', 'skipped') else 'Correct')
        elif outcome == 'Needs detailed review':
            for field in CHECKS:
                labels[field] = choice(row.get(field))
                allowed = {'Correct', 'Incorrect', 'Cannot determine'}
                if field != 'extraction_correct':
                    allowed.add('Not applicable')
                if labels[field] not in allowed:
                    raise ValueError('Unknown component judgment')
            if not any(v in ('Incorrect', 'Cannot determine') for v in labels.values()):
                raise ValueError('Detailed review has no incorrect or unresolved judgment')
        else:
            raise ValueError('Unknown overall review choice')
        verdict = ('incorrect' if 'Incorrect' in labels.values() else
                   'unresolved' if 'Cannot determine' in labels.values() else 'correct')
        judgments.append({'item_id': item_id, 'verdict': verdict, 'judgments': labels,
                          'annotation_id': row.get('annotation_id'), 'annotator': row.get('annotator'),
                          'notes': row.get('notes', '')})
    counts = Counter(row['verdict'] for row in judgments)
    return {'cases': len(cases), 'confirmed_correct': counts['correct'], 'incorrect': counts['incorrect'],
            'unresolved': counts['unresolved'], 'confirmed_correct_fraction': counts['correct'] / len(cases),
            'wilson_95': wilson(counts['correct'], len(cases)),
            'denominator': 'All frozen cases; unresolved judgments remain in the denominator.',
            'component_judgments': {field: dict(Counter(row['judgments'][field] for row in judgments)) for field in CHECKS},
            'judgments': judgments}
