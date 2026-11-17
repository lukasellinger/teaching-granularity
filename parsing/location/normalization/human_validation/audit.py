"""Exclusion and frozen-input helpers for the final Base/FT geographic audit."""
import json
from pathlib import Path

from shared.io import read_rows, file_hash

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parents[3]
LOCATION = REPO / 'parsing/location'
CHECKS = ('extraction_correct', 'entity', 'hierarchy', 'granularity', 'omissions', 'consensus')


def question_key(text):
    return ' '.join(text.split()).casefold()


def check_files(directory, manifest):
    for name, expected in manifest['files'].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f'Frozen audit file changed: {name}')


def exclusion_questions(extra_items=()):
    registry = json.loads((DIRECTORY / 'prior_questions.json').read_text())
    questions = {question_key(q) for q in registry['questions']}
    for snapshot in sorted((DIRECTORY / 'audits').glob('*/exclusions.json')):
        questions.update(json.loads(snapshot.read_text()).get('questions', []))
    paths = [LOCATION / 'extractor' / name for name in (
        'training/train.jsonl', 'training/dev.jsonl', 'human_validation/annotations.jsonl')]
    paths.append(LOCATION / 'normalization/tests/fixtures/regressions.jsonl')
    paths.extend(sorted((DIRECTORY / 'audits').glob('*/items.jsonl')))
    paths.extend(extra_items)
    # Saved normalization corrections were inspected too; exclude their questions.
    corrections = REPO / 'evaluation/vital_persons/corrections.jsonl'
    inspected_answers = {row['answer'] for row in read_rows(corrections)}
    for path in sorted((REPO / 'evaluation/vital_persons/results').glob('*/*/location_extractions/*.jsonl')):
        questions.update(question_key(row['question']) for row in read_rows(path)
                         if row['answer'] in inspected_answers)
    for path in paths:
        questions.update(question_key(row['question']) for row in read_rows(path))
    return questions
