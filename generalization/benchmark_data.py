"""Load benchmark questions and gold answers directly from Hugging Face."""
import ast
from functools import lru_cache
import hashlib
import json
from pathlib import Path

CONFIG = Path(__file__).resolve().with_name('datasets.json')


def dataset_settings(name):
    return json.loads(CONFIG.read_text())[name]


@lru_cache(maxsize=1)
def held_out_relations():
    return set(dataset_settings('granola')['held_out_relations'])


def granola_cases(rows, relations):
    """Original cleaning: unique, contiguous 1–4 answer levels, error score 0."""
    cases = []
    for row in rows:
        if row['relation'] not in relations or row['score_for_potential_error'] != 0:
            continue
        levels = ['' if row.get(f'granola_answer_{i}') is None else str(row[f'granola_answer_{i}'])
                  for i in range(1, 15)]
        answers = [value for value in levels if value]
        if not answers or len(answers) > 4 or len(set(answers)) != len(answers):
            continue
        if any(not value for value in levels[:len(answers)]):
            continue
        identity = str(row['Unnamed: 0'])
        qid = row.get('question_entity_qid')
        known_qid = isinstance(qid, str) and bool(qid)
        cases.append({'id': identity, 'qid': qid if known_qid else f'ROW:{identity}',
            'qid_source': 'question_entity_qid' if known_qid else 'row_id_fallback',
            'question': row['question'].strip(), 'relation': row['relation'],
            'reference_answers': [a.strip() for a in answers]})
    cases.sort(key=lambda r: (0, int(r['id'])) if r['id'].isdigit() else (1, r['id']))
    return cases


def simpleqa_cases(rows):
    cases = []
    for row in rows:
        question = row['problem'].strip()
        metadata = row['metadata']
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = ast.literal_eval(metadata)
        if not isinstance(metadata, dict) or not isinstance(metadata.get('topic'), str) or not metadata['topic'].strip():
            raise ValueError('SimpleQA requires a nonempty metadata.topic')
        metadata = {**metadata, 'topic': metadata['topic'].strip()}
        identity = hashlib.sha256(question.encode()).hexdigest()
        cases.append({'id': identity, 'qid': identity, 'qid_source': 'simpleqa_question_sha256',
            'question': question, 'relation': metadata['topic'], 'topic': metadata['topic'],
            'reference_answer': row['answer'], 'answer_type': metadata.get('answer_type'), 'metadata': metadata})
    return cases


def truthfulqa_cases(rows):
    cases = []
    for row in rows:
        question, category = row['Question'].strip(), row['Category'].strip()
        identity = hashlib.sha256(question.encode()).hexdigest()
        cases.append({'id': identity, 'qid': identity, 'qid_source': 'truthfulqa_question_sha256',
            'question': question, 'relation': category, 'category': category,
            'type': row.get('Type'), 'best_answer': row['Best Answer'],
            'correct_answers': row['Correct Answers'], 'incorrect_answers': row['Incorrect Answers'],
            'source': row.get('Source')})
    return cases


@lru_cache(maxsize=3)
def load_cases(name):
    from datasets import load_dataset
    settings = dataset_settings(name)
    rows = load_dataset(settings['dataset_id'], revision=settings['revision'], split=settings['split'])
    if name == 'granola':
        cases = granola_cases(rows, set(settings['held_out_relations']))
    else:
        cases = {'simpleqa': simpleqa_cases, 'truthfulqa': truthfulqa_cases}[name](rows)
    if not cases or len({r['id'] for r in cases}) != len(cases) or any(not r['question'] or not r['relation'] for r in cases):
        raise ValueError('Expected nonempty questions and relations with unique source IDs')
    return cases
