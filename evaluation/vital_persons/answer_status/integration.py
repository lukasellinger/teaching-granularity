"""Opt-in answer-presence overrides; preserve original text and parsed answers."""
import json
from pathlib import Path

from parsing.temporal import is_idk_response
from shared.io import digest, read_rows
from .judge import ROOT, LABELS, make_job


def generation_abstains(generation):
    record = generation.get('answer_status_judge')
    if record is None:
        return is_idk_response(generation['answer'])
    if record['label'] not in LABELS:
        raise ValueError('Unknown answer-status label')
    return record['label'] == 'No answer'


def load_predictions(path):
    path = Path(path)
    config = json.loads((path.parent/'summary.json').read_text())
    prompt = (ROOT/'prompt.txt').read_text()
    if config['prompt_sha256'] != digest(prompt) or config['scope'] != 'full_answer':
        raise ValueError('Status predictions have a different prompt or answer scope')
    index = {}
    for row in read_rows(path):
        key, _ = make_job(row, prompt, config['model'])
        if row['request_key'] != key or row['judge']['key'] != key or row['judge']['label'] not in LABELS:
            raise ValueError('Invalid status prediction identity')
        identity = row['question'], row['answer']
        value = {'label': row['judge']['label'], 'reason': row['judge']['reason'],
                 'request_key': key, 'model': config['model'], 'prompt_sha256': config['prompt_sha256']}
        if identity in index and index[identity] != value:
            raise ValueError('Conflicting judgments for identical question/answer')
        if is_idk_response(row['answer']) and value['label'] != 'No answer':
            raise ValueError('Judge disagrees with a canonical standalone abstention; audit before applying')
        index[identity] = value
    return index


def attach(generations, cases, index, outcomes):
    questions = {c.case_id: c.prompt for c in cases}
    parsed = {r['case_id']: r for r in outcomes}
    rows = []
    for r in generations:
        identity = questions[r['case_id']], r['answer']
        outcome = parsed[r['case_id']]
        if outcome['answer'] != r['answer']:
            raise ValueError('Routing outcome and generation differ')
        if is_idk_response(r['answer']):
            decision = {'label': 'No answer', 'source': 'canonical_idk'}
        elif outcome.get('answer_status_routing_parse_status', outcome['parse_status']) == 'parsed':
            decision = {'label': 'Answer', 'source': 'parsed_answer'}
        else:
            if identity not in index:
                raise ValueError(f'Missing answer-status judgment: {r["case_id"]}; no fallback label applied')
            decision = {**index[identity], 'source': 'judge'}
        rows.append({**r, 'answer_status_judge': decision})
    return rows
