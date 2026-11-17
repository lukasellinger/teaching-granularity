"""Frozen split metadata and local reconstruction integrity checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from shared.constants import MODEL_NAMES
from shared.io import digest
from . import restore


def test_released_assignments_have_no_benchmark_text_and_preserve_splits():
    allowed = {'original_row_id', 'qid', 'relation', 'split', 'selected_level', 'example_sha256'}
    for model in MODEL_NAMES:
        directory = restore.ROOT/'datasets'/model
        public = json.loads((directory/'split_assignments.json').read_text())
        manifest = json.loads((directory/'manifest.json').read_text())
        rows = public['assignments']
        assert all(set(r) == allowed for r in rows)
        assert len({r['original_row_id'] for r in rows}) == len(rows)
        assert dict(Counter(r['split'] for r in rows)) == manifest['accepted_by_split']
        splits = {}
        for row in rows:
            assert splits.setdefault(row['qid'], row['split']) == row['split']
            assert (row['split'] == 'test') == (row['qid'] in manifest['held_out_entity_qids'])
        train_qids = {q for q, s in splits.items() if s != 'test'}
        assert {q for q, s in splits.items() if s == 'dev'} == restore.development_qids(train_qids)


def fixture(directory):
    # A minimal frozen training record with a known expected complete response.
    raw = [{'id': 7, 'question': 'Who founded Example?', 'question_entity': 'Example',
            'question_entity_qid': 'Q7', 'relation': 'P112', 'score_for_potential_error': 0,
            'granola_answer_1': 'Ada Example', 'granola_answer_2': 'A founder'}]
    assignment = {'original_row_id': '7', 'qid': 'Q7', 'relation': 'P112',
                  'split': 'train', 'selected_level': 4.0}
    source = restore.prepare_rows([{**raw[0], 'Unnamed: 0': 7}])[0]
    example = restore.build_example(assignment, source, MODEL_NAMES['lfm'])
    assert example['positive_answer'] == 'Example was founded by a founder.'
    assignment['example_sha256'] = digest(example)
    payloads = {'train': (json.dumps(example, ensure_ascii=False)+'\n').encode(), 'dev': b'', 'test': b''}
    manifest = {'source_model': MODEL_NAMES['lfm'], 'held_out_entity_qids': [],
                'held_out_relations': ['P50', 'P26', 'P127', 'P264'], 'seed': 42,
                'dev_fraction': 0.1, 'accepted_by_split': {'train': 1},
                'output_files': [{'path': split+'.jsonl', 'sha256': hashlib.sha256(payload).hexdigest()}
                                 for split, payload in payloads.items()]}
    public = {'schema_version': 1, 'model': 'lfm', 'source_model': MODEL_NAMES['lfm'],
              'dataset': restore.dataset_settings('granola'), 'assignments': [assignment]}
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'split_assignments.json').write_text(json.dumps(public))
    return raw


def test_local_reconstruction_checks_source_and_never_overwrites_different_data(tmp_path):
    raw = fixture(tmp_path)
    out = tmp_path/'restored'
    counts = restore.restore(tmp_path, out, raw, 'lfm')
    assert counts == {'train.jsonl': 1, 'dev.jsonl': 0, 'test.jsonl': 0}
    assert restore.restore(tmp_path, out, raw, 'lfm') == counts
    (out/'train.jsonl').write_text('existing local data\n')
    with pytest.raises(ValueError, match='Refusing to overwrite'):
        restore.restore(tmp_path, out, raw, 'lfm')
    assert (out/'train.jsonl').read_text() == 'existing local data\n'
    raw[0]['granola_answer_2'] = 'A changed answer'
    with pytest.raises(ValueError, match='Reconstructed example differs'):
        restore.restore(tmp_path, tmp_path/'must-not-exist', raw, 'lfm')
    assert not (tmp_path/'must-not-exist').exists()


def test_reconstruction_rejects_changed_ids_and_selected_levels(tmp_path):
    raw = fixture(tmp_path)
    public_path = tmp_path/'split_assignments.json'
    public = json.loads(public_path.read_text())
    public['assignments'][0]['selected_level'] = 1.0
    public_path.write_text(json.dumps(public))
    with pytest.raises(ValueError, match='Reconstructed example differs'):
        restore.reconstruct(tmp_path, raw, 'lfm')
    public['assignments'][0]['original_row_id'] = '8'
    public_path.write_text(json.dumps(public))
    with pytest.raises(ValueError, match='Missing cleaned source row'):
        restore.reconstruct(tmp_path, raw, 'lfm')
