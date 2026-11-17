from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shared.io import read_rows, write
from evaluation.vital_persons.cases import VITAL
from evaluation.vital_persons.data import load_cases
from evaluation.vital_persons.generate import drag_samples, variant_prompt, validate_records, run, settings
from evaluation.vital_persons.report import score


@pytest.mark.parametrize('model', ['lfm', 'qwen', 'gemma', 'ministral'])
def test_full_model_specific_grid_includes_unavailable_targets(model):
    for split in ('dev', 'test'):
        cases = load_cases(model, split)
        assert {key: len(rows) for key, rows in cases.items()} == {'temporal': 328, 'location': 328}
        people = {c.qid for c in cases['temporal']}
        assert people == {c.qid for c in cases['location']}
        assert all(c.split == split for rows in cases.values() for c in rows)
        assert any(c.target_status == 'unavailable' for rows in cases.values() for c in rows)
    assert {c.qid for c in load_cases(model, 'test')['temporal']}.isdisjoint(c.qid for c in load_cases(model, 'dev')['temporal'])


def test_drag_uses_all_twenty_original_samples():
    cases = load_cases('lfm')['location'][:2]
    samples = drag_samples('lfm', cases)
    for case in cases:
        pool = read_rows(VITAL/'samples/lfm'/f'{case.relation.value}_samples.jsonl')
        assert samples[case.case_id] == next(row['samples'] for row in pool if row['qid'] == case.qid)
        prompt, system = variant_prompt('base_drag', case.prompt, samples[case.case_id])
        assert all(sample in prompt for sample in samples[case.case_id])
        assert system is None
    with pytest.raises(ValueError):
        variant_prompt('base_drag', cases[0].prompt, ['one answer'])


def generation(case, variant='base_plain', answer='IDK'):
    prompt, system = variant_prompt(variant, case.prompt)
    return {'case_id': case.case_id, 'variant': variant, 'user_prompt': prompt, 'system_prompt': system,
            'answer': answer, 'generated_tokens': 1, 'finish_reason': 'eos'}


def test_wrong_question_duplicate_and_incomplete_generations_rejected():
    cases = load_cases('lfm')['temporal'][:2]
    rows = [generation(c) for c in cases]
    validate_records(rows, cases, 'base_plain', complete=True)
    for bad in ([rows[0]], rows+[rows[0]], [{**rows[0], 'user_prompt': 'Wrong question'}, rows[1]]):
        with pytest.raises(ValueError):
            validate_records(bad, cases, 'base_plain', complete=True)


def test_scoring_needs_no_model_or_adapter_and_writes_simple_report(tmp_path):
    cases = load_cases('lfm')['temporal'][:4]
    inputs = tmp_path/'generations'
    for variant in ('base_plain', 'finetuned'):
        write(inputs/'temporal'/f'{variant}.jsonl', [generation(c, variant) for c in cases], rows=True)
    result = score({'temporal': cases}, ['base_plain', 'finetuned'], inputs, tmp_path/'result', num_resamples=5)
    assert result['modalities']['temporal']['variants']['base_plain']['metrics']['cases'] == 4
    assert (tmp_path/'result/summary.txt').is_file()
    assert (tmp_path/'result/outcomes/temporal/finetuned.jsonl').is_file()
    subprocess.run([sys.executable, '-m', 'evaluation.vital_persons', '--model', 'lfm', '--stage', 'score',
        '--modality', 'temporal', '--variants', 'base_plain', 'finetuned', '--adapter', '/nonexistent/not-needed',
        '--limit', '4', '--generation-dir', str(inputs), '--output-dir', str(tmp_path/'cli'), '--bootstrap-samples', '0'], check=True, capture_output=True)


def test_complete_generation_resume_never_loads_weights(tmp_path, monkeypatch):
    from evaluation.vital_persons import generate
    cases = load_cases('lfm')['temporal'][:2]
    config = settings('lfm')
    directory = tmp_path/'generations'
    path = directory/'temporal/base_plain.jsonl'
    condition = [variant_prompt('base_plain', c.prompt) for c in cases]
    from evaluation.vital_persons.pipeline import digest
    signature = {'settings': config, 'device': 'cuda', 'variant': 'base_plain',
                 'tokenizer_source': 'base',
                 'questions': digest([(c.case_id, pair) for c, pair in zip(cases, condition)])}
    write(path, [generation(c) for c in cases], rows=True)
    write(path.with_suffix('.run.json'), {'signature': signature})
    monkeypatch.setattr(generate, 'load_model', lambda *a, **k: pytest.fail('Cached generation loaded weights'))
    run({'temporal': cases}, ['base_plain'], directory, config)
    with pytest.raises(ValueError, match='changed'):
        run({'temporal': cases}, ['base_plain'], directory, {**config, 'batch_size': 1})


def test_metadata_may_change_gold_but_not_split_membership(tmp_path):
    original = load_cases('lfm')
    metadata = read_rows(VITAL/'metadata/gold_adapted.jsonl')
    chosen = next(row for row in metadata if row['qid'] == original['temporal'][0].qid)
    chosen['birth_date'] = '+1900-01-01T00:00:00Z'
    chosen['birth_precision'] = 11
    path = tmp_path/'metadata.jsonl'
    write(path, metadata, rows=True)
    updated = load_cases('lfm', metadata=path)
    assert [c.case_id for c in original['temporal']] == [c.case_id for c in updated['temporal']]
    assert updated['temporal'][0].gold.year == 1900


def test_generation_disables_adapter_for_baselines_and_resumes_without_weights(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from contextlib import contextmanager
    from evaluation.vital_persons import generate
    from shared import generation as generation_utils
    cases = load_cases('lfm')['temporal'][:3]
    class Model:
        disabled = False
        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False
    model = Model()
    calls = []
    def fake_generate(current, tokenizer, prompts, **kwargs):
        assert current is model
        calls.append((model.disabled, len(prompts)))
        return [generation_utils.GeneratedAnswer('IDK', 1, 'eos') for prompt in prompts]
    monkeypatch.setattr(generate, 'load_model', lambda *a, **k: (model, None))
    monkeypatch.setattr(generation_utils, 'generate_greedy', fake_generate)
    config = settings('lfm', batch_size=2)
    generate.run({'temporal': cases}, ['base_plain', 'finetuned'], tmp_path, config, device='cpu')
    assert calls == [(True, 2), (True, 1), (False, 2), (False, 1)]
    monkeypatch.setattr(generate, 'load_model', lambda *a, **k: pytest.fail('Complete run loaded a model'))
    generate.run({'temporal': cases}, ['base_plain', 'finetuned'], tmp_path, config, device='cpu')


@pytest.mark.parametrize('person,event,city', [
    ('Q79759', 'birth', 'London'),       # Cheapside street
    ('Q7259', 'death', 'London'),       # Marylebone / City of Westminster
    ('Q61064', 'death', 'Neuilly-sur-Seine'),
    ('Q33391', 'death', 'Mexico City'),
    ('Q183337', 'birth', 'Isleworth'),  # town remains CITY
])
def test_reviewed_gold_matches_shared_city_representation(person, event, city):
    from evaluation.vital_persons.cases import parse_gold_location
    from evaluation.vital_persons.location import correct_at_prediction_level
    from parsing.location.normalization.values import LocationValue, LocationLevel
    from shared.geography import project_location, FOUR_LEVEL
    row = next(p for p in read_rows(VITAL/'metadata/gold_adapted.jsonl') if p['qid'] == person)
    gold = project_location(parse_gold_location(row, event), FOUR_LEVEL)
    prediction = LocationValue(level=LocationLevel.CITY, city=city, country=gold.country)
    assert correct_at_prediction_level(prediction, gold)
    wrong = replace(prediction, city='Rome' if city != 'Rome' else 'Pavia')
    assert not correct_at_prediction_level(wrong, gold)


def test_london_gold_preserves_distinct_subcities():
    rows = {p['qid']: p for p in read_rows(VITAL/'metadata/gold_adapted.jsonl')}
    assert rows['Q79759']['birth_subcity'] == ['City of London']
    assert rows['Q7259']['death_subcity'] == ['Marylebone', 'City of Westminster']
    assert rows['Q79759']['birth_entity_ids']['subcity'] == ['Q23311']
    assert rows['Q7259']['death_entity_ids']['subcity'] == ['Q123634', 'Q179351']
