"""Matched-data invariants, resumability and offline replay of the paper inputs."""
import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shared.io import read_rows, write
from data_preparation.vital_persons import greedy_control as module
from data_preparation.vital_persons.preparation.realize import example


@pytest.fixture
def args(tmp_path):
    parent = tmp_path / 'parent'
    examples, targets = [], []
    for qid, split in (('Q1', 'train'), ('Q2', 'dev'), ('Q3', 'test')):
        for relation in module.RELATIONS:
            temporal = relation in ('birth', 'death')
            event = 'was born' if relation in ('birth', 'birthplace') else 'died'
            row = {'case_id': f'{module.MODEL}:{qid}:{relation}:2', 'qid': qid, 'source_model': module.MODEL,
                   'source_relation': relation, 'sample_index': 2, 'answer': f"Alice {event} in {'1850' if temporal else 'France'}."}
            level = 'YEAR' if temporal else 'COUNTRY'
            if split != 'test':
                examples.append(example(row, split, 'Alice', level, row['answer'], [], 42))
            targets.append({'case_id': f'{module.MODEL}:{qid}:{relation}', 'qid': qid, 'split': split,
                'source_model': module.MODEL, 'relation': relation, 'sample_count': 20,
                'required_primary_values': 16, 'available_primary_values': 20, 'target_status': 'known', 'target_level': level})
    write(parent / 'manifest.json', {'fixture': True})
    for split in ('train', 'dev', 'test'):
        write(parent / f'{split}.jsonl', [r for r in examples if r['split'] == split], rows=True)
    write(parent / 'case_targets.jsonl', targets, rows=True)
    metadata = tmp_path / 'people.jsonl'
    write(metadata, [{'qid': f'Q{i}', 'name': 'Alice'} for i in (1, 2, 3)], rows=True)
    return SimpleNamespace(parent_data_dir=parent, metadata=metadata, work_dir=tmp_path / 'work',
        output_dir=tmp_path / 'output', revision=module.REVISION, seed=42, max_new_tokens=128,
        batch_size=2, dtype='bfloat16', attention_implementation='sdpa', device='cuda')


def generate(args):
    answers = {'birth': 'Alice was born on May 6, 1900.', 'death': 'IDK',
               'birthplace': 'Alice was born in Berlin, Germany.', 'deathplace': 'Alice traveled to Paris.'}
    by_prompt = {r['prompt']: answers[r['relation']] for r in module.selected_cases(args.parent_data_dir)}
    return module.generate(args, generator=lambda prompts: [{'answer': by_prompt[p]} for p in prompts])


def processed(args):
    rows = read_rows(args.work_dir / 'input.jsonl')
    for row in rows:
        location = row['source_relation'] == 'birthplace'
        row.update(extraction={'status': 'ok', 'location': 'Berlin; Germany' if location else 'IDK',
                               'components': ['Berlin', 'Germany'] if location else []},
                   normalization={'status': 'normalized', 'consensus_value': {
                       'level': 'CITY', 'city': 'Berlin', 'country': 'Germany', 'continent': 'Europe'} if location else None},
                   filter={'filter_status': 'ok', 'filter_prediction': 'Single'})
    write(args.work_dir / 'processed.jsonl', rows, rows=True)


def ready(args, prediction='Alice was born in Berlin.'):
    generate(args)
    processed(args)
    module.plan(args)
    with sqlite3.connect(args.work_dir / 'rewrite/rewrites.sqlite3') as cache:
        cache.execute('CREATE TABLE rewrites(cache_key TEXT PRIMARY KEY, result TEXT NOT NULL)')
        for key in module.read(args.work_dir / 'rewrite/tasks.json'):
            cache.execute('INSERT INTO rewrites VALUES (?, ?)', (key, json.dumps({'status': 'ok', 'prediction': prediction})))


def test_stable_one_realization_and_no_dev_test_generation(args):
    first = module.selected_cases(args.parent_data_dir)[0]
    rows = read_rows(args.parent_data_dir / 'train.jsonl')
    write(args.parent_data_dir / 'train.jsonl', [*rows, {**first, 'case_id': first['case_id'] + ':later'}], rows=True)
    selected = module.selected_cases(args.parent_data_dir)
    assert len(selected) == 4
    assert all(r['qid'] == 'Q1' for r in selected)
    assert first in selected
    generate(args)
    assert {r['prompt'] for r in module.read(args.work_dir / 'requests.json')} == {r['prompt'] for r in selected}
    assert module.generate(args, generator=lambda _: pytest.fail('Must reuse completed generation'))['status'] == 'already complete'
    args.batch_size = 4
    with pytest.raises(ValueError, match='settings/inputs'):
        generate(args)


def test_interruption_reuses_completed_batches(args):
    calls = []
    def interrupted(prompts):
        calls.append(prompts)
        if len(calls) == 2:
            raise RuntimeError('interrupted')
        return [{'answer': 'Alice was born in France.'} for _ in prompts]
    with pytest.raises(RuntimeError, match='interrupted'):
        module.generate(args, generator=interrupted)
    assert len(read_rows(args.work_dir / 'generations.jsonl')) == 2
    resumed = []
    module.generate(args, generator=lambda prompts: resumed.extend(prompts) or [{'answer': 'IDK'} for _ in prompts])
    assert len(resumed) == 2 and not set(resumed) & set(calls[0])


def test_greedy_uses_own_value_and_level_not_consensus(args):
    generate(args)
    processed(args)
    module.plan(args)
    tasks = module.read(args.work_dir / 'rewrite/tasks.json')
    assert next(iter(tasks.values()))['replacement_location'] == 'Berlin'
    entries = module.read(args.work_dir / 'rewrite/entries.json')
    birth = next(r['control'] for r in entries if r['control']['relation'] == 'birth')
    assert birth['primary_target_level'] == 'DAY'
    assert '1900' in birth['positive_answer'] and '1850' not in birth['positive_answer']
    assert next(r['control'] for r in entries if r['control']['relation'] == 'death')['is_idk']
    assert len(module.read(args.work_dir / 'rewrite/paired_rejections.json')) == 1


def test_matched_arms_and_identical_original_dev(args):
    ready(args)
    report = module.assemble(args)
    assert report['shared_train_cases'] == 3
    assert report['granola'] is False
    a, b = (read_rows(args.output_dir / arm / 'train.jsonl') for arm in ('consensus', 'greedy'))
    assert {(r['qid'], r['relation'], r['prompt']) for r in a} == {(r['qid'], r['relation'], r['prompt']) for r in b}
    for arm in ('consensus', 'greedy'):
        for split in ('dev', 'test'):
            assert (args.output_dir / arm / f'{split}.jsonl').read_bytes() == (args.parent_data_dir / f'{split}.jsonl').read_bytes()
    assert all(not r['contrastive_groups'] for r in a + b)
    assert module.assemble(args) == report
    with (args.output_dir / 'greedy/train.jsonl').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Existing dataset changed'):
        module.assemble(args)


def test_bad_rewrite_excludes_both_arms(args):
    ready(args, 'Alice was born in Paris.')
    result = module.assemble(args)
    assert result['shared_train_cases'] == 2
    assert result['paired_rejections'] == 2


def test_missing_rewrites_produce_no_dataset(args):
    generate(args)
    processed(args)
    module.plan(args)
    with pytest.raises(ValueError, match='Rewrites incomplete'):
        module.assemble(args)
    assert not args.output_dir.exists()


def test_empty_answer_is_rejected_not_idk(args):
    module.generate(args, generator=lambda prompts: [{'answer': ''} for _ in prompts])
    write(args.work_dir / 'processed.jsonl', [], rows=True)
    module.plan(args)
    assert not module.read(args.work_dir / 'rewrite/entries.json')
    assert all('Empty generation' in r['reason'] for r in module.read(args.work_dir / 'rewrite/paired_rejections.json'))


def test_operational_normalization_cannot_replace_missing_consensus():
    row = {'source_relation': 'birthplace', 'answer': 'Alice was born in Berlin.',
           'extraction': {'status': 'ok', 'components': ['Berlin']},
           'normalization': {'status': 'normalized', 'value': {'level': 'CITY', 'city': 'Berlin'}, 'consensus_value': None},
           'filter': {'filter_status': 'ok', 'filter_prediction': 'Single'}}
    with pytest.raises(ValueError, match='not IDK'):
        module.own_location(row)


def test_plan_and_parent_change_guard(args):
    generate(args)
    processed(args)
    module.plan(args)
    with (args.work_dir / 'processed.jsonl').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Rewrite inputs/settings changed'):
        module.plan(args)
    with (args.parent_data_dir / 'train.jsonl').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='Parent dataset changed'):
        module.check_generation(args)


def test_help_does_not_load_models():
    result = subprocess.run([sys.executable, '-m', 'data_preparation.vital_persons.greedy_control', '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and 'preprocess' in result.stdout


def test_recorded_preparation_recreates_reported_datasets(tmp_path, require_local_files):
    work = module.REPO / 'ablations/lfm/greedy-control/preparation'
    require_local_files(
        [work/'generation_manifest.json', work/'processed.jsonl', work/'rewrite/rewrites.sqlite3',
         *[work.parent/arm/'data'/name for arm in ('consensus', 'greedy')
           for name in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'case_targets.jsonl')]],
        reason='Recorded matched-control preparation traces and datasets are local-only')
    args = SimpleNamespace(parent_data_dir=module.PARENT, work_dir=work, output_dir=tmp_path / 'replayed',
        seed=42, batch_size=16, dtype='bfloat16', attention_implementation='sdpa', device='cuda')
    report = module.assemble(args)
    assert report['shared_train_cases'] == 2045
    for arm in ('consensus', 'greedy'):
        expected = work.parent / arm / 'data'
        for filename in ('train.jsonl', 'dev.jsonl', 'test.jsonl', 'case_targets.jsonl'):
            assert read_rows(args.output_dir / arm / filename) == read_rows(expected / filename)


def test_recorded_replay_keeps_its_original_rewriter_identity(args, monkeypatch):
    ready(args)
    original = module.read(args.work_dir / 'rewrite/manifest.json')
    monkeypatch.setattr(module, 'SETTINGS', {**module.SETTINGS, 'adapter_revision': 'a' * 40})
    with pytest.raises(ValueError, match='inputs/settings changed'):
        module.plan(args)
    report = module.assemble(args)
    assert report['shared_train_cases'] > 0
    assert module.read(args.work_dir / 'rewrite/manifest.json') == original
