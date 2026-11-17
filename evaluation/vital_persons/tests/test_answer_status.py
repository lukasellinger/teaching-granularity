import csv
from collections import Counter
import json

import pytest

from shared.io import digest, write
from evaluation.vital_persons.answer_status import audit, judge, integration
from evaluation.vital_persons.pipeline import parse_component_response
from evaluation.vital_persons.tests.test_temporal_evaluation import case, generation
from evaluation.vital_persons.cases import Relation, TemporalValue, TemporalLevel
from evaluation.vital_persons.temporal import score_temporal_generations


def item(answer='The date is not documented.'):
    return {'item_id': 'one', 'question': 'When was she born?', 'answer': answer,
            'partition': 'evaluation', 'model': 'lfm', 'cohort': 'stratified'}


def test_request_contains_no_hidden_metadata_and_identity_tracks_changes():
    r = item()
    key, request = judge.make_job(r, 'instructions')
    assert key == judge.make_job({**r, 'gold': '1900', 'model': 'gemma'}, 'instructions')[0]
    assert set(json.loads(request['input'])) == {'question', 'response'}
    assert request['model'] == 'gpt-6-luna'
    assert key != judge.make_job(r, 'new instructions')[0]
    assert key != judge.make_job(item('She was born in 1900.'), 'instructions')[0]


def test_batch_preparation_is_offline_and_deduplicates(tmp_path):
    r = item()
    state = judge.prepare_batch([r, {**r, 'item_id': 'two'}], 'p', judge.MODEL,
                                tmp_path/'cache.jsonl', tmp_path/'batch')
    assert state['items'] == 2
    assert state['unique_requests'] == 1
    assert len(judge.read_rows(tmp_path/'batch/input.jsonl')) == 1
    with pytest.raises(ValueError, match='already'):
        judge.prepare_batch([r], 'p', judge.MODEL, tmp_path/'cache.jsonl', tmp_path/'batch')


def test_incomplete_invalid_or_refused_output_is_not_an_abstention():
    good = {'status': 'completed', 'output': [{'type': 'message', 'content': [
        {'type': 'output_text', 'text': json.dumps({'label': 'No answer', 'reason': 'No date supplied.'})}]}]}
    assert judge.parse_body(good)['label'] == 'No answer'
    with pytest.raises(ValueError, match='Incomplete'):
        judge.parse_body({**good, 'status': 'incomplete'})
    with pytest.raises(ValueError):
        judge.parse_body({'status': 'completed', 'output': []})
    with pytest.raises(ValueError):
        judge.parse_body({'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': '{"label":"Unknown","reason":"x"}'}]}]})


def test_score_never_writes_partial_predictions(tmp_path):
    with pytest.raises(ValueError, match='missing'):
        judge.score([item()], 'p', judge.MODEL, tmp_path/'absent.jsonl', tmp_path/'out')
    assert not (tmp_path/'out/predictions.jsonl').exists()


def test_human_labels_require_full_matching_export(tmp_path):
    path = tmp_path/'annotations.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['item_id', 'question', 'answer', 'answer_status'])
        writer.writeheader()
        writer.writerow({**{k: item()[k] for k in ('item_id', 'question', 'answer')}, 'answer_status': 'No answer'})
    assert audit.read_labels(path, [item()]) == {'one': 'No answer'}
    with pytest.raises(ValueError, match='differs'):
        audit.read_labels(path, [item('Different response')])


def test_metrics_report_false_and_missed_no_answers():
    result = audit.metrics(['Answer', 'No answer', 'No answer', 'Answer'], ['No answer', 'No answer', 'Answer', 'Answer'])
    assert result['agreement'] == .5
    assert result['false_no_answers'] == result['missed_no_answers'] == 1
    assert result['no_answer_precision'] == result['no_answer_recall'] == .5


def test_temporal_abstention_is_opt_in_and_keeps_original_response():
    c = case('Q1', Relation.BIRTH, TemporalValue(level=TemporalLevel.YEAR, year=1900))
    g = generation(c.case_id, 'Her birth date is not documented.')
    _, old = score_temporal_generations([c], [g])
    assert old[0]['parse_status'] == 'unparsed'
    assert old[0]['utility_uniform_wrong_penalty'] == -1
    _, new = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'No answer'}}])
    assert new[0]['parse_status'] == 'idk'
    assert new[0]['utility_uniform_wrong_penalty'] == 0
    assert new[0]['answer'] == g['answer']
    assert new[0]['truth_correct'] is False


def test_coarse_hedged_and_failed_answers_keep_existing_parser():
    c = case('Q1', Relation.BIRTH, TemporalValue(level=TemporalLevel.YEAR, year=1900))
    g = generation(c.case_id, 'The exact day is unknown, but she was born in 1900.')
    _, rows = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'Answer'}}])
    assert rows[0]['parse_status'] == 'parsed'
    assert rows[0]['prediction']['year'] == 1900
    g = generation(c.case_id, 'She was a painter.')
    _, rows = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'Answer'}}])
    assert rows[0]['parse_status'] == 'unparsed'
    assert rows[0]['utility_uniform_wrong_penalty'] == -1


def test_location_judge_abstention_does_not_confuse_extractor_idk():
    extraction = {'status': 'ok', 'components': []}
    assert parse_component_response('His deathplace is not documented.', extraction).status == 'unparsed'
    assert parse_component_response('His deathplace is not documented.', extraction, abstains=True).status == 'idk'
    assert parse_component_response('He died in Rome.', extraction, abstains=False).status == 'unparsed'


def test_judged_abstention_matches_idk_target_without_becoming_factually_correct():
    c = case('Q1', Relation.BIRTH, TemporalValue(level=TemporalLevel.YEAR, year=1900), target_status='idk')
    g = generation(c.case_id, 'No reliable birth date is available.')
    _, rows = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'No answer'}}])
    assert rows[0]['target_match'] is True
    assert rows[0]['truth_correct'] is False
    assert rows[0]['utility_uniform_wrong_penalty'] == 0


def test_apply_requires_exact_text_and_current_prompt(tmp_path):
    prompt = (judge.ROOT/'prompt.txt').read_text()
    key, _ = judge.make_job(item(), prompt)
    record = {**item(), 'request_key': key, 'judge': {'key': key, 'label': 'No answer', 'reason': 'No value'}}
    write(tmp_path/'predictions.jsonl', [record], rows=True)
    write(tmp_path/'summary.json', {'model': judge.MODEL, 'scope': 'full_answer', 'prompt_sha256': digest(prompt)})
    index = integration.load_predictions(tmp_path/'predictions.jsonl')
    assert index[(item()['question'], item()['answer'])]['label'] == 'No answer'
    record['answer'] = 'Tampered answer'
    write(tmp_path/'predictions.jsonl', [record], rows=True)
    with pytest.raises(ValueError, match='identity'):
        integration.load_predictions(tmp_path/'predictions.jsonl')


def test_opt_in_report_rescores_without_changing_generations(tmp_path):
    from evaluation.vital_persons.data import load_cases
    from evaluation.vital_persons.report import score
    from evaluation.vital_persons.tests.test_workflow import generation as saved_generation
    c = load_cases('lfm', 'test')['temporal'][0]
    text = 'The birth date is not documented.'
    g = saved_generation(c, answer=text)
    inputs = tmp_path/'generations'
    path = inputs/'temporal/base_plain.jsonl'
    write(path, [g], rows=True)
    original = path.read_bytes()
    r = {**item(text), 'question': c.prompt}
    prompt = (judge.ROOT/'prompt.txt').read_text()
    key, _ = judge.make_job(r, prompt)
    write(tmp_path/'cache.jsonl', [{'key': key, 'label': 'No answer', 'reason': 'No date.'}], rows=True)
    judge.score([r], prompt, judge.MODEL, tmp_path/'cache.jsonl', tmp_path/'judged')
    report = score({'temporal': [c]}, ['base_plain'], inputs, tmp_path/'rescored',
                   num_resamples=0, answer_status_predictions=tmp_path/'judged/predictions.jsonl')
    assert path.read_bytes() == original
    out = judge.read_rows(tmp_path/'rescored/outcomes/temporal/base_plain.jsonl')[0]
    assert out['answer'] == text and out['parse_status'] == 'idk'
    assert out['answer_status_judge']['label'] == 'No answer'
    assert out['utility_uniform_wrong_penalty'] == 0
    assert report['answer_status_judge']['applied'] is True


def test_location_judge_abstention_skips_extractor_and_normalizer(tmp_path):
    from evaluation.vital_persons.pipeline import LocationPipeline
    from evaluation.vital_persons.tests.test_pipeline import case as location_case
    class NeverCalled:
        def extract_many(self, *args, **kwargs):
            pytest.fail('Abstention must not call the extractor')
        def process_many(self, *args, **kwargs):
            pytest.fail('Abstention must not call the normalizer')
    c = location_case()
    g = {'case_id': c.case_id, 'answer': 'No historical record of the location exists.',
         'answer_status_judge': {'label': 'No answer'}}
    pipeline = LocationPipeline(tmp_path/'extractions', tmp_path/'normalization.sqlite3',
                                extractor=NeverCalled(), normalizer=NeverCalled())
    _, rows = pipeline.score([c], [g])
    assert rows[0]['parse_status'] == 'idk'
    assert rows[0]['answer'] == g['answer']
    assert rows[0]['utility_uniform_wrong_penalty'] == 0


def test_routed_audit_is_eligible_unique_and_blinded():
    directory = judge.ROOT/'human_validation/routed-40'
    if not (directory/'items.jsonl').exists():
        pytest.skip('Frozen audit not included')
    rows = judge.read_rows(directory/'items.jsonl')
    assert len(rows) == 40
    assert all(judge.needs_judge(r) for r in rows)
    assert len({audit.content_key(r) for r in rows}) == 40
    tasks = json.loads((directory/'evaluation-tasks.json').read_text())
    assert all(set(t['data']) == {'item_id', 'question', 'answer'} for t in tasks)


def test_routing_and_attachment_require_only_failed_parses():
    c = case('Q1', Relation.BIRTH, TemporalValue(level=TemporalLevel.YEAR, year=1900))
    for text, status, label, source in [
        ('IDK', 'idk', 'No answer', 'canonical_idk'),
        ('She was born in 1900.', 'parsed', 'Answer', 'parsed_answer')]:
        g = generation(c.case_id, text)
        outcome = {**g, 'parse_status': status}
        assert not judge.needs_judge(outcome)
        result = integration.attach([g], [c], {}, [outcome])[0]
        assert result['answer_status_judge'] == {'label': label, 'source': source}
    g = generation(c.case_id, 'No historical record exists.')
    for status in ('unparsed', 'multiple', 'normalization_failed', 'extraction_failed'):
        outcome = {**g, 'parse_status': status}
        assert judge.needs_judge(outcome)
        with pytest.raises(ValueError, match='Missing'):
            integration.attach([g], [c], {}, [outcome])
        index = {(c.prompt, g['answer']): {'label': 'Answer'}}
        assert integration.attach([g], [c], index, [outcome])[0]['answer_status_judge']['source'] == 'judge'


def test_no_answer_includes_irrelevant_text_but_api_failures_never_count():
    c = case('Q1', Relation.BIRTH, TemporalValue(level=TemporalLevel.YEAR, year=1900))
    g = generation(c.case_id, 'She was a celebrated painter.')
    _, rows = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'No answer'}}])
    assert rows[0]['parse_status'] == 'idk'
    assert rows[0]['utility_uniform_wrong_penalty'] == 0
    with pytest.raises(ValueError):
        judge.parse_body({'status': 'failed', 'output': []})


def test_sample_reproducible_and_only_routes_failures():
    rows = [{**item(), 'item_id': str(i), 'answer': f'No record {i}',
             'parse_status': 'unparsed', 'modality': 'temporal', 'variant': 'base_plain'}
            for i in range(60)]
    rows += [{**rows[0], 'item_id': 'parsed', 'parse_status': 'parsed'},
             {**rows[0], 'item_id': 'idk', 'answer': 'IDK', 'parse_status': 'idk'}]
    first = audit.sample(rows)
    assert first == audit.sample(list(reversed(rows)))
    assert len(first) == 40
    assert all(judge.needs_judge(r) for r in first)
    with pytest.raises(ValueError, match='requested'):
        audit.sample(rows[:2])


def test_annotation_import_preserves_export_and_labels(tmp_path):
    rows = [item()]
    write(tmp_path/'items.jsonl', rows, rows=True)
    write(tmp_path/'manifest.json', {'items_sha256': digest(rows)})
    source = tmp_path/'export.csv'
    source.write_text('item_id,answer_status\none,Answer\n')
    summary = audit.import_annotations(tmp_path, source)
    assert summary['labels'] == {'Answer': 1}
    assert (tmp_path/'evaluation-annotations.csv').read_bytes() == source.read_bytes()
    assert judge.read_rows(tmp_path/'evaluation-human-labels.jsonl') == [{'item_id': 'one', 'label': 'Answer'}]
    assert audit.import_annotations(tmp_path, source) == summary
    source.write_text('item_id,answer_status\none,No answer\n')
    with pytest.raises(ValueError, match='already imported'):
        audit.import_annotations(tmp_path, source)


def test_premise_rejection_remains_answer_not_abstention():
    c = case('Q1', Relation.DEATH, TemporalValue(level=TemporalLevel.YEAR, year=1900))
    g = generation(c.case_id, 'As of my last update, he is still alive.')
    _, rows = score_temporal_generations([c], [{**g, 'answer_status_judge': {'label': 'Answer'}}])
    assert rows[0]['parse_status'] == 'unparsed'
    assert rows[0]['utility_uniform_wrong_penalty'] == -1


def test_challenge_sampling_excludes_previous_content_and_preserves_quotas():
    rows = [{**item(), 'item_id': str(i), 'answer': f'The location is not documented. Note {i}.',
             'parse_status': 'unparsed', 'modality': 'location', 'variant': 'base_plain'}
            for i in range(30)]
    rows += [{**rows[0], 'item_id': f'b{i}', 'answer': f'Possibly York or Bath. Note {i}.'}
             for i in range(15)]
    excluded = {audit.content_key(rows[0])}
    selected = audit.sample_challenge(rows, excluded=excluded)
    assert len(selected) == 20
    assert Counter(r['cohort'] for r in selected) == {
        'abstention_wording': 15, 'uncertainty_boundary': 5}
    assert not ({audit.content_key(r) for r in selected} & excluded)
    assert selected == audit.sample_challenge(list(reversed(rows)), excluded=excluded)
    assert all(judge.needs_judge(r) for r in selected)
    with pytest.raises(ValueError, match='requested'):
        audit.sample_challenge(rows[:2])


def test_challenge_patterns_are_sampling_hints_not_labels():
    assert audit.challenge_kind(item('I can’t determine the date.')) == 'abstention_wording'
    assert audit.challenge_kind(item('The exact day is unknown, but she died in 1900.')) == 'abstention_wording'
    assert audit.challenge_kind(item('She likely died around 1900.')) == 'uncertainty_boundary'
    assert audit.challenge_kind(item('She died in 1900.')) is None


def test_import_combined_annotations_keeps_cohorts_separate(tmp_path):
    directories = [tmp_path/'original', tmp_path/'challenge']
    for directory, uid in zip(directories, ('one', 'two'), strict=True):
        rows = [{**item(), 'item_id': uid}]
        write(directory/'items.jsonl', rows, rows=True)
        write(directory/'manifest.json', {'items_sha256': digest(rows)})
    export = tmp_path/'export.csv'
    export.write_text('item_id,answer_status\none,Answer\ntwo,No answer\n')
    summary = audit.import_combined_annotations(directories, export)
    assert summary['original']['labels'] == {'Answer': 1}
    assert summary['challenge']['labels'] == {'No answer': 1}
    assert (tmp_path/'annotations.csv').read_bytes() == export.read_bytes()
    assert len(judge.read_rows(directories[1]/'evaluation-human-labels.jsonl')) == 1
    assert audit.read_labels(directories[1]/'evaluation-human-labels.jsonl',
                             judge.read_rows(directories[1]/'items.jsonl')) == {'two': 'No answer'}
    assert audit.import_combined_annotations(directories, export) == summary


def test_direct_requests_deduplicate_cache_and_resume(tmp_path):
    from types import SimpleNamespace
    calls = []
    def create(**body):
        calls.append(body)
        output = {'status': 'completed', 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': json.dumps({'label': 'No answer', 'reason': 'No value.'})}]}]}
        return SimpleNamespace(model_dump=lambda: output)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    cache = tmp_path/'judgments.jsonl'
    rows = [item(), {**item(), 'item_id': 'duplicate'}]
    judge.run_direct(rows, 'p', judge.MODEL, cache, client, 2)
    judge.run_direct(rows, 'p', judge.MODEL, cache, client, 2)
    assert len(calls) == len(judge.read_cache(cache)) == 1


def test_direct_request_failure_never_creates_no_answer_label(tmp_path):
    from types import SimpleNamespace
    def create(**body):
        raise RuntimeError('API failure')
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    cache = tmp_path/'judgments.jsonl'
    with pytest.raises(ValueError, match='requests failed'):
        judge.run_direct([item()], 'p', judge.MODEL, cache, client)
    assert judge.read_cache(cache) == {}
