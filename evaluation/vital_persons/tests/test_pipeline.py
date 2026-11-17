from dataclasses import replace
import json

import pytest

from evaluation.vital_persons.cases import (
    LocationEvaluationCase, LocationLevel, LocationValue, Relation, Split,
    TemporalLevel, parse_gold_temporal, parse_gold_location,
)
from evaluation.vital_persons import location
from evaluation.vital_persons.pipeline import LocationPipeline, parse_component_response
from parsing.location.normalization.schemas import ComponentResolution, NormalizedLocation


def test_saved_output_scoring_does_not_replace_prediction_with_gold(tmp_path):
    from evaluation.vital_persons.pipeline import SavedLocationPipeline
    from shared.io import write
    c = case()
    generation = {'case_id': c.case_id, 'answer': 'Pavia, Italy', 'variant': 'finetuned'}
    value = NormalizedLocation(level='CITY', explicit_levels=['CITY', 'COUNTRY'], city='Pavia', country='Italy')
    resolution = ComponentResolution(status='normalized', components=['Pavia', 'Italy'],
                                     value=value, consensus_value=value)
    write(tmp_path/'finetuned.jsonl', [{'case_id': c.case_id, 'question': c.prompt,
          'answer': generation['answer'], 'extraction': {'status': 'ok', 'components': ['Pavia', 'Italy']},
          'normalization': resolution.model_dump(mode='json')}], rows=True)
    pipeline = SavedLocationPipeline(tmp_path)
    _, rows = pipeline.score([c], [generation])
    assert rows[0]['parse_status'] == 'parsed'
    assert rows[0]['prediction']['city'] == 'Pavia'
    assert rows[0]['truth_correct'] is False  # gold is Berlin; it stays wrong
    _, abstained = pipeline.score([c], [{**generation, 'answer_status_judge': {'label': 'No answer'}}])
    assert abstained[0]['parse_status'] == 'idk'


def test_output_correction_keeps_original_answer_status_judge_routing(tmp_path):
    from evaluation.vital_persons.pipeline import SavedLocationPipeline
    from evaluation.vital_persons.answer_status.integration import attach
    from shared.io import write
    c = case()
    generation = {'case_id': c.case_id, 'answer': 'I cannot decide; Berlin is one possibility.', 'variant': 'finetuned'}
    corrected = normalized()
    original = normalized('partial')
    write(tmp_path/'finetuned.jsonl', [{'case_id': c.case_id, 'question': c.prompt,
          'answer': generation['answer'], 'extraction': {'status': 'ok', 'components': ['Berlin']},
          'normalization': corrected.model_dump(mode='json'),
          'normalization_before_correction': original.model_dump(mode='json')}], rows=True)
    pipeline = SavedLocationPipeline(tmp_path)
    _, rows = pipeline.score([c], [generation])
    assert rows[0]['parse_status'] == 'parsed'
    assert rows[0]['answer_status_routing_parse_status'] == 'normalization_failed'
    judged = attach([generation], [c], {(c.prompt, generation['answer']): {'label': 'No answer'}}, rows)
    assert judged[0]['answer_status_judge']['source'] == 'judge'
    _, scored = pipeline.score([c], judged)
    assert scored[0]['parse_status'] == 'idk'


def case(qid='Q1', **fields):
    return LocationEvaluationCase(case_id=f'MODEL:{qid}:birthplace', source_model='MODEL',
        split=Split.TEST, qid=qid, name='Example', relation=Relation.BIRTHPLACE,
        prompt='Where was Example born?', gold=LocationValue(LocationLevel.CITY, city='Berlin', country='Germany'),
        target_status='known', target_level=LocationLevel.CITY, location_taxonomy='four-level', **fields)


def normalized(status='normalized'):
    if status == 'failed':
        return ComponentResolution(status=status, components=['Berlin'], error='Test failure')
    value = NormalizedLocation(level=LocationLevel.CITY, explicit_levels=[LocationLevel.CITY],
                               city='Berlin', country='Germany', continent='Europe')
    return ComponentResolution(status=status, components=['Berlin'], value=value, consensus_value=value)


def test_extractor_idk_does_not_become_model_abstention():
    result = {'status': 'ok', 'components': [], 'location': 'IDK'}
    assert parse_component_response('Born in an unknown place.', result).status == 'unparsed'
    assert parse_component_response("I don't know.", result).status == 'idk'


@pytest.mark.parametrize('status', ['partial', 'failed'])
def test_partial_or_failed_normalization_not_scored_as_country_fallback(status):
    result = parse_component_response('Born in Berlin.', {'status': 'ok', 'components': ['Berlin']}, normalized(status))
    assert result.status == 'normalization_failed'
    assert result.value is None


def test_cache_full_answer_and_deduplication(tmp_path):
    class Extractor:
        def __init__(self):
            self.calls = []
        def extract_many(self, rows):
            self.calls.extend(rows)
            return [{'status': 'ok', 'location': 'Berlin', 'components': ['Berlin']} for row in rows]
    class Normalizer:
        def __init__(self):
            self.calls = []
        def process_many(self, rows, workers):
            self.calls.extend(rows)
            return [normalized().model_dump(mode='json') for row in rows]
        def close(self):
            pass
    extractor, normalizer = Extractor(), Normalizer()
    pipeline = LocationPipeline(tmp_path/'extract', tmp_path/'normalization.sqlite3', extractor=extractor, normalizer=normalizer)
    cases = [case(), case('Q2')]
    text = 'Example was born in **Berlin**. Later unrelated narrative.'
    generations = [{'case_id': c.case_id, 'answer': text} for c in cases]
    first = pipeline.score(cases, generations)
    second = pipeline.score(cases, generations)
    assert len(extractor.calls) == 1
    assert len(normalizer.calls) == 2  # resolution cache is the normalizer's responsibility
    assert all(row['answer'] == text for row in extractor.calls+normalizer.calls)
    assert first == second
    assert first[0]['truth_accuracy'] == 1
    assert all(row['location_audit']['normalization']['status'] == 'normalized' for row in first[1])
    bad = generations+[generations[0]]
    with pytest.raises(ValueError, match='unique'):
        pipeline.score(cases, bad)


@pytest.mark.parametrize('status', ['unparsed', 'extraction_failed', 'normalization_failed'])
def test_non_evaluable_non_idk_receives_uniform_error_penalty(status):
    parsed = location.ParsedLocationResponse(status, None, (), ())
    row = location.score_location_prediction(case(), {'answer': 'Unresolved answer.'}, parsed)
    assert row['utility_uniform_wrong_penalty'] == -1
    assert row['utility_positive_informativeness'] == 0
    assert not row['truth_correct']


def test_primary_and_strict_context_correctness_remain_separate():
    gold = LocationValue(LocationLevel.CITY, city='Berlin', country='Germany', needs_disambiguation=True)
    prediction = LocationValue(LocationLevel.CITY, city='Berlin', country='Germany')
    assert location.correct_at_prediction_level(prediction, gold)
    assert not location.correct_at_prediction_level_strict_context(prediction, gold, [LocationLevel.CITY])
    assert not location.correct_at_prediction_level(replace(prediction, country='United States'), gold)


def test_finer_correct_location_is_not_an_exact_target_match():
    c = replace(case(), target_level=LocationLevel.COUNTRY)
    parsed = parse_component_response('Berlin.', {'status': 'ok', 'components': ['Berlin']}, normalized())
    row = location.score_location_prediction(c, {'answer': 'Berlin.'}, parsed)
    assert row['truth_correct'] and not row['target_match']
    assert row['utility_uniform_wrong_penalty'] == 1


def test_subcity_projects_to_containing_city_without_sibling_aliases():
    c = replace(case(), gold=LocationValue(LocationLevel.SUBCITY, subcity=('Manhattan',), city='New York City'))
    value = LocationValue(LocationLevel.SUBCITY, subcity=('Manhattan',), city='New York City')
    parsed = location.ParsedLocationResponse('parsed', value, (LocationLevel.SUBCITY,), (value,))
    row = location.score_location_prediction(c, {'answer': 'Manhattan.'}, parsed)
    assert row['prediction']['city'] == 'New York City'
    assert row['prediction_category'] == 'CITY'
    assert row['truth_correct']


def test_gold_precision_and_subcity_hierarchy():
    date = parse_gold_temporal('-0450-00-00T00:00:00Z', 9)
    assert date.year == -450 and date.level == TemporalLevel.YEAR
    gold = parse_gold_location({'birth_subcity': ['Manhattan'], 'birth_city': 'New York City',
                               'birth_region': ['New York'], 'birth_country': 'United States'}, 'birth')
    assert gold.level == LocationLevel.SUBCITY and gold.city == 'New York City'


def test_no_extractor_or_normalizer_needed_for_explicit_abstention(tmp_path):
    pipeline = LocationPipeline(tmp_path/'extract', tmp_path/'unused.sqlite3')
    metrics, outcomes = pipeline.score([case()], [{'case_id': case().case_id, 'answer': 'IDK'}])
    assert outcomes[0]['utility_uniform_wrong_penalty'] == 0
    assert not list(tmp_path.iterdir())


def test_location_bootstrap_intervals_are_for_differences():
    cases = [case('Q1'), case('Q2')]
    base, ft = [], []
    for c in cases:
        base.append(location.score_location_prediction(c, {'answer': 'IDK'}, location.ParsedLocationResponse('idk', None, (), ())))
        ft.append(location.score_location_prediction(c, {'answer': 'Berlin'}, parse_component_response('Berlin', {'status': 'ok', 'components': ['Berlin']}, normalized())))
    result = location.paired_qid_bootstrap(base, ft, num_resamples=10, confidence_level=.95, seed=42)
    assert result == location.paired_qid_bootstrap(base, ft, num_resamples=10, confidence_level=.95, seed=42)
    assert result['num_qid_clusters'] == 2
    metric = result['metrics']['truth_accuracy']
    assert metric['difference']['estimate'] == 1
    assert metric['difference']['ci_lower'] == 1
