import json
import math
import pytest
from evaluation.vital_persons.location_informativeness import (derive, DEFAULT_METADATA, PARAMETERS_PATH, estimate_remaining_candidates)
from evaluation.vital_persons.location import location_informativeness
from evaluation.vital_persons.rescore_informativeness import refresh_row
from parsing.location.normalization.values import LocationLevel


def test_parameters_reproduce_current_adapted_gold():
    parameters=json.loads(PARAMETERS_PATH.read_text())
    assert derive(DEFAULT_METADATA)==parameters
    assert parameters['prior_candidates']==1092
    for level in LocationLevel:
        expected=parameters['informativeness'].get(level.value.lower(),1.)
        assert location_informativeness(level)==pytest.approx(expected)


def test_estimator_preserves_two_stage_region_geometric_mean():
    records=[{'city':'A','regions':['R1','R2'],'country':'C','continent':'T'},
             {'city':'B','regions':['R1'],'country':'C','continent':'T'}]
    remaining,prior=estimate_remaining_candidates(records)
    assert prior==2
    assert remaining['country']==pytest.approx(2)
    assert remaining['region']==pytest.approx(2**.75)


def test_utility_refresh_preserves_decisions_and_fixed_penalties():
    row={'gold':{'level':'CITY'},'prediction':{'level':'REGION'},'truth_correct':True,
         'parse_status':'parsed','utility':999,'utility_positive_informativeness':999,
         'utility_uniform_wrong_penalty':999}
    result=refresh_row(row)
    assert result['truth_correct'] is True
    assert result['utility']==pytest.approx(location_informativeness(LocationLevel.REGION))
    wrong=refresh_row(dict(row,truth_correct=False))
    assert wrong['utility_uniform_wrong_penalty']==-1
    abstain=refresh_row(dict(row,prediction=None,truth_correct=False,parse_status='idk'))
    assert abstain['utility_uniform_wrong_penalty']==0
    missing=dict(row,gold=None)
    assert refresh_row(missing)==missing


def test_refresh_synchronizes_standalone_metrics_without_changing_decisions(tmp_path):
    from evaluation.vital_persons.rescore_informativeness import refresh_metric_file
    from shared.io import write
    path = tmp_path / 'metrics/location/finetuned.json'
    write(path, {'evaluation_code_sha256': 'original-code', 'metrics': {
        'truth_accuracy': .5, 'location_utility': 99.,
        'by_relation': {'birthplace': {'truth_accuracy': .25, 'location_utility': 99.}}}})
    fresh = {'truth_accuracy': .9, 'location_utility': .6,
             'by_relation': {'birthplace': {'truth_accuracy': .8, 'location_utility': .7}}}
    assert refresh_metric_file(tmp_path, 'finetuned', fresh)
    assert json.loads(path.read_text()) == {'evaluation_code_sha256': 'original-code', 'metrics': {
        'truth_accuracy': .5, 'location_utility': .6,
        'by_relation': {'birthplace': {'truth_accuracy': .25, 'location_utility': .7}}}}
    assert not refresh_metric_file(tmp_path, 'finetuned', fresh)
    assert not refresh_metric_file(tmp_path, 'base_plain', fresh)
