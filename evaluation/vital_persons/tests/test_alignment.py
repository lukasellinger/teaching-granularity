import json

import pytest

from evaluation.vital_persons.plot_alignment import collect, matrix


def row(case_id, target='YEAR', predicted='YEAR', correct=True, parse='parsed', status='known'):
    return {'case_id': case_id, 'qid': 'Q'+str(case_id), 'target_status': status,
            'target_level': target, 'gold': {'year': 1900}, 'parse_status': parse,
            'truth_correct': correct, 'prediction': {'level': predicted},
            'prediction_level_for_target_metrics': predicted}


def test_correct_finer_folds_but_wrong_finer_does_not():
    rows = [row(1, predicted='DAY'), row(2), row(3, predicted='CENTURY'),
            row(4, predicted='DAY', correct=False), row(5, correct=False),
            row(6, correct=False, parse='idk'), row(7, correct=False, parse='unparsed')]
    b = matrix(rows, 'temporal')
    assert b['row_totals']['YEAR'] == 7
    assert b['counts']['YEAR'] == {'DAY': 0, 'MONTH': 0, 'YEAR': 2, 'DECADE': 0,
                                  'CENTURY': 1, 'abstain': 1, 'wrong': 2, 'pipeline_failure': 1}
    assert b['finer_correct_folded_into_diagonal']['YEAR'] == 1
    assert b['wrong_by_predicted_level']['counts']['YEAR']['DAY'] == 1
    assert b['wrong_by_predicted_level']['row_totals']['YEAR'] == 2
    assert b['wrong_by_predicted_level']['row_normalized']['YEAR']['DAY'] == 1/2
    assert sum(b['row_normalized']['YEAR'].values()) == pytest.approx(1)
    assert sum(b['wrong_by_predicted_level']['row_normalized']['YEAR'].values()) == 1
    assert all(v is None for v in b['row_normalized']['MONTH'].values())


def test_coverage_and_idk_targets_are_separate():
    rows = [row(1), row(2, status='idk'), row(3, status='idk', parse='idk'),
            row(4, status='unavailable'), row(5)]
    rows[-1]['gold'] = None
    b = matrix(rows, 'temporal')
    assert b['coverage'] == {'total': 5, 'included': 1, 'missing_gold': 1,
                             'idk_target': 2, 'unavailable_target': 1}
    assert b['idk_target_outcomes'] == {'correct': 1, 'wrong': 0, 'abstain': 1, 'pipeline_failure': 0}


def test_location_uses_projected_level_not_raw_place():
    r = row(1, target='COUNTRY', predicted='CITY')
    r['prediction']['level'] = 'PLACE'
    b = matrix([r], 'location')
    assert b['counts']['COUNTRY']['COUNTRY'] == 1
    assert b['finer_correct_folded_into_diagonal']['COUNTRY'] == 1
    assert b['wrong_by_predicted_level']['row_totals']['COUNTRY'] == 0
    assert all(v is None for v in b['wrong_by_predicted_level']['row_normalized']['COUNTRY'].values())


def test_rejects_duplicates_invalid_levels_and_missing_truth():
    with pytest.raises(ValueError, match='Duplicate'):
        matrix([row(1), row(1)], 'temporal')
    with pytest.raises(ValueError, match='Unexpected temporal prediction'):
        matrix([row(1, predicted='PLACE')], 'temporal')
    with pytest.raises(ValueError, match='Boolean'):
        matrix([row(1, correct=None)], 'temporal')


def test_methods_must_share_cases_targets_and_gold(tmp_path):
    variants = ['base_plain', 'finetuned']
    for modality in ('temporal', 'location'):
        for variant in variants:
            path = tmp_path/'lfm/test/outcomes'/modality/f'{variant}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            r = row(1, target='YEAR' if modality == 'temporal' else 'CITY',
                    predicted='YEAR' if modality == 'temporal' else 'CITY')
            path.write_text(json.dumps(r)+'\n')
    report = collect(tmp_path, ['lfm'], variants, 'test')
    assert len(report['sources']) == 4
    path = tmp_path/'lfm/test/outcomes/temporal/finetuned.jsonl'
    r = json.loads(path.read_text())
    r['gold'] = {'year': 1901}
    path.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError, match='different case grids, targets or gold'):
        collect(tmp_path, ['lfm'], variants, 'test')



def test_display_excludes_failures_but_preserves_raw_counts():
    b=matrix([row(1),row(2,parse='unparsed',correct=False),row(3,correct=False)],'temporal')
    assert b['row_totals']['YEAR']==3
    assert b['display']['row_totals']['YEAR']==2
    assert 'pipeline_failure' not in b['display']['columns']
    assert b['display']['excluded_pipeline_failures']['YEAR']==1
    assert b['display']['row_normalized']['YEAR']['YEAR']==.5
    assert b['display']['counts']['YEAR']['YEAR']==1
    b=matrix([row(1,parse='unparsed',correct=False)],'temporal')
    assert b['display']['row_totals']['YEAR']==0
    assert all(v is None for v in b['display']['row_normalized']['YEAR'].values())



def test_unfolded_keeps_finer_correct_at_actual_level():
    rows=[row(1,predicted='DAY'),row(2),row(3,predicted='CENTURY'),row(4,predicted='DAY',correct=False)]
    folded=matrix(rows,'temporal')
    actual=matrix(rows,'temporal',fold_finer=False)
    assert actual['display']['counts']['YEAR']['DAY']==1
    assert actual['display']['counts']['YEAR']['YEAR']==1
    assert folded['display']['counts']['YEAR']['YEAR']==2
    assert actual['display']['counts']['YEAR']['CENTURY']==1
    assert actual['display']['row_totals']==folded['display']['row_totals']
    assert actual['wrong_by_predicted_level']==folded['wrong_by_predicted_level']



def test_base_ft_comparison_matches_cases_when_only_one_method_fails(tmp_path):
    from evaluation.vital_persons.plot_alignment import collect_comparison
    for modality in ('temporal','location'):
        level='YEAR' if modality=='temporal' else 'CITY'
        for variant in ('base_plain','finetuned'):
            path=tmp_path/'lfm/test/outcomes'/modality/f'{variant}.jsonl'
            path.parent.mkdir(parents=True,exist_ok=True)
            rows=[row(1,target=level,predicted=level),row(2,target=level,predicted=level)]
            if variant=='base_plain':rows[1]['parse_status']='unparsed'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=collect_comparison(tmp_path,['lfm'],'test')
    for modality in ('temporal','location'):
        assert report['paired_cohorts']['lfm'][modality]['paired_cases']==1
        assert report['paired_cohorts']['lfm'][modality]['excluded_pipeline_failure_either_method']==1
        assert report['models']['lfm'][modality]['base_plain']['display']['row_totals']==report['models']['lfm'][modality]['finetuned']['display']['row_totals']
