import pytest
from evaluation.vital_persons.consensus_utility import infer, stats, score_target, values_for
from evaluation.vital_persons.table import temporal_value


def test_missing_evidence_is_not_renormalized_or_imputed_as_idk():
    value = temporal_value({'level':'YEAR','year':1900})
    target,status = infer([value]*16+[None]*4,'temporal',.8)
    assert status == 'known' and target.year == 1900
    target,status = infer([value]*16+[None]*4,'temporal',1.)
    assert status == 'unavailable' and target is None
    assert score_target(target,status,value,'temporal') is None


def test_explicit_idk_and_missing_utility_have_distinct_denominators():
    values = [temporal_value({'level':'CENTURY','century':i+1}) for i in range(20)]
    target,status = infer(values,'temporal',.8)
    assert status == 'idk'
    assert score_target(target,status,values[0],'temporal') == 0
    result = stats([{'status':'idk','utility':0.}, {'status':'unavailable','utility':None}])
    assert result['n'] == 2 and result['utility_n'] == 1
    assert result['utility'] == 0 and result['utility_lower_bound'] == -.5


def test_saved_report_has_matching_cohorts_and_reproduces_mean_utilities(require_local_files):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    report_path = root/'results-consensus-utility/test/summary.json'
    require_local_files([report_path], reason='Saved consensus-utility reports are local-only')
    report = json.loads(report_path.read_text())
    require_local_files(
        [root/'results-corrected'/model/'test/outcomes'/modality/(method+'.jsonl')
         for modality, block in report['domains'].items() for model in block['models']
         for method in ('base_plain', 'finetuned')],
        reason='Saved per-answer evaluation outcomes are local-only')
    for modality,block in report['domains'].items():
        keys = {tuple(k) for k in block['comparison80_keys']}
        assert len(keys) == block['comparison80_n']
        for model,values in block['models'].items():
            assert values['comparison80']['consensus80']['utility_n'] == len(keys)
            for method in ('base_plain','finetuned'):
                path = root/'results-corrected'/model/'test'/'outcomes'/modality/(method+'.jsonl')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                selected = [r for r in rows if (r['qid'],r['relation']) in keys]
                assert len(selected) == len(keys)
                expected = sum(r['utility_uniform_wrong_penalty'] for r in selected)/len(keys)
                assert values['comparison80'][method] == pytest.approx(expected)
            assert all(r['utility_n'] == values['model_fixed_sweep_n'] for r in values['model_fixed_cohort_sweep'].values())
