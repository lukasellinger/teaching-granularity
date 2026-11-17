import pytest
from evaluation.vital_persons.alignment_agreement import agreement, control_row


def row(target, predicted, correct=True, status='parsed'):
    return {'target_level': target, 'prediction': {'level': predicted},
            'parse_status': status, 'truth_correct': correct}


def test_fold_only_correct_finer_and_keep_wrong_abstain_in_denominator():
    rows = [row('YEAR', 'DAY'), row('CENTURY', 'YEAR'),
            row('YEAR', 'CENTURY'), row('YEAR', 'DAY', False),
            row('CENTURY', 'DAY', False, 'idk')]
    result = agreement(rows, 'temporal')
    assert result['correct_n'] == 3
    assert result['counts'] == {'correct_at_target_or_finer': 2, 'correct_coarser': 1, 'wrong': 1, 'abstain': 1}
    assert result['correct_at_target_or_finer_rate'] == .4
    assert result['correct_only_kappa'] == pytest.approx(.4)


def test_perfect_folded_and_undefined_single_level():
    result = agreement([row('YEAR', 'DAY'), row('CENTURY', 'YEAR')], 'temporal')
    assert result['correct_only_kappa'] == 1
    assert result['correct_only_unfolded_kappa'] != 1
    assert agreement([row('YEAR', 'DAY')], 'temporal')['correct_only_kappa'] is None


def test_cap_rechecks_correctness_after_removing_wrong_detail():
    r = row('YEAR', 'DAY', False)
    r['prediction'].update(year=1900, month=5, day=2)
    r['gold'] = {'level': 'DAY', 'year': 1900, 'month': 6, 'day': 3}
    projected = control_row(r, 'temporal', 'cap-YEAR')
    assert projected['truth_correct']
    assert projected['prediction_level_for_target_metrics'] == 'YEAR'
