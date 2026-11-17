from copy import deepcopy

import pytest

from shared.io import write, read_rows
from evaluation.vital_persons.correct_location import apply, correct_row, snapshot, input_key
from parsing.location.normalization.schemas import ComponentResolution, NormalizedLocation


def example():
    value = NormalizedLocation(level='CITY', explicit_levels=['CITY'],
                               city='Pavia', country='Philippines', continent='Asia')
    row = {'case_id': 'example', 'question': 'Where did Example die?',
           'answer': 'He died in Pavia, Italy.',
           'extraction': {'status': 'ok', 'components': ['Pavia', 'Italy']},
           'normalization': ComponentResolution(status='partial', components=['Pavia', 'Italy'],
                                                value=value).model_dump(mode='json')}
    after = NormalizedLocation(level='CITY', explicit_levels=['CITY', 'COUNTRY'],
                               city='Pavia', country='Italy', continent='Europe')
    correction = {'components': row['extraction']['components'], 'answer': row['answer'],
                  'before': snapshot(row['normalization']),
                  'after': {'status': 'normalized', 'value': after.model_dump(mode='json')},
                  'reviewer': 'Codex', 'note': 'Explicit Italy disambiguates the city.'}
    return row, correction


def test_correction_preserves_answer_extraction_original_and_input_row():
    row, correction = example()
    before = deepcopy(row)
    key = input_key(correction['components'], correction['answer'])
    changed, matched = correct_row(row, {key: correction}, 'manifest-hash')
    assert matched == key
    assert row == before
    assert changed['answer'] == row['answer']
    assert changed['extraction'] == row['extraction']
    assert changed['normalization_before_correction'] == row['normalization']
    assert changed['normalization']['value']['city'] == 'Pavia'
    assert changed['normalization']['value']['country'] == 'Italy'
    assert changed['normalization']['traces'] == []  # not fabricated resolver evidence


def test_different_full_answer_is_not_implicitly_corrected():
    row, correction = example()
    row['answer'] += ' Another sentence.'
    key = input_key(correction['components'], correction['answer'])
    assert correct_row(row, {key: correction}, 'hash') == (row, None)


def test_stale_normalization_is_rejected():
    row, correction = example()
    key = input_key(correction['components'], correction['answer'])
    row['normalization']['value']['country'] = 'Changed'
    with pytest.raises(ValueError, match='no longer matches'):
        correct_row(row, {key: correction}, 'hash')


def test_apply_does_not_change_source_and_refuses_existing_output(tmp_path):
    row, correction = example()
    source = tmp_path/'original/lfm/test/location_extractions/base_plain.jsonl'
    write(source, [row], rows=True)
    manifest = tmp_path/'corrections.jsonl'
    write(manifest, [correction], rows=True)
    before = source.read_bytes()
    output = tmp_path/'corrected'
    apply(tmp_path/'original', manifest, output)
    assert source.read_bytes() == before
    assert read_rows(output/'lfm/test/location_extractions/base_plain.jsonl')[0]['normalization']['value']['country'] == 'Italy'
    with pytest.raises(FileExistsError):
        apply(tmp_path/'original', manifest, output)


def test_preflight_failure_writes_no_partial_output(tmp_path):
    row, correction = example()
    write(tmp_path/'original/lfm/test/location_extractions/base_plain.jsonl', [row], rows=True)
    correction['before']['status'] = 'normalized'
    manifest = tmp_path/'corrections.jsonl'
    write(manifest, [correction], rows=True)
    with pytest.raises(ValueError, match='no longer matches'):
        apply(tmp_path/'original', manifest, tmp_path/'corrected')
    assert not (tmp_path/'corrected').exists()
