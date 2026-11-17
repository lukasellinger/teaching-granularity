import hashlib
import json
from pathlib import Path

import pytest

from shared.io import digest, file_hash, iter_rows, read_rows, write
from shared.constants import MODEL_NAMES, QUESTIONS
from shared.geography import project_four_level_location, project_location
from parsing.location.normalization.values import LocationLevel, LocationValue


def test_saved_digest_formats_are_not_accidentally_unified():
    from training.configuration import identity
    from importlib import import_module
    study = import_module('generalization.human-study.human_study')
    value = {'name': 'José', 'parts': ['Washington, D.C.', 'France']}
    assert digest(value) == hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert identity(value) == hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert study.digest(value) == hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    assert len({digest(value), identity(value), study.digest(value)}) == 3


def test_jsonl_bytes_and_streaming_are_unchanged(tmp_path):
    rows = [{'z': 'José', 'a': 1}, {'value': 'Washington, D.C.'}]
    path = tmp_path/'nested/rows.jsonl'
    write(path, rows, rows=True)
    expected = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows)
    assert path.read_bytes() == expected.encode()
    assert read_rows(path) == list(iter_rows(path)) == rows
    assert file_hash(path) == hashlib.sha256(expected.encode()).hexdigest()
    assert not path.with_suffix('.jsonl.tmp').exists()
    write(path, {'name': 'José'})
    assert path.read_text() == json.dumps({'name': 'José'}, indent=2)+'\n'


def test_sentence_helpers_are_one_implementation_but_policies_stay_separate():
    from shared import text
    from parsing.temporal import sentences as temporal
    from parsing.location.rewriter import sentences as rewriter
    assert temporal._is_nonterminal_period is rewriter._is_nonterminal_period is text._is_nonterminal_period
    assert temporal._SENTENCE_END is rewriter._SENTENCE_END
    assert rewriter.complete_sentence_prefix('He died in Paris. Unfinished continuation') == 'He died in Paris.'
    source = 'He met Dr. Smith. He died in Paris.'
    assert temporal.sentence_bounds_containing(source, source.index('Smith')) == (0, 17)


def test_chat_rendering_has_one_shared_implementation():
    pytest.importorskip('torch')
    from shared.chat import render_chat
    from shared.generation import render_chat as generation_chat
    from training.encoding import render_chat as training_chat
    assert render_chat is generation_chat is training_chat


def test_canonical_questions_and_model_names_have_not_changed():
    from evaluation.vital_persons.cases import QUESTIONS as evaluation_questions, SOURCE_MODELS
    from data_preparation.vital_persons.generate import PROMPTS
    from data_preparation.granola.recognition import MODEL_NAMES as recognition_models
    assert evaluation_questions is PROMPTS is QUESTIONS
    assert recognition_models is SOURCE_MODELS is MODEL_NAMES
    assert list(MODEL_NAMES) == ['lfm', 'qwen', 'gemma', 'ministral']


def test_preparation_and_evaluation_share_four_level_projection():
    from data_preparation.vital_persons.preparation.targets import project_location as prepare
    value = LocationValue(LocationLevel.PLACE, place=('Hospital',), subcity=('Manhattan',),
                          city='New York City', country='United States')
    assert prepare is project_four_level_location
    assert prepare(value) == project_location(value, 'four-level')
    assert prepare(value).level == LocationLevel.CITY
    assert project_location(value) is value


def test_package_imports_do_not_change_the_module_search_path():
    import importlib
    import sys
    before = list(sys.path)
    for name in ('generalization.generate', 'generalization.preference', 'generalization.coarsening',
                 'generalization.human-study.human_study', 'generalization.correctness-judge.judge',
                 'generalization.granularity-judge.judge', 'generalization.granularity-judge.batch'):
        importlib.import_module(name)
    assert sys.path == before
