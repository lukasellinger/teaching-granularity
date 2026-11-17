"""Public annotations reproduce reported metrics without the private run files."""
import json
from pathlib import Path
import pytest
from shared.io import read_rows
from human_annotations.report_generalization import report
from human_annotations.restore_generalization import restore

PUBLIC=Path(__file__).resolve().parents[1]/'generalization'


def test_public_annotations_reproduce_reported_human_results(tmp_path):
    summary=report(PUBLIC,tmp_path/'report')
    assert summary['pairs']==200 and summary['evaluated_pairs']==162
    assert summary['human_agreement']['accuracy']==pytest.approx(.68)
    assert summary['overall']['combined']['accuracy']==pytest.approx(113/162)
    correctness=json.loads((tmp_path/'report/results/correctness-summary.json').read_text())
    assert correctness['pairs']==68 and correctness['answers']==136
    by_variant={v:sum(group[v]['correctness']['Correct'] for group in correctness['by_model'].values()) for v in ('base','finetuned')}
    assert by_variant=={'base':14,'finetuned':31}


def test_export_preserves_labels_without_upstream_question_or_gold_text():
    pairs=read_rows(PUBLIC/'pairs.jsonl')
    labels=read_rows(PUBLIC/'granularity.jsonl')
    correctness=read_rows(PUBLIC/'correctness.jsonl')
    assert len({r['pair_id'] for r in pairs})==200
    assert sum(r['adjudicated'] for r in labels)==64
    for r in pairs:
        assert 'question' not in r and 'reference_answers' not in r
        assert len(r['question_sha256'])==64
        assert r['base']['text'] and r['finetuned']['text']
    for r in correctness:
        assert 'question' not in r and 'reference_answers' not in r
        assert 'annotator' not in r and 'annotation_id' not in r
        assert len(r['reference_answers_sha256'])==64


def test_question_identity_mismatch_stops_local_restoration(tmp_path):
    pairs=read_rows(PUBLIC/'pairs.jsonl')
    cases={name:[] for name in ('granola','simpleqa','truthfulqa')}
    for r in pairs:
        cases[r['dataset']].append({'id':r['source_id'],'question':'A different question'})
    target=tmp_path/'restored'
    with pytest.raises(ValueError,match='Reconstructed question differs'):
        restore(PUBLIC,target,cases)
    assert not target.exists()
