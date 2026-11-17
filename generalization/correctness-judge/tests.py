"""Offline regression tests. No API key, network or model inference required."""
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'generalization.correctness-judge'

from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from . import judge
from .cache import SQLiteJudgmentCache, content_key
from .metrics import agreement, split_pairs, summary


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'cache.sqlite3'
        self.row = {'question': 'Who owns Example?', 'reference_answers': ['Owner', 'A company'],
                    'base_response': 'Owner.', 'finetuned_response': 'A company.'}

    def fake_client(self, output='Correct'):
        calls = []
        def request(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text=output, status='completed')
        return SimpleNamespace(responses=SimpleNamespace(create=request)), calls

    def test_content_identity_includes_every_input(self):
        _, _, identity = judge.configuration()
        fields = {**identity, 'question': 'Who?', 'reference_answers': ['Owner'], 'response': 'Owner.'}
        original = content_key(**fields)
        for field, value in (('model', 'other'), ('prompt_sha256', 'other'), ('cache_namespace', 'other'),
                             ('question', 'Other?'), ('reference_answers', ['Other']), ('response', 'Other.')):
            self.assertNotEqual(original, content_key(**{**fields, field: value}))

    def test_cache_only_never_calls_api_or_creates_database(self):
        client, calls = self.fake_client()
        with self.assertRaisesRegex(ValueError, 'no API calls'):
            judge.judge_rows([self.row], self.path, client=client)
        self.assertEqual(calls, [])
        self.assertFalse(self.path.exists())

    def test_payload_blinding_deduplication_and_cache_reuse(self):
        client, calls = self.fake_client()
        private = {**self.row, 'human_final_base': 'PRIVATE_HUMAN_MARKER', 'model': 'PRIVATE_MODEL_MARKER'}
        rows, counts = judge.judge_rows([private, private], self.path, allow_api=True, client=client)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(set(call) == {'model', 'input'} for call in calls))
        self.assertTrue(all('PRIVATE_' not in call['input'] for call in calls))
        self.assertEqual(counts['unique_judgments'], 2)
        self.assertEqual(rows[0]['judge_base_correct'], 'Correct')
        judge.judge_rows([self.row], self.path, client=client)
        self.assertEqual(len(calls), 2)

    def test_conflicting_labels_and_metadata_are_rejected(self):
        client, _ = self.fake_client()
        judge.judge_rows([self.row], self.path, allow_api=True, client=client)
        _, _, identity = judge.configuration()
        cache = SQLiteJudgmentCache(self.path, identity)
        try:
            record = cache[next(iter(cache))]
            new = {**record, 'response': 'Another answer'}
            new['key'] = content_key(**identity, question=new['question'],
                                     reference_answers=new['reference_answers'], response=new['response'])
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                cache.put_many([new, {**record, 'label': 'Incorrect'}])
            self.assertNotIn(new['key'], cache)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                SQLiteJudgmentCache(self.path, {**identity, 'model': 'different'})
        finally:
            cache.close()

    def test_invalid_response_is_not_cached(self):
        client, calls = self.fake_client('Correct or Incorrect')
        with self.assertRaisesRegex(RuntimeError, 'requests failed'):
            judge.judge_rows([self.row], self.path, allow_api=True, workers=1, client=client)
        self.assertEqual(len(calls), 6)
        self.assertFalse(self.path.exists())

    def test_incomplete_response_is_not_cached(self):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(output_text='Correct', status='incomplete')))
        with self.assertRaisesRegex(RuntimeError, 'requests failed'):
            judge.judge_rows([self.row], self.path, allow_api=True, client=client)
        self.assertFalse(self.path.exists())

    def test_strict_labels_and_no_deterministic_abstention_shortcut(self):
        self.assertEqual(judge.parse_correctness_label('**Correct**'), 'Correct')
        self.assertEqual(judge.parse_correctness_label('Correct. A supported abstraction.'), 'Correct')
        with self.assertRaises(ValueError):
            judge.parse_correctness_label('Correct. But perhaps Incorrect.')
        client, calls = self.fake_client('Abstain')
        rows, _ = judge.judge_rows([{**self.row, 'base_response': 'IDK'}], self.path,
                                  allow_api=True, client=client)
        self.assertEqual(len(calls), 2)
        self.assertEqual(rows[0]['judge_base_correct'], 'Abstain')

    def test_production_uses_exact_scope_and_gold(self):
        case = {'id': '1', 'qid': 'Q1', 'question': 'Who?', 'relation': 'P50', 'reference_answers': ['Author']}
        directory = self.path.parent
        records = [{**case, 'variant': v, 'user_prompt': 'Who?', 'system_prompt': None,
                    'first_sentence': 'First.', 'answer': 'First. Second.'} for v in ('base', 'finetuned')]
        judge.write(directory/'granola/lfm.jsonl', records, rows=True)
        with patch.object(judge, 'load_cases', return_value=[case]):
            rows = judge.generation_pairs(['lfm'], ['first-sentence', 'full-answer'], directory)
            self.assertEqual([r['base_response'] for r in rows], ['First.', 'First. Second.'])
            self.assertTrue(all(r['reference_answers'] == ['Author'] for r in rows))
            judge.write(directory/'granola/lfm.jsonl', records[:1], rows=True)
            with self.assertRaisesRegex(ValueError, 'Both'):
                judge.generation_pairs(['lfm'], ['first-sentence'], directory)

    def test_validation_split_is_paired_balanced_and_order_invariant(self):
        rows = [{'pair_id': f'{m}{i}', 'qid': 'shared' if i == 0 else f'{m}{i}',
                 'model': m, 'question': 'Shared?' if i == 0 else f'{m}{i}?'}
                for m in ('A', 'B') for i in range(4)]
        dev, evaluation = split_pairs(rows, 1, 42)
        self.assertEqual(Counter(r['model'] for r in dev), {'A': 1, 'B': 1})
        self.assertFalse({r['qid'] for r in dev} & {r['qid'] for r in evaluation})
        self.assertEqual((dev, evaluation), split_pairs(list(reversed(rows)), 1, 42))

    def test_agreement_and_correctness_denominators(self):
        self.assertEqual(agreement(['Correct', 'Incorrect'], ['Correct', 'Incorrect'])['cohen_kappa'], 1)
        rows = [{'dataset': 'granola', 'qid': qid, 'reference_answers': ['Reference'],
                 'judge_base_correct': base, 'judge_finetuned_correct': ft}
                for qid, base, ft in (('Q1', 'Incorrect', 'Correct'), ('Q1', 'Correct', 'Correct'),
                                      ('Q2', 'Correct', 'Incorrect'), ('Q3', 'Cannot determine', 'Correct'))]
        result = summary(rows, 20, 42)
        self.assertEqual(result['variants']['base']['correct_share_all_items'], .5)
        self.assertEqual(result['paired_correctness_difference']['determinate_pairs'], 3)
        self.assertEqual(result['paired_correctness_difference']['interval']['num_qid_clusters'], 2)


if __name__ == '__main__':
    unittest.main()
