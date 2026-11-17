from importlib import import_module
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'generalization.granularity-judge'

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from . import judge
study = import_module('generalization.human-study.human_study')


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.rows = study.load_study(judge.STUDY)
        self.prompt, self.labels = judge.instructions(judge.STUDY)

    def test_blinding_and_cache_identity(self):
        row = self.rows[0]
        key, request = judge.make_job(row, self.prompt, self.labels, 'gpt-6-luna')
        payload = json.loads(request['input'])
        self.assertEqual(set(payload), {'question', 'answer_a', 'answer_b'})
        self.assertNotIn('granuscore', request['input'])
        self.assertNotIn('human_label', request['input'])
        self.assertNotEqual(key, judge.make_job(row, self.prompt+'x', self.labels, 'gpt-6-luna')[0])
        self.assertNotEqual(key, judge.make_job(row, self.prompt, self.labels, 'other')[0])

    def test_display_order(self):
        for row in self.rows:
            _, request = judge.make_job(row, self.prompt, self.labels, 'gpt-6-luna')
            assignment = row['assignments']['annotator_1']
            self.assertEqual(json.loads(request['input'])['answer_a'], row[assignment['a_variant']]['text'])
            self.assertEqual(study.choice('A more fine-grained', assignment), assignment['a_variant']+'_more_specific')

    def test_api_is_opt_in_and_cache_resumes(self):
        calls = []
        def create(**request):
            calls.append(request)
            return SimpleNamespace(status='completed', id='test', model='gpt-6-luna', usage=None,
                                   output_text=json.dumps({'label': 'Same granularity', 'reason': 'Test'}))
        client = SimpleNamespace(responses=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)/'cache.jsonl'
            with self.assertRaises(ValueError):
                judge.judge(self.rows[:2], self.prompt, self.labels, 'gpt-6-luna', cache)
            first = judge.judge(self.rows[:2], self.prompt, self.labels, 'gpt-6-luna', cache, allow_api=True, client=client)
            second = judge.judge(self.rows[:2], self.prompt, self.labels, 'gpt-6-luna', cache)
            self.assertEqual(len(calls), 2)
            self.assertEqual(first, second)

    def test_comparison_same_pairs_and_nonorderable_penalized(self):
        rows, human = study.adjudicated_study(judge.STUDY)
        judged = [{**row, 'judge': {'label': 'Not comparable', 'reason': 'test'}} for row in rows]
        metrics = study.read_jsonl(judge.STUDY/'results/predictions.jsonl')
        records, summary = judge.evaluate(judged, human, metrics)
        self.assertEqual(len(records), 200)
        overall = summary['overall']
        self.assertEqual(overall['all_five_labels']['n'], 200)
        for method, result in overall['orderable_human_pairs'].items():
            self.assertEqual(result['n'], 162)
        self.assertEqual(overall['orderable_human_pairs']['llm_judge']['accuracy'], 0)
        self.assertAlmostEqual(overall['orderable_human_pairs']['combined']['accuracy'], 113/162)

    def test_stale_metrics_rejected(self):
        rows, human = study.adjudicated_study(judge.STUDY)
        judged = [{**row, 'judge': {'label': 'Same granularity', 'reason': 'test'}} for row in rows]
        with self.assertRaises(ValueError):
            judge.evaluate(judged, human, [])

    def test_incomplete_response_rejected(self):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(status='incomplete')))
        with self.assertRaises(ValueError):
            judge.request_judgment(client, {}, self.labels)

    def test_production_keeps_all_outcomes_and_source_texts(self):
        rows = []
        for i, label in enumerate(self.labels):
            row = {**self.rows[0], 'source_id': str(i), 'qid': str(i), 'relation': 'P26',
                   'judge': {'label': label, 'reason': 'test', 'key': str(i)}}
            rows.append(row)
        records, summary = judge.production_results(rows)
        self.assertEqual(summary['overall']['pairs'], 5)
        self.assertEqual(len(summary['overall']['directions']), 5)
        self.assertEqual(records[0]['base_response'], rows[0]['base']['text'])

    def test_coarsening_excludes_nonorderable_not_high_score(self):
        from generalization import coarsening
        features, predictions = [], []
        for i, label in enumerate(['base_more_specific', 'equal', 'finetuned_more_specific', 'not_comparable', 'abstention_present']):
            row = {'model': 'lfm', 'dataset': 'granola', 'source_id': str(i), 'question': 'Q',
                   'base_response': 'A', 'finetuned_response': 'B', 'base_answer_tokens': 1,
                   'base_logprob': -1., 'base_granuscore': 100.}
            features.append(row)
            predictions.append({**row, 'direction': label})
        eligible, excluded = coarsening.label_judge(features, predictions)
        self.assertEqual([r['coarser'] for r in eligible], [1, 0, 0])
        self.assertEqual({r['exclusion'] for r in excluded}, {'not_comparable', 'abstention_present'})
        with self.assertRaises(ValueError):
            coarsening.label_judge(features, predictions[:-1])

    def test_matched_coarsening_shares_cases_and_cutoffs(self):
        from generalization import coarsening
        features, predictions = [], []
        for i, label in enumerate(['base_more_specific', 'equal', 'finetuned_more_specific', 'not_comparable']):
            row = {'model': 'lfm', 'dataset': 'granola', 'source_id': str(i), 'relation': 'P26',
                   'question': 'Q', 'base_response': 'A', 'finetuned_response': 'B',
                   'base_answer_tokens': 1, 'base_logprob': -float(i+1),
                   'base_granuscore': 99., 'finetuned_granuscore': 100.}
            features.append(row)
            predictions.append({**row, 'direction': label})
        parameters = {'alpha': 1., 'band': 0., 'granuscore_scale': 1., 'length_scale': 1.}
        from unittest.mock import patch
        with patch.object(coarsening, 'attach_quartiles', wraps=coarsening.attach_quartiles) as quartiles:
            combined, judged, excluded = coarsening.matched_outcomes(features, predictions, parameters)
            self.assertEqual(quartiles.call_count, 1)
        self.assertEqual(len(combined), 3)
        self.assertEqual(len(excluded), 1)
        for a, b in zip(combined, judged, strict=True):
            for key in ('source_id', 'q25', 'q75', 'low_confidence', 'base_logprob'):
                self.assertEqual(a[key], b[key])
        self.assertEqual([r['coarser'] for r in combined], [1, 1, 1])
        self.assertEqual([r['coarser'] for r in judged], [1, 0, 0])
        features[0]['base_granuscore'] = None
        with self.assertRaises(ValueError):
            coarsening.matched_outcomes(features, predictions, parameters)

    def test_primary_relation_only_and_fully_adjusted_sensitivity(self):
        from generalization import coarsening
        rng = coarsening.np.random.default_rng(42)
        rows = [{'model': 'lfm', 'dataset': 'granola', 'relation': 'P26' if i%3 else 'P50',
                 'qid': str(i//2), 'low_confidence': i%2, 'base_granuscore': float(rng.uniform(1, 100)),
                 'base_words': int(rng.integers(2, 40)),
                 'coarser': int(rng.random() < .25+.3*(i%2))} for i in range(400)]
        primary = coarsening.logistic(rows)
        sensitivity = coarsening.logistic(rows, adjustment='linear')
        self.assertEqual(primary['status'], 'estimated')
        self.assertEqual(sensitivity['status'], 'estimated')
        self.assertIn('C(dataset_relation)', primary['formula'])
        self.assertNotIn('base_granuscore', primary['formula'])
        self.assertNotIn('base_words', primary['formula'])
        self.assertIn('base_granuscore', sensitivity['formula'])
        self.assertIn('base_words', sensitivity['formula'])
        self.assertEqual(primary['cases'], sensitivity['cases'])
        self.assertEqual(primary['qid_clusters'], sensitivity['qid_clusters'])

    def test_batch_submits_only_missing_and_does_not_duplicate(self):
        from . import batch
        client = SimpleNamespace(files=SimpleNamespace(create=Mock(return_value=SimpleNamespace(id='file-test'))),
            batches=SimpleNamespace(create=Mock(return_value=SimpleNamespace(id='batch-test', status='validating'))))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'batch'
            cache = Path(tmp)/'cache.jsonl'
            key, _ = judge.make_job(self.rows[0], self.prompt, self.labels, 'gpt-6-luna')
            study.save(cache, [{'key': key}], jsonl=True)
            batch.submit(self.rows[:2], self.prompt, self.labels, 'gpt-6-luna', cache, directory, client)
            batch.submit(self.rows[:2], self.prompt, self.labels, 'gpt-6-luna', cache, directory, client)
            self.assertEqual(client.batches.create.call_count, 1)
            requests = study.read_jsonl(directory/'input.jsonl')
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]['url'], '/v1/responses')
            self.assertNotEqual(requests[0]['custom_id'], key)

    def test_batch_collect_maps_out_of_order_and_is_idempotent(self):
        from . import batch
        from openai.types.responses import Response
        with tempfile.TemporaryDirectory() as tmp:
            directory, cache = Path(tmp), Path(tmp)/'cache.jsonl'
            jobs = dict(judge.make_job(row, self.prompt, self.labels, 'gpt-6-luna') for row in self.rows[:2])
            study.save(directory/'input.jsonl', [{'custom_id': k} for k in jobs], jsonl=True)
            import hashlib
            study.save(directory/'state.json', {'batch_id': 'batch-test',
                'input_sha256': hashlib.sha256((directory/'input.jsonl').read_bytes()).hexdigest()})
            response = Response(id='resp-test', created_at=1, model='gpt-6-luna', object='response', status='completed',
                parallel_tool_calls=True, tool_choice='auto', tools=[], output=[{'id': 'msg-test', 'type': 'message',
                'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'annotations': [],
                'text': json.dumps({'label': 'Same granularity', 'reason': 'test'})}]}])
            output = '\n'.join(json.dumps({'custom_id': k, 'error': None,
                'response': {'status_code': 200, 'body': response.model_dump()}}) for k in reversed(list(jobs))).encode()
            client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _: SimpleNamespace(id='batch-test',
                status='completed', request_counts={}, error_file_id=None, output_file_id='file-out')),
                files=SimpleNamespace(content=lambda _: SimpleNamespace(content=output)))
            self.assertTrue(batch.collect(directory, cache, self.labels, client))
            self.assertTrue(batch.collect(directory, cache, self.labels, client))
            self.assertEqual(set(judge.read_cache(cache)), set(jobs))
            self.assertEqual(len(study.read_jsonl(cache)), 2)

    def test_pending_batch_does_not_touch_cache(self):
        from . import batch
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            study.save(directory/'input.jsonl', [{'custom_id': 'test'}], jsonl=True)
            study.save(directory/'state.json', {'batch_id': 'batch-test',
                'input_sha256': hashlib.sha256((directory/'input.jsonl').read_bytes()).hexdigest()})
            client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _: SimpleNamespace(
                id='batch-test', status='in_progress', request_counts={})))
            self.assertFalse(batch.collect(directory, directory/'cache.jsonl', self.labels, client))
            self.assertFalse((directory/'cache.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
