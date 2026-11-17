"""Offline recognition tests; no model downloads or CUDA required."""
import hashlib
import math
import unittest

from . import recognition
from .scoring import (ContinuationEncoder, answers_are_confusable, build_candidate_pools,
                      build_hierarchy_conflicts, create_prompt_variants, summarize_token_logprobs)


class FakeScorer:
    def score(self, questions, candidates):
        return [summarize_token_logprobs([-1. if a in {'Target Person', 'Target Group'} else -10.]) for a in candidates]


def sources():
    rows = [{'id': i, 'question': f'Who founded Object {i}?', 'question_entity': f'Object {i}',
             'question_entity_qid': f'Q{i}', 'relation': 'P112',
             'granola_answers': [hashlib.sha256(f'{i}:person'.encode()).hexdigest()[:16],
                                 hashlib.sha256(f'{i}:group'.encode()).hexdigest()[:16]],
             'normalized_levels': [1., 4.]} for i in range(50)]
    rows[0]['granola_answers'] = ['Target Person', 'Target Group']
    return rows


class RecognitionTests(unittest.TestCase):
    def test_full_protocol_and_selection(self):
        rows = sources()
        result = recognition.score_row(rows[0], FakeScorer(), build_candidate_pools(rows), build_hierarchy_conflicts(rows), 'lfm')
        self.assertEqual(result['selected_level'], 1.)
        self.assertEqual(result['selected_answer'], 'Target Person')
        self.assertEqual(result['scoring_model'], 'LFM2_5_1_2B')
        self.assertEqual(len(result['prompt_variants']), 3)
        self.assertTrue(all(len(level['distractors']) == 20 for level in result['level_results']))

    def test_all_models_share_the_same_protocol(self):
        rows = sources()
        pool, conflicts = build_candidate_pools(rows), build_hierarchy_conflicts(rows)
        results = [recognition.score_row(rows[0], FakeScorer(), pool, conflicts, m) for m in recognition.MODEL_NAMES]
        self.assertTrue(all(r['level_results'] == results[0]['level_results'] for r in results))

    def test_no_level_does_not_create_idk(self):
        row = sources()[0]
        result = recognition.score_row(row, FakeScorer(), {}, {}, 'lfm')
        self.assertIsNone(result['selected_level'])
        self.assertIsNone(result['selected_answer'])
        self.assertTrue(result['no_level_chosen'])

    def test_prefix_stops_at_first_failure(self):
        levels = [{'level': 4., 'passed': True}, {'level': 2.5, 'passed': False}, {'level': 1., 'passed': True}]
        self.assertEqual(recognition.select_level(levels)['level'], 4.)
        levels[0]['passed'] = False
        self.assertIsNone(recognition.select_level(levels))

    def test_every_prompt_and_distractor_must_be_beaten_strictly(self):
        raw = {'correct': {'prompt_scores': [{'mean_logprob': -1.}]*3},
               'distractors': [{'prompt_scores': [{'mean_logprob': -2.}]*3} for _ in range(20)]}
        self.assertTrue(recognition.measure_level(raw)['passed'])
        raw['distractors'][0]['prompt_scores'] = [{'mean_logprob': -1.}]*3
        self.assertFalse(recognition.measure_level(raw)['passed'])
        raw['distractors'][0]['prompt_scores'] = [{'mean_logprob': math.nan}]*3
        with self.assertRaises(ValueError):
            recognition.measure_level(raw)

    def test_prompts_and_panels_are_deterministic(self):
        rows = sources()
        self.assertEqual(create_prompt_variants(rows[0], 3, 42), create_prompt_variants(rows[0], 3, 42))
        args = rows[0], FakeScorer(), build_candidate_pools(rows), build_hierarchy_conflicts(rows), 'lfm'
        self.assertEqual(recognition.score_row(*args), recognition.score_row(*args))

    def test_distractor_conflicts(self):
        self.assertTrue(answers_are_confusable('New York', 'New York City'))
        self.assertTrue(answers_are_confusable('person', 'writer', {'person': {'writer'}}))
        self.assertFalse(answers_are_confusable('Alpha', 'Zulu'))

    def test_hf_cleaning_and_levels(self):
        rows = []
        for i, n in enumerate([1, 2, 3, 4, 5]):
            rows.append({'Unnamed: 0': i, 'relation': 'P112', 'question': f'Question {i}?',
                'question_entity': 'Object', 'question_entity_qid': 'Q1', 'score_for_potential_error': 0,
                **{f'granola_answer_{j}': f'Answer {j}' if j <= n else None for j in range(1, 15)}})
        result = recognition.prepare_rows(rows)
        self.assertEqual([r['id'] for r in result], [0, 1, 2, 3])
        self.assertEqual([r['normalized_levels'] for r in result], [[1.], [1., 4.], [1., 2.5, 4.], [1., 2., 3., 4.]])
        rows[0]['score_for_potential_error'] = 1
        self.assertNotIn(0, [r['id'] for r in recognition.prepare_rows(rows)])

    def test_encoder_scores_only_candidate_without_eos(self):
        class Tokenizer:
            is_fast = True
            def apply_chat_template(self, messages, **kwargs):
                return 'User: '+messages[0]['content']+'\nAssistant:'
            def __call__(self, text, **kwargs):
                return {'input_ids': list(range(len(text))), 'offset_mapping': [(i, i+1) for i in range(len(text))]}
        encoded = ContinuationEncoder(Tokenizer(), '').encode('Who?', 'Target')
        self.assertEqual(len(encoded['candidate_positions']), len('Target'))
        self.assertEqual(encoded['candidate_positions'][-1], len(encoded['input_ids'])-1)
        with self.assertRaises(ValueError):
            summarize_token_logprobs([])


if __name__ == '__main__':
    unittest.main()
