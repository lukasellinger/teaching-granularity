"""Final Base/FT audit sampling, frozen inputs and annotation integrity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from shared.io import read_rows, file_hash
from .audit import check_files
from .base_ft import choose_pairs, evaluate as evaluate_base_ft, MODELS
from .study import evaluate, read_annotations


DIRECTORY = Path(__file__).resolve().parent
FINAL = DIRECTORY/'audits/base-ft-20pairs-v1'


def example(item='one'):
    return {'item_id': item, 'question': 'Where was Example born?', 'answer': 'Example was born in York.',
            'extraction': {'status': 'ok', 'raw_location': 'York'}, 'normalization': {'status': 'normalized'}}


class AuditTests(unittest.TestCase):
    def test_base_ft_sampling_is_balanced_person_disjoint_and_deterministic(self):
        population = {(model, relation): {f'Q{1000*i+100*j+n}' for n in range(1, 9)}
                      for i, model in enumerate(MODELS, 1)
                      for j, relation in enumerate(('birthplace', 'deathplace'))}
        # One person is eligible in every stratum but may be selected only once.
        for people in population.values():
            people.add('Q99999')
        selected = choose_pairs(population, 20261116)
        self.assertEqual(selected, choose_pairs(population, 20261116))
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({qid for _, qid, _ in selected}), 20)
        self.assertEqual(Counter(model for model, _, _ in selected), dict.fromkeys(MODELS, 5))
        self.assertEqual(Counter(relation for _, _, relation in selected), {'birthplace': 10, 'deathplace': 10})
        self.assertTrue(all(qid in population[model, relation] for model, qid, relation in selected))
        with self.assertRaisesRegex(ValueError, 'Insufficient fresh people'):
            choose_pairs({slot: {'Q1'} for slot in population}, 20261116)

    def test_final_audit_reproduces_recorded_results_without_changing_files(self):
        before = {p.name: file_hash(p) for p in FINAL.iterdir() if p.is_file()}
        check_files(FINAL, json.loads((FINAL/'frozen.json').read_text()))
        annotations = read_annotations(FINAL/'annotations.final.csv')
        manifest = json.loads((FINAL/'manifest.json').read_text())
        result = evaluate_base_ft(read_rows(FINAL/'cases.jsonl'), annotations, manifest['utility_gains'])
        saved = json.loads((FINAL/'results.json').read_text())
        for key, value in result.items():
            self.assertEqual(value, saved[key], key)
        self.assertEqual(file_hash(FINAL/'annotations.final.csv'), saved['annotations_sha256'])
        self.assertEqual(result['processing']['confirmed_correct'], 36)
        self.assertEqual(result['complete_pairs_for_utility'], 20)
        self.assertEqual(before, {p.name: file_hash(p) for p in FINAL.iterdir() if p.is_file()})

    def test_manual_scores_require_four_level_evaluation(self):
        annotations = copy.deepcopy(read_annotations(FINAL/'annotations.final.csv'))
        correct = next(row for row in annotations if row['manual_outcome'] == 'Correct')
        correct['manual_level'] = 'PLACE'
        manifest = json.loads((FINAL/'manifest.json').read_text())
        with self.assertRaisesRegex(ValueError, 'finest supported evaluation level'):
            evaluate_base_ft(read_rows(FINAL/'cases.jsonl'), annotations, manifest['utility_gains'])

    def test_scoring_requires_full_unchanged_human_judgments(self):
        cases = [example(), example('two')]
        annotations = [{k: r[k] for k in ('item_id', 'question', 'answer')}
                       | {'review_outcome': 'All checks correct'} for r in cases]
        annotations[1].update(review_outcome='Needs detailed review', extraction_correct='Correct',
                              entity='Incorrect', hierarchy='Correct', granularity='Correct',
                              omissions='Correct', consensus='Cannot determine')
        result = evaluate(cases, annotations)
        self.assertEqual((result['confirmed_correct'], result['incorrect'], result['cases']), (1, 1, 2))
        for changed in (annotations[:1], annotations+[annotations[0]],
                        [{**annotations[0], 'answer': 'different'}, annotations[1]]):
            with self.assertRaises(ValueError):
                evaluate(cases, changed)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'export.json'
            path.write_text(json.dumps([{'data': {'item_id': 'one'}, 'annotations': []}]))
            with self.assertRaises(ValueError):
                read_annotations(path)

    def test_frozen_input_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = directory/'cases.jsonl'
            path.write_text('original\n')
            frozen = {'files': {'cases.jsonl': file_hash(path)}}
            check_files(directory, frozen)
            path.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'Frozen audit file changed'):
                check_files(directory, frozen)


if __name__ == '__main__':
    unittest.main()
