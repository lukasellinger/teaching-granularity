"""Prepare and score a held-out, blinded Base/FT geographic processing audit."""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from shared.io import read_rows, write, file_hash
from .audit import REPO, DIRECTORY, exclusion_questions, question_key, check_files
from .study import evaluate as evaluate_processing, read_annotations, choice
from evaluation.vital_persons.location import location_informativeness
from parsing.location.normalization.values import LocationLevel
from shared.geography import FOUR_LEVEL, FOUR_LEVELS

MODELS = ('lfm', 'ministral', 'qwen', 'gemma')
METHODS = ('base_plain', 'finetuned')
OUTCOMES = {'Correct', 'Incorrect or non-evaluable', 'No answer', 'Cannot determine'}


def exclusions():
    questions = exclusion_questions()
    paths = sorted((DIRECTORY / 'audits').glob('*/cases.jsonl'))
    for path in paths:
        questions.update(question_key(r['question']) for r in read_rows(path))
    names = {}
    for r in read_rows(REPO / 'data_preparation/vital_persons/metadata/gold_adapted.jsonl'):
        for name in (r['name'], (r.get('wiki_title') or '').replace('_', ' ')):
            if name:
                names[question_key(name)] = r['qid']
    people = set((REPO / 'parsing/location/extractor/human_validation/excluded_people.txt').read_text().split())
    for snapshot in sorted((DIRECTORY / 'audits').glob('*/exclusions.json')):
        people.update(json.loads(snapshot.read_text()).get('people', []))
    for q in questions:
        m = re.fullmatch(r'where (?:was|did) (.+) (?:born|die)\?', q)
        if m and m[1] in names:
            people.add(names[m[1]])
    # Preserve direct person IDs as well as question/name matches.
    for name in ('training/train.jsonl', 'training/dev.jsonl', 'human_validation/annotations.jsonl'):
        for r in read_rows(REPO / 'parsing/location/extractor' / name):
            person = r.get('qid') or r.get('person_group', '')
            if re.fullmatch(r'Q\d+', person):
                people.add(person)
    return questions, people


def choose_pairs(population, seed):
    """Five pairs/model, ten pairs/event, twenty distinct people; no output screening."""
    rng = random.Random(seed)
    slots = [(model, relation) for i, model in enumerate(MODELS)
             for relation, n in (('birthplace', 3 if i % 2 == 0 else 2),
                                 ('deathplace', 2 if i % 2 == 0 else 3)) for _ in range(n)]
    candidates = {}
    for slot in sorted(set(slots)):
        candidates[slot] = sorted(population[slot])
        rng.shuffle(candidates[slot])
    rng.shuffle(slots)
    slots.sort(key=lambda s: len(candidates[s]))
    selected, used = [], set()

    def fill(index):
        if index == len(slots):
            return True
        model, relation = slot = slots[index]
        for qid in candidates[slot]:
            if qid in used:
                continue
            used.add(qid); selected.append((model, qid, relation))
            if fill(index + 1):
                return True
            selected.pop(); used.remove(qid)
        return False

    if not fill(0):
        raise ValueError('Insufficient fresh people for the predefined model/event balance')
    rng.shuffle(selected)
    return selected


def prepare(args):
    directory = args.output_dir
    if directory.exists():
        raise ValueError('Use a new directory; previous audits are preserved')
    questions, people = exclusions()
    population, records, hashes = {}, {}, {}
    for model in MODELS:
        indices, outputs = {}, {}
        for method in METHODS:
            source = REPO / f'evaluation/vital_persons/results/{model}/test/location_extractions/{method}.jsonl'
            outcome = REPO / f'evaluation/vital_persons/results-corrected/{model}/test/outcomes/location/{method}.jsonl'
            for path in (source, outcome):
                hashes[str(path.relative_to(REPO))] = file_hash(path)
            indices[method] = {r['case_id']: r for r in read_rows(source)}
            outputs[method] = {r['case_id']: r for r in read_rows(outcome)}
            if {r.get('location_taxonomy') for r in outputs[method].values()} != {FOUR_LEVEL}:
                raise ValueError('This audit interface requires four-level evaluation outcomes')
        for case_id, r in outputs['base_plain'].items():
            if r['gold'] is None or r['qid'] in people or question_key(r['prompt']) in questions:
                continue
            if any(case_id not in indices[m] or case_id not in outputs[m] for m in METHODS):
                raise ValueError('Missing member of an evaluation pair')
            for method in METHODS:
                raw, scored = indices[method][case_id], outputs[method][case_id]
                if raw['answer'] != scored['answer'] or raw['question'] != r['prompt']:
                    raise ValueError('Saved pipeline and evaluation inputs disagree')
                if scored['gold'] != r['gold']:
                    raise ValueError('Pair references differ')
            key = (model, r['qid'], r['relation'])
            records[key] = {m: (indices[m][case_id], outputs[m][case_id]) for m in METHODS}
            population.setdefault((model, r['relation']), set()).add(r['qid'])
    selected = choose_pairs(population, args.seed)
    gains = {level.value: location_informativeness(level) for level in LocationLevel}
    cases = []
    for number, key in enumerate(selected, 1):
        for method in METHODS:
            raw, scored = records[key][method]
            cases.append({'pair_id': f'pair-{number:02d}', 'model': key[0], 'qid': key[1],
                          'relation': key[2], 'method': method, 'question': raw['question'],
                          'answer': raw['answer'], 'extraction': raw['extraction'],
                          'normalization': raw['normalization'], 'gold': scored['gold'],
                          'saved_evaluation': {k: scored[k] for k in ('truth_correct', 'parse_status',
                              'prediction', 'utility_uniform_wrong_penalty')}})
    random.Random(args.seed + 1).shuffle(cases)
    tasks = []
    for number, case in enumerate(cases, 1):
        case['item_id'] = f'geo-{number:02d}'
        tasks.append({'data': {'item_id': case['item_id'], 'question': case['question'],
                       'answer': case['answer'],
                       'extractor_output': case['extraction'].get('raw_location') or '(no output)',
                       'extraction_status': case['extraction']['status'],
                       'normalization': json.dumps(case['normalization'], ensure_ascii=False, indent=2),
                       'gold_reference': json.dumps(case['gold'], ensure_ascii=False, indent=2)}})
    directory.mkdir(parents=True)
    write(directory / 'cases.jsonl', cases, rows=True)
    write(directory / 'tasks.json', tasks)
    write(directory / 'exclusions.json', {'questions': sorted(questions), 'people': sorted(people)})
    config = Path(__file__).with_name('base_ft_labeling_config.xml')
    (directory / 'labeling_config.xml').write_bytes(config.read_bytes())
    exclusion_sources = [REPO / 'parsing/location/extractor' / n for n in (
        'training/train.jsonl', 'training/dev.jsonl', 'human_validation/annotations.jsonl',
        'human_validation/excluded_people.txt')]
    exclusion_sources += [REPO / 'data_preparation/vital_persons/metadata/gold_adapted.jsonl', DIRECTORY / 'prior_questions.json', REPO / 'evaluation/vital_persons/corrections.jsonl',
                          REPO / 'parsing/location/normalization/tests/fixtures/regressions.jsonl']
    exclusion_sources += sorted((DIRECTORY / 'audits').glob('*/items.jsonl'))
    exclusion_sources += [p for p in sorted((DIRECTORY / 'audits').glob('*/cases.jsonl'))
                          if p.parent != directory]
    manifest = {'pairs': 20, 'responses': 40, 'unique_people': 20, 'split': 'test', 'seed': args.seed,
                'pairs_by_model': dict(Counter(k[0] for k in selected)),
                'pairs_by_relation': dict(Counter(k[2] for k in selected)),
                'eligible_people_by_model': {m: len(set.union(*(population.get((m, r), set()) for r in
                    ('birthplace', 'deathplace')))) for m in MODELS},
                'sampling': 'Five matched Base/FT question pairs per model; ten birth and ten death questions; distinct people. Gold-available test questions only. No filtering by response length, quality, abstention, extraction/normalization status or correctness.',
                'exclusions': 'Exclude whole identifiable people from recorded extractor train/dev/reference data, normalizer development/regression cases, inspected-person registry, prior audits and reviewed evaluation corrections.',
                'exclusion_scope': 'No overlap with recorded development inputs; unrecorded historical inspection cannot be verified.',
                'excluded_people': len(people), 'excluded_questions': len(questions),
                'source_sha256': hashes,
                'exclusion_source_sha256': {str(p.relative_to(REPO)): file_hash(p) for p in exclusion_sources},
                'script_sha256': file_hash(Path(__file__)), 'utility_gains': gains,
                'evaluation_taxonomy': FOUR_LEVEL,
                'scope': 'Audit frozen saved four-level evaluation outputs; no new inference, corrections or pipeline changes.',
                'human_review': 'pending; no accuracy claimed',
                'audit_protocol': 'Final four-level processing and score-impact audit'}
    write(directory / 'manifest.json', manifest)
    write(directory / 'frozen.json', {'files': {n: file_hash(directory / n) for n in
          ('cases.jsonl', 'tasks.json', 'exclusions.json', 'labeling_config.xml', 'manifest.json')}})
    print(json.dumps({k: manifest[k] for k in ('pairs', 'responses', 'pairs_by_model', 'pairs_by_relation',
                     'eligible_people_by_model', 'excluded_people', 'human_review')}, indent=2))


def evaluate(cases, annotations, gains):
    processing_cases = [{**c, 'normalization': c.get('normalization') or {'status': 'skipped'}} for c in cases]
    processing = evaluate_processing(processing_cases, annotations)
    processing.pop('wilson_95', None)  # Responses are paired; do not report an IID interval.
    index = {r['item_id']: r for r in annotations}
    judgments = []
    for case in cases:
        row = index[case['item_id']]
        outcome = choice(row.get('manual_outcome'))
        if outcome not in OUTCOMES:
            raise ValueError('Unknown manual evaluation outcome')
        utility = None
        if outcome == 'Correct':
            level = choice(row.get('manual_level'))
            if level not in {x.value for x in FOUR_LEVELS}:
                raise ValueError('Use the finest supported evaluation level: CITY, REGION, COUNTRY or CONTINENT')
            utility = gains[level]
        elif outcome != 'Cannot determine':
            utility = 0. if outcome == 'No answer' else -1.
        saved = case['saved_evaluation']
        verdict = next(r['verdict'] for r in processing['judgments'] if r['item_id'] == case['item_id'])
        judgments.append({'item_id': case['item_id'], 'pair_id': case['pair_id'], 'model': case['model'],
                          'method': case['method'], 'processing_verdict': verdict, 'manual_outcome': outcome,
                          'manual_utility': utility,
                          'correctness_label_changed': None if utility is None else
                              (outcome == 'Correct') != saved['truth_correct'],
                          'utility_change': None if utility is None else utility - saved['utility_uniform_wrong_penalty']})
    by_method = {}
    for method in METHODS:
        rows = [r for r in judgments if r['method'] == method]
        known = [r for r in rows if r['manual_utility'] is not None]
        by_method[method] = {'responses': len(rows),
            'processing_counts': dict(Counter(r['processing_verdict'] for r in rows)),
            'evaluation_resolved': len(known),
            'correctness_labels_changed': sum(r['correctness_label_changed'] for r in known),
            'utility_scores_changed': sum(abs(r['utility_change']) > 1e-10 for r in known),
            'mean_utility_change_x100_resolved': 100 * sum(r['utility_change'] for r in known) / len(known) if known else None}
    complete = [p for p in sorted({r['pair_id'] for r in judgments})
                if all(r['manual_utility'] is not None for r in judgments if r['pair_id'] == p)]
    changes = [r['utility_change'] * (1 if r['method'] == 'finetuned' else -1)
               for r in judgments if r['pair_id'] in complete]
    return {'evaluation_taxonomy': FOUR_LEVEL, 'processing': processing, 'by_method': by_method, 'evaluation_judgments': judgments,
            'complete_pairs_for_utility': len(complete),
            'audited_ft_minus_base_utility_difference_change_x100':
                100 * sum(changes) / len(complete) if complete else None,
            'scope': 'Descriptive results for the small balanced audit, not a population correction or full-table ranking estimate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--seed', type=int, default=20261116)
    score = commands.add_parser('score')
    score.add_argument('--audit-dir', type=Path, required=True)
    score.add_argument('--annotations', type=Path, required=True)
    score.add_argument('--output', type=Path, help='New result path; defaults to AUDIT_DIR/results.json')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        directory = args.audit_dir
        destination = args.output or directory / 'results.json'
        if destination.exists():
            raise ValueError('Results already exist; preserve the completed audit')
        check_files(directory, json.loads((directory / 'frozen.json').read_text()))
        manifest = json.loads((directory / 'manifest.json').read_text())
        result = evaluate(read_rows(directory / 'cases.jsonl'), read_annotations(args.annotations), manifest['utility_gains'])
        result['scorer_sha256'] = file_hash(Path(__file__))
        result['annotations_sha256'] = file_hash(args.annotations)
        result['frozen_sha256'] = file_hash(directory / 'frozen.json')
        write(destination, result)
        print(json.dumps(result['by_method'], indent=2))


if __name__ == '__main__':
    main()
