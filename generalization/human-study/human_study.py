"""Prepare Label Studio tasks, adjudicate disagreements, evaluate granularity."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'generalization.human-study'

import unicodedata
from urllib.parse import urlparse
from generalization.benchmark_data import held_out_relations, load_cases

import numpy as np

ROOT = Path(__file__).resolve().parent
ORDER = {'finetuned_more_specific': -1, 'equal': 0, 'base_more_specific': 1}
OTHER = {'Same granularity': 'equal', 'Not comparable': 'not_comparable',
         'Abstention present': 'abstention_present'}
METHODS = {'granuscore': [1.0], 'length': [0.0], 'combined': np.linspace(0, 1, 11)}
METHOD_LABELS = {'granuscore': 'GranuScore', 'length': 'Length', 'combined': 'GranuScore + length', 'llm_judge': 'LLM judge'}
MODELS = ('lfm', 'qwen', 'gemma', 'ministral')
from shared.constants import MODEL_NAMES as MODEL_IDS
QUOTAS = {'granola': 17, 'simpleqa': 17, 'truthfulqa': 16}
CORRECTNESS = ('Correct', 'Incorrect', 'Abstain', 'Cannot determine')
EVIDENCE_ROUTES = ('References only', 'Web verification', 'Unresolved without lookup')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def abstention(text):
    value = unicodedata.normalize('NFKC', text).replace('’', "'").replace('‘', "'")
    value = re.sub(r'\s+', ' ', value).casefold().strip(" \t\n\r.,!?;:*_`#\"'“”()[]{}")
    return value in {'idk', "i don't know", 'i do not know'}


def sample_pairs(root, generations, seed):
    """Seeded stratified sampling of non-identical, non-IDK generation pairs."""
    rows = []
    for model in MODELS:
        for dataset, quota in QUOTAS.items():
            paired = {}
            for record in read_jsonl(generations/dataset/f'{model}.jsonl'):
                variant = record['variant']
                if variant not in ('base', 'finetuned'):
                    raise ValueError('Expected base/finetuned generation variants')
                pair = paired.setdefault(str(record['id']), {})
                if variant in pair:
                    raise ValueError('Duplicate generation ID/variant')
                pair[variant] = record
            candidates = {}
            for source_id, pair in sorted(paired.items()):
                if set(pair) != {'base', 'finetuned'}:
                    raise ValueError('Incomplete generation pair')
                base, ft = pair['base'], pair['finetuned']
                if any(base.get(k) != ft.get(k) for k in ('question', 'qid', 'relation')):
                    raise ValueError('Base/FT identity mismatch')
                if dataset == 'granola' and base['relation'] not in held_out_relations():
                    continue
                texts = [pair[v]['first_sentence'] for v in ('base', 'finetuned')]
                if any(not isinstance(t, str) or not t.strip() for t in texts):
                    raise ValueError('Stored first_sentence required')
                if re.sub(r'\s+', ' ', texts[0]).strip() == re.sub(r'\s+', ' ', texts[1]).strip() or any(map(abstention, texts)):
                    continue
                row = {'model': model, 'dataset': dataset, 'source_id': source_id,
                       'question': base['question'], 'pair_id': digest([MODEL_IDS[model], dataset, source_id])}
                for variant in ('base', 'finetuned'):
                    score = pair[variant]['first_sentence_granuscore']
                    if not math.isfinite(score):
                        raise ValueError('Finite cached GranuScore required')
                    row[variant] = {'text': pair[variant]['first_sentence'], 'granuscore': score}
                candidates[source_id] = row
            rows.extend(random.Random(f'{seed}:{MODEL_IDS[model]}:{dataset}').sample(list(candidates.values()), quota))
    rows.sort(key=lambda r: r['pair_id'])
    for name in ('annotator_1', 'annotator_2'):
        orientations = ['base']*(len(rows)//2) + ['finetuned']*(len(rows)-len(rows)//2)
        random.Random(f'{seed}:specificity:{name}').shuffle(orientations)
        for row, a in zip(rows, orientations):
            row.setdefault('assignments', {})[name] = {
                'task_uid': digest([seed, row['pair_id'], name]), 'a_variant': a,
                'b_variant': 'finetuned' if a == 'base' else 'base'}
    for row in rows:
        row['adjudication_task_uid'] = digest([seed, row['pair_id'], 'adjudication'])
    if (root/'data/pairs.jsonl').exists():
        raise ValueError('Output pairs.jsonl already exists; use --root for a new output directory.')
    save(root/'data/pairs.jsonl', rows, jsonl=True)
    prepare(root, rows)
    print(f'Created {len(rows)} pairs from generation records.')


def read_jsonl(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    if not rows:
        raise ValueError(f'Empty input: {path}')
    return rows


def save(path, value, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        text = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in value)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    path.write_text(text, encoding='utf-8')


def load_study(root):
    rows = read_jsonl(root/'data/pairs.jsonl')
    if len({r['pair_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate study pair')
    ids = []
    for row in rows:
        for variant in ('base', 'finetuned'):
            if not row[variant]['text'].strip() or not math.isfinite(row[variant]['granuscore']):
                raise ValueError('Every answer requires text and a finite cached GranuScore')
        for name in ('annotator_1', 'annotator_2'):
            a = row['assignments'][name]
            if {a['a_variant'], a['b_variant']} != {'base', 'finetuned'}:
                raise ValueError('Invalid A/B assignment')
            ids.append(a['task_uid'])
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate annotator task ID')
    discussion_ids = [r['adjudication_task_uid'] for r in rows]
    if len(set(discussion_ids)) != len(discussion_ids):
        raise ValueError('Duplicate adjudication task ID')
    return rows


def task_data(row, assignment):
    return {'task_uid': assignment['task_uid'], 'prompt': row['question'],
            'response_a': row[assignment['a_variant']]['text'],
            'response_b': row[assignment['b_variant']]['text']}


def prepare(root, rows):
    for name in ('annotator_1', 'annotator_2'):
        tasks = [{'data': task_data(r, r['assignments'][name])} for r in rows]
        random.Random(f'42:consolidated-task-order:{name}').shuffle(tasks)
        save(root/f'label_studio/{name}.json', tasks)
    print(f'Prepared {len(rows)} tasks per annotator in {root / "label_studio"}.')


def choice(text, assignment):
    if text.startswith('['):
        values = json.loads(text)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError('One annotation choice required')
        text = values[0]
    if text in ('A more fine-grained', 'B more fine-grained'):
        return assignment['a_variant' if text.startswith('A ') else 'b_variant']+'_more_specific'
    if text not in OTHER:
        raise ValueError(f'Invalid annotation choice: {text!r}')
    return OTHER[text]


def displayed(label, assignment):
    if label in ('base_more_specific', 'finetuned_more_specific'):
        return 'A more fine-grained' if assignment['a_variant'] == label.removesuffix('_more_specific') else 'B more fine-grained'
    return {v: k for k, v in OTHER.items()}[label]


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def correctness_tasks(rows, references, seed=42):
    """One blinded task per answer, using the Granola subset of the same pairs."""
    index = {str(r['id']): r for r in references}
    if len(index) != len(references):
        raise ValueError('Duplicate Granola reference ID')
    tasks, keys = [], {}
    for row in rows:
        if row['dataset'] != 'granola':
            continue
        reference = index[row['source_id']]
        if reference['question'] != row['question']:
            raise ValueError('Reference question does not match study pair')
        answers = reference['reference_answers']
        if not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError('Nonempty reference hierarchy required')
        reference_text = 'Acceptable answer hierarchy (fine to coarse):\n'+'\n'.join('• '+a for a in answers)
        for variant in ('base', 'finetuned'):
            uid = digest(['correctness-single-annotator-v1', row['pair_id'], variant])
            data = {'task_uid': uid, 'prompt': row['question'], 'response': row[variant]['text'],
                    'reference_text': reference_text,
                    'reference_sources': 'No source links supplied; targeted lookup is allowed.'}
            if uid in keys:
                raise ValueError('Duplicate correctness task')
            keys[uid] = {'pair_id': row['pair_id'], 'model': row['model'],
                         'dataset': row['dataset'], 'variant': variant,
                         'reference_answers': answers}
            tasks.append({'data': data})
    if not tasks:
        raise ValueError('No Granola pairs in the study')
    random.Random(seed).shuffle(tasks)
    for i, task in enumerate(tasks, 1):
        task['data']['progress'] = f'{i} / {len(tasks)}'
    return tasks, keys


def read_correctness(tasks, keys, export):
    """Match judgments to exact displayed question, answer and references."""
    fields = ('prompt', 'response', 'reference_text', 'reference_sources')
    by_uid = {t['data']['task_uid']: t['data'] for t in tasks}
    by_text = {}
    for data in by_uid.values():
        by_text.setdefault(tuple(data[k] for k in fields), []).append(data)
    imported, seen = [], set()
    for record in csv_rows(export):
        if not record.get('annotation_id', '').strip() or not record.get('annotator', '').strip():
            raise ValueError('Completed correctness annotation and annotator required')
        signature = tuple(record.get(k) for k in fields)
        data = by_uid.get(record.get('task_uid'))
        if data is None:
            # Export IDs can differ after a Label Studio reimport. Require a
            # unique exact-text match; never infer a label or match fuzzily.
            matches = by_text.get(signature, [])
            if len(matches) != 1:
                raise ValueError('Unknown or ambiguous correctness answer')
            data = matches[0]
        if signature != tuple(data[k] for k in fields) or data['task_uid'] in seen:
            raise ValueError('Altered or duplicate correctness answer')
        seen.add(data['task_uid'])
        reference = record.get('reference_correctness')
        final = record.get('correctness')
        route = record.get('evidence_route')
        if reference not in CORRECTNESS or final not in CORRECTNESS or route not in EVIDENCE_ROUTES:
            raise ValueError('Invalid correctness judgment or evidence route')
        urls, note = (record.get(k, '').strip() for k in ('evidence_urls', 'evidence_note'))
        if route == 'Web verification':
            links = urls.split()
            if not links or not note or any(urlparse(u).scheme not in ('http', 'https') or not urlparse(u).netloc for u in links):
                raise ValueError('Web verification requires source URLs and an evidence note')
        elif final != reference:
            raise ValueError('Changed correctness verdict requires web verification')
        if route == 'Unresolved without lookup' and final != 'Cannot determine':
            raise ValueError('Unresolved answers must remain Cannot determine')
        imported.append({**keys[data['task_uid']], 'question': data['prompt'],
            'response': data['response'], 'reference_correctness': reference, 'correctness': final,
            'evidence_route': route, 'evidence_urls': urls, 'evidence_note': note,
            'review_note': record.get('review_note', ''),
            'annotator': record['annotator'], 'annotation_id': record['annotation_id']})
    if seen != by_uid.keys():
        raise ValueError(f'Expected {len(tasks)} correctness annotations, got {len(seen)}')
    return sorted(imported, key=lambda r: (r['pair_id'], r['variant']))


def evaluate_correctness(root, records):
    def counts(group):
        return {'answers': len(group),
                'reference_correctness': {label: sum(r['reference_correctness'] == label for r in group) for label in CORRECTNESS},
                'correctness': {label: sum(r['correctness'] == label for r in group) for label in CORRECTNESS},
                'evidence_route': dict(Counter(r['evidence_route'] for r in group))}
    summary = {**counts(records), 'pairs': len({r['pair_id'] for r in records}),
               'changed_after_verification': sum(r['reference_correctness'] != r['correctness'] for r in records),
               'by_model': {model: {variant: counts([r for r in records if r['model'] == model and r['variant'] == variant])
                   for variant in ('base', 'finetuned')} for model in sorted({r['model'] for r in records})}}
    save(root/'results/correctness.jsonl', records, jsonl=True)
    save(root/'results/correctness-summary.json', summary)
    lines = [f'Correctness human study: {summary["pairs"]} Granola pairs, {len(records)} individual answers.',
             'One human annotation per answer; reference-only and final judgments kept separate.',
             '', 'Final judgments (counts)',
             'Model        Variant       Correct Incorrect Abstain Cannot determine']
    for model, variants in summary['by_model'].items():
        for variant, result in variants.items():
            values = result['correctness']
            lines.append(f'{model:<12} {variant:<13} {values["Correct"]:>6} {values["Incorrect"]:>9} {values["Abstain"]:>7} {values["Cannot determine"]:>16}')
    lines += ['', 'This is the annotated Granola subset, not full-benchmark accuracy.', '']
    (root/'results/correctness-summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def check_text(row, expected):
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Export text does not match task {expected["task_uid"]}')


def independent_labels(rows, paths):
    result = {}
    for name, path in zip(('annotator_1', 'annotator_2'), paths, strict=True):
        tasks = {r['assignments'][name]['task_uid']: r for r in rows}
        seen = set()
        for record in csv_rows(path):
            uid = record.get('task_uid')
            if uid not in tasks or uid in seen or not record.get('annotation_id', '').strip():
                raise ValueError(f'Unknown, duplicate or incomplete annotation in {path}')
            seen.add(uid)
            row = tasks[uid]
            assignment = row['assignments'][name]
            check_text(record, task_data(row, assignment))
            result[(row['pair_id'], name)] = choice(record.get('specificity', '').strip(), assignment)
        if seen != tasks.keys():
            raise ValueError(f'{name}: expected exactly {len(rows)} annotations, got {len(seen)}')
    return result


def discussion_tasks(rows, labels):
    tasks = []
    disputed = [r for r in rows if labels[(r['pair_id'], 'annotator_1')] != labels[(r['pair_id'], 'annotator_2')]]
    for i, row in enumerate(disputed, 1):
        a = {**row['assignments']['annotator_1'], 'task_uid': row['adjudication_task_uid']}
        tasks.append({'data': {**task_data(row, a), 'progress': f'Disagreement {i} of {len(disputed)}',
                               'annotator_1_label': displayed(labels[(row['pair_id'], 'annotator_1')], a),
                               'annotator_2_label': displayed(labels[(row['pair_id'], 'annotator_2')], a)}})
    return tasks


def agreement(y, predictions):
    if not y:
        return {'n': 0, 'accuracy': None, 'kappa': None}
    n = len(y)
    accuracy = sum(a == b for a, b in zip(y, predictions, strict=True))/n
    a, b = Counter(y), Counter(predictions)
    chance = sum(a[k]*b[k] for k in a.keys() | b.keys())/n**2
    return {'n': n, 'accuracy': accuracy,
            'kappa': (accuracy-chance)/(1-chance) if chance < 1 else None}


def final_labels(rows, independent, path):
    tasks = {r['data']['task_uid']: r['data'] for r in discussion_tasks(rows, independent)}
    decisions = {}
    for record in csv_rows(path):
        uid = record.get('task_uid')
        if uid not in tasks or uid in decisions or not record.get('annotation_id', '').strip():
            raise ValueError('Unknown, duplicate or incomplete adjudication')
        check_text(record, tasks[uid])
        decisions[uid] = record.get('final_specificity', '').strip()
    if decisions.keys() != tasks.keys():
        raise ValueError('Adjudication export must cover every disagreement')
    result = {}
    for row in rows:
        uid = row['pair_id']
        a, b = independent[(uid, 'annotator_1')], independent[(uid, 'annotator_2')]
        result[uid] = a if a == b else choice(decisions[row['adjudication_task_uid']], row['assignments']['annotator_1'])
    return result


def adjudicated_study(root):
    """Read the displayed pairs and adjudicated labels from the human exports."""
    rows = load_study(root)
    independent = independent_labels(rows, [root/'annotations/annotator-1.csv', root/'annotations/annotator-2.csv'])
    return rows, final_labels(rows, independent, root/'annotations/adjudication.csv')


def predict(score, band):
    return np.where(score > band, 1, np.where(score < -band, -1, 0))


def fit(g, length, y, alphas):
    gs, ls = float(np.std(g)), float(np.std(length))
    gs, ls = gs if gs > 1e-12 else 1., ls if ls > 1e-12 else 1.
    best = None
    for alpha in alphas:
        score = float(alpha)*g/gs+(1-float(alpha))*length/ls
        absolute = np.unique(np.abs(score))
        bands = np.unique(np.concatenate(([0.], absolute)))
        matches = (predict(score[None, :], bands[:, None]) == y[None, :]).sum(axis=1)
        for i in np.flatnonzero(matches == matches.max()):
            band = float(bands[i])
            higher = absolute[absolute > band]
            if band > 0 and len(higher):
                band = float((band+higher[0])/2)
            rank = (int(matches[i]), -float(alpha), -band)
            if best is None or rank > best[0]:
                best = rank, {'alpha': float(alpha), 'band': band,
                              'granuscore_scale': gs, 'length_scale': ls}
    return best[1]


def apply(parameters, g, length):
    p = parameters
    return predict(p['alpha']*g/p['granuscore_scale']+(1-p['alpha'])*length/p['length_scale'], p['band'])


def evaluate(root, rows, labels, independent, folds, seed):
    question = lambda r: re.sub(r'\s+', ' ', r['question']).strip().casefold()
    groups = sorted({question(r) for r in rows})
    if not 2 <= folds <= len(groups):
        raise ValueError('Invalid number of question-grouped folds')
    rng = np.random.default_rng(seed)
    mapping = {groups[int(i)]: j % folds for j, i in enumerate(rng.permutation(len(groups)))}
    selected = [r for r in rows if labels[r['pair_id']] in ORDER]
    if not selected:
        raise ValueError('No orderable final labels')
    assignment = np.array([mapping[question(r)] for r in selected])
    y = np.array([ORDER[labels[r['pair_id']]] for r in selected])
    g = np.array([r['finetuned']['granuscore']-r['base']['granuscore'] for r in selected])
    length = np.array([len(r['base']['text'].split())-len(r['finetuned']['text'].split()) for r in selected])
    predictions, recipes = {}, {}
    for method, alphas in METHODS.items():
        prediction = np.full(len(y), 99, dtype=int)
        recipes[method] = []
        for fold in range(folds):
            train, test = assignment != fold, assignment == fold
            if not train.any() or not test.any():
                raise ValueError('Empty orderable training or test fold')
            parameters = fit(g[train], length[train], y[train], alphas)
            prediction[test] = apply(parameters, g[test], length[test])
            recipes[method].append({'fold': fold+1, **parameters})
        if (prediction == 99).any():
            raise ValueError('Missing held-out prediction')
        predictions[method] = prediction
    summary = {'pairs': len(rows), 'evaluated_pairs': len(selected), 'label_counts': dict(Counter(labels.values())),
               'human_agreement': agreement([independent[(r['pair_id'], 'annotator_1')] for r in rows],
                                            [independent[(r['pair_id'], 'annotator_2')] for r in rows]),
               'cross_validation': {'folds': folds, 'seed': seed, 'group_by': 'question'},
               'overall': {m: agreement(y.tolist(), p.tolist()) for m, p in predictions.items()}, 'by_model': {}}
    for model in sorted({r['model'] for r in selected}):
        mask = np.array([r['model'] == model for r in selected])
        summary['by_model'][model] = {m: agreement(y[mask].tolist(), p[mask].tolist()) for m, p in predictions.items()}
    full = fit(g, length, y, METHODS['combined'])
    metric = {'parameters': full, 'fit_pairs': len(selected), 'scope': 'first_sentence',
              'formula': 'alpha*(FT GranuScore-base GranuScore)/granuscore_scale + (1-alpha)*(base words-FT words)/length_scale',
              'decision': 'Inside +/-band: equal; above band: FT coarser; below -band: FT finer.',
              'calibration_in_sample': agreement(y.tolist(), apply(full, g, length).tolist())}
    output = root/'results'
    save(output/'summary.json', summary)
    save(output/'metric.json', metric)
    save(output/'folds.json', recipes)
    save(output/'final_labels.jsonl', [{'pair_id': r['pair_id'], 'label': labels[r['pair_id']]} for r in rows], jsonl=True)
    save(output/'predictions.jsonl', [{'pair_id': r['pair_id'], 'model': r['model'], 'fold': int(assignment[i])+1,
                                     'human_label': labels[r['pair_id']],
                                     **{m: int(p[i]) for m, p in predictions.items()}} for i, r in enumerate(selected)], jsonl=True)
    human = summary['human_agreement']
    lines = [f'Human study: {len(rows)} pairs, two independent annotators, joint adjudication.',
             f"Independent agreement: {100*human['accuracy']:.2f}%; Cohen kappa: {human['kappa']:.4f}.",
             f'{folds}-fold question-grouped CV, seed {seed}; {len(selected)} orderable pairs, including equal.',
             '', 'Held-out performance', 'Model        Method                  Accuracy (%)   Cohen kappa']
    for model, methods in [('overall', summary['overall']), *summary['by_model'].items()]:
        for method, value in methods.items():
            kappa = 'n/a' if value['kappa'] is None else f"{value['kappa']:.4f}"
            lines.append(f"{model:<13}{METHOD_LABELS.get(method, method):<24}{100*value['accuracy']:>8.2f}       {kappa}")
    lines.extend(['', 'The saved metric is refitted on all orderable pairs for use on new answers.',
                  'Its calibration agreement is in-sample, not held-out validation.', ''])
    (output/'summary.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT, help='Study directory; defaults to this script directory')
    commands = parser.add_subparsers(dest='command', required=True)
    sampling = commands.add_parser('sample', help='Create pairs.jsonl and blinded tasks from generations')
    sampling.add_argument('--generations-dir', type=Path, default=ROOT.parent/'generations')
    sampling.add_argument('--seed', type=int, default=42)
    commands.add_parser('prepare', help='Write blinded tasks for both annotators')
    commands.add_parser('prepare-correctness', help='Write individual Granola correctness tasks')
    correctness = commands.add_parser('evaluate-correctness', help='Import and summarize human correctness judgments')
    correctness.add_argument('--annotations', type=Path, help='Label Studio correctness CSV export')
    for name in ('adjudicate', 'evaluate'):
        command = commands.add_parser(name)
        command.add_argument('--annotator-1', type=Path)
        command.add_argument('--annotator-2', type=Path)
        if name == 'evaluate':
            command.add_argument('--adjudication', type=Path)
            command.add_argument('--folds', type=int, default=5)
            command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.command == 'sample':
        sample_pairs(args.root, args.generations_dir, args.seed)
        return
    rows = load_study(args.root)
    if args.command in ('prepare-correctness', 'evaluate-correctness'):
        references = load_cases('granola')
        tasks, keys = correctness_tasks(rows, references)
        if args.command == 'prepare-correctness':
            save(args.root/'label_studio/correctness.json', tasks)
            print(f'Prepared {len(tasks)} individual Granola correctness tasks.')
        else:
            records = read_correctness(tasks, keys,
                args.annotations or args.root/'annotations/correctness.csv')
            evaluate_correctness(args.root, records)
        return
    if args.command == 'prepare':
        prepare(args.root, rows)
        return
    paths = [args.annotator_1 or args.root/'annotations/annotator-1.csv',
             args.annotator_2 or args.root/'annotations/annotator-2.csv']
    labels = independent_labels(rows, paths)
    if args.command == 'adjudicate':
        tasks = discussion_tasks(rows, labels)
        save(args.root/'label_studio/adjudication.json', tasks)
        save(args.root/'results/independent_labels.jsonl', [{'pair_id': r['pair_id'], 'annotator': name, 'label': labels[(r['pair_id'], name)]}
             for r in rows for name in ('annotator_1', 'annotator_2')], jsonl=True)
        print(f'{len(tasks)} disagreement tasks written to label_studio/adjudication.json.')
    else:
        final = final_labels(rows, labels, args.adjudication or args.root/'annotations/adjudication.csv')
        evaluate(args.root, rows, final, labels, args.folds, args.seed)


if __name__ == '__main__':
    main()
