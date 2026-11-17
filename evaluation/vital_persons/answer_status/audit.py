"""Prepare blinded answer-status annotation and compare a frozen judge with humans."""
import argparse
import csv
import json
from pathlib import Path
import random
import re

from shared.io import digest, file_hash, read_rows, write
from .judge import ROOT, RESULTS, LABELS, MODELS, VARIANTS, pool, needs_judge


def content_key(row):
    return digest([row['question'], row['answer']])


ABSTENTION_WORDING = re.compile(
    r"\b(?:unknown|undocumented|unrecorded|unavailable)\b|"
    r"\bnot\s+(?:known|documented|recorded|available)\b|"
    r"\bnot\s+(?:well[- ]|widely\s+)(?:documented|known)\b|"
    r"\bno\s+(?:(?:reliable|historical|definitive|clear|specific|exact|public|confirmed)\s+){0,3}"
    r"(?:record|records|information|evidence|consensus)\b|"
    r"\b(?:cannot|can't|could not|unable to)\s+(?:determine|confirm|provide|find|establish|say)\b|"
    r"\b(?:do not|don't|does not|doesn't)\s+know\b", re.I)
UNCERTAINTY_WORDING = re.compile(
    r"\b(?:uncertain|disputed|approximately|around|possibly|likely)\b|"
    r"\b(?:believed|thought|said)\s+to\b", re.I)


def challenge_kind(row):
    """Sampling heuristics, never proposed human labels."""
    text = row['answer'].replace('\u2019', "'")
    if ABSTENTION_WORDING.search(text):
        return 'abstention_wording'
    if UNCERTAINTY_WORDING.search(text):
        return 'uncertainty_boundary'
    return None


def sample(evaluation, seed=42, count=40, *, excluded=()):
    """Balance a deduplicated sample from the production judge-eligible pool."""
    if count < 1:
        raise ValueError('Sample count must be positive')
    rng = random.Random(seed)
    excluded = set(excluded)
    candidates = sorted((r for r in evaluation if needs_judge(r) and content_key(r) not in excluded),
                        key=lambda r: r['item_id'])
    rng.shuffle(candidates)
    groups = {}
    for row in candidates:
        groups.setdefault((row['model'], row['modality'], row['variant']), []).append(row)
    selected, used = [], set()
    while len(selected) < count:
        progressed = False
        strata = sorted(groups)
        rng.shuffle(strata)
        for key in strata:
            while groups[key]:
                row = groups[key].pop()
                content = content_key(row)
                if content in used:
                    continue
                used.add(content)
                selected.append({**row, 'partition': 'evaluation', 'cohort': 'routed'})
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError(f'Only {len(selected)} distinct eligible responses; requested {count}')
    rng.shuffle(selected)
    return selected


def sample_challenge(population, seed=42, count=20, *, excluded=()):
    if count < 2:
        raise ValueError('Challenge requires at least two tasks to include boundary cases')
    boundary_count = max(1, count//4)
    rows = []
    for kind, quota in [('abstention_wording', count-boundary_count),
                        ('uncertainty_boundary', boundary_count)]:
        selected = sample([r for r in population if challenge_kind(r) == kind],
                          seed, quota, excluded=excluded)
        rows.extend({**r, 'cohort': kind} for r in selected)
    random.Random(seed).shuffle(rows)
    return rows


def prepare(results, directory, seed, count=40, *, challenge=False, exclude_items=()):
    if (directory/'items.jsonl').exists():
        raise ValueError('Audit already exists; use a new output directory rather than overwriting annotations')
    population = pool(results, 'test')
    excluded = {content_key(r) for path in exclude_items for r in read_rows(path)}
    eligible = [r for r in population if needs_judge(r) and content_key(r) not in excluded
                and (not challenge or challenge_kind(r) is not None)]
    sampler = sample_challenge if challenge else sample
    rows = sampler(population, seed, count, excluded=excluded)
    write(directory/'items.jsonl', rows, rows=True)
    write(directory/'evaluation-tasks.json', [{'data': {
        'item_id': r['item_id'], 'question': r['question'], 'answer': r['answer']}} for r in rows])
    write(directory/'manifest.json', {'seed': seed, 'scope': 'full_answer',
        'evaluation_items': len(rows), 'eligible_rows': len(eligible),
        'profile': 'natural-abstention-challenge' if challenge else 'routed',
        'eligible_pool_sha256': digest(sorted(eligible, key=lambda r: r['item_id'])),
        'excluded_items_sha256': [file_hash(path) for path in exclude_items],
        'excluded_content_count': len(excluded),
        'challenge_quotas': {'abstention_wording': count-max(1, count//4),
                             'uncertainty_boundary': max(1, count//4)} if challenge else None,
        'sampling': 'Balanced over model/modality/method where eligible distinct responses exist; content deduplicated.',
        'routing': 'Noncanonical responses whose saved parse_status is not parsed.',
        'caveat': 'Balanced audit of routed failures, not a population prevalence estimate.',
        'items_sha256': digest(rows)})
    print(f'Prepared {len(rows)} annotation tasks from {len(eligible)} eligible rows in {directory}')


def read_labels(path, items):
    expected = {r['item_id']: r for r in items}
    labels = {}
    if path.suffix == '.jsonl':
        for row in read_rows(path):
            uid, label = row['item_id'], row['label']
            if uid not in expected or uid in labels or label not in LABELS:
                raise ValueError('Unknown/duplicate task ID or invalid human label')
            labels[uid] = label
        if labels.keys() != expected.keys():
            raise ValueError(f'{len(expected.keys()-labels.keys())} human labels missing')
        return labels
    with path.open(newline='', encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            raw = row.get('answer_status', '').strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value = raw
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
            if value not in LABELS:
                raise ValueError(f'Invalid human label: {value}')
            uid = row.get('item_id')
            if uid not in expected:
                raise ValueError('Unknown task ID; use the matching audit export and partition')
            if any(key in row and row[key] != expected[uid][key] for key in ('question', 'answer')):
                raise ValueError('Export question/answer differs from frozen audit task')
            if uid in labels:
                raise ValueError('Duplicate annotated task; provide one final human label per task')
            labels[uid] = value
    if labels.keys() != expected.keys():
        raise ValueError(f'{len(expected.keys()-labels.keys())} human labels missing')
    return labels


def metrics(human, predicted):
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for a, b in zip(human, predicted, strict=True):
        confusion[a][b] += 1
    n = len(human)
    if not n:
        raise ValueError('Empty annotation set')
    accuracy = sum(confusion[a][a] for a in LABELS)/n
    expected = sum(sum(confusion[a].values())*sum(confusion[b][a] for b in LABELS) for a in LABELS)/n**2
    tp = confusion['No answer']['No answer']
    predicted_no_answer = sum(confusion[a]['No answer'] for a in LABELS)
    true_no_answer = sum(confusion['No answer'].values())
    return {'items': n, 'agreement': accuracy,
            'cohen_kappa': (accuracy-expected)/(1-expected) if expected < 1 else None,
            'confusion_human_rows': confusion,
            'no_answer_precision': tp/predicted_no_answer if predicted_no_answer else None,
            'no_answer_recall': tp/true_no_answer if true_no_answer else None,
            'false_no_answers': predicted_no_answer-tp, 'missed_no_answers': true_no_answer-tp}


def import_annotations(directory, annotations, partition='evaluation'):
    """Validate and preserve the human export without relabeling any responses."""
    from collections import Counter
    import shutil
    rows = read_rows(directory/'items.jsonl')
    manifest = json.loads((directory/'manifest.json').read_text())
    if digest(rows) != manifest['items_sha256']:
        raise ValueError('Frozen audit items changed')
    items = [r for r in rows if r['partition'] == partition]
    labels = read_labels(annotations, items)
    destination = directory/f'{partition}-annotations.csv'
    if destination.exists() and file_hash(destination) != file_hash(annotations):
        raise ValueError('Different annotation export already imported; preserve it before replacing')
    if annotations.resolve() != destination.resolve():
        shutil.copyfile(annotations, destination)
    write(directory/f'{partition}-human-labels.jsonl', [
        {'item_id': r['item_id'], 'label': labels[r['item_id']]} for r in items], rows=True)
    summary = {'items': len(labels), 'labels': dict(Counter(labels.values())),
               'annotations_sha256': file_hash(destination), 'items_sha256': manifest['items_sha256']}
    write(directory/f'{partition}-human-summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


def import_combined_annotations(directories, annotations, partition='evaluation'):
    """Import one Label Studio export spanning multiple separately sampled cohorts."""
    from collections import Counter
    import shutil
    if len({p.resolve() for p in directories}) != len(directories):
        raise ValueError('Study directories must be distinct')
    if len({p.parent.resolve() for p in directories}) != 1:
        raise ValueError('Combined studies must share a parent directory')
    cohorts = {}
    for directory in directories:
        rows = read_rows(directory/'items.jsonl')
        manifest = json.loads((directory/'manifest.json').read_text())
        if digest(rows) != manifest['items_sha256']:
            raise ValueError('Frozen audit items changed')
        cohorts[directory] = [r for r in rows if r['partition'] == partition]
    items = [r for rows in cohorts.values() for r in rows]
    if len({r['item_id'] for r in items}) != len(items):
        raise ValueError('Study task IDs overlap')
    labels = read_labels(annotations, items)
    destination = directories[0].parent/'annotations.csv'
    if destination.exists() and file_hash(destination) != file_hash(annotations):
        raise ValueError('Different combined export already imported; preserve it before replacing')
    if destination.resolve() != annotations.resolve():
        shutil.copyfile(annotations, destination)
    summaries = {}
    for directory, rows in cohorts.items():
        write(directory/f'{partition}-human-labels.jsonl', [
            {'item_id': r['item_id'], 'label': labels[r['item_id']]} for r in rows], rows=True)
        summary = {'items': len(rows), 'labels': dict(Counter(labels[r['item_id']] for r in rows)),
                   'annotations_sha256': file_hash(destination), 'annotations_file': '../annotations.csv',
                   'items_sha256': digest(read_rows(directory/'items.jsonl'))}
        write(directory/f'{partition}-human-summary.json', summary)
        summaries[directory.name] = summary
    print(json.dumps(summaries, indent=2))
    return summaries


def evaluate(directory, annotations, predictions, partition, output):
    from parsing.temporal import is_idk_response
    from .judge import make_job
    all_items = read_rows(directory/'items.jsonl')
    if digest(all_items) != json.loads((directory/'manifest.json').read_text())['items_sha256']:
        raise ValueError('Frozen audit items changed')
    items = [r for r in all_items if r['partition'] == partition]
    human = read_labels(annotations, items)
    judged = {}
    for row in read_rows(predictions):
        if row['item_id'] in judged:
            raise ValueError('Duplicate judged audit item')
        judged[row['item_id']] = row
    selected = []
    summary_path = predictions.parent/'summary.json'
    judge_summary = json.loads(summary_path.read_text())
    prompt = (ROOT/'prompt.txt').read_text()
    if judge_summary['prompt_sha256'] != digest(prompt):
        raise ValueError('Prompt changed since judging; score the current prompt before validating')
    for r in items:
        j = judged.get(r['item_id'])
        if j is None or any(j[k] != r[k] for k in ('question', 'answer')):
            raise ValueError('Missing or mismatched judged audit item')
        if make_job(r, prompt, judge_summary['model'])[0] != j['request_key']:
            raise ValueError('Prediction is from another prompt/model')
        selected.append({**r, 'human_label': human[r['item_id']], 'judge_label': j['judge']['label'],
                         'judge_reason': j['judge']['reason'],
                         'canonical_label': 'No answer' if is_idk_response(r['answer']) else 'Answer'})
    def summarize(group):
        y = [r['human_label'] for r in group]
        return {'judge': metrics(y, [r['judge_label'] for r in group]),
                'canonical_detector': metrics(y, [r['canonical_label'] for r in group])}
    report = {'partition': partition, 'prompt_sha256': digest(prompt), 'judge_model': judge_summary['model'],
              'scope': 'full_answer', 'overall': summarize(selected),
              'by_cohort': {c: summarize([r for r in selected if r['cohort'] == c])
                            for c in sorted({r['cohort'] for r in selected})},
              'by_model': {m: summarize([r for r in selected if r['model'] == m])
                           for m in sorted({r['model'] for r in selected})},
              'note': 'Balanced audit of routed failures; not population-level no-answer prevalence.'}
    write(output/'summary.json', report)
    lines = [f'Answer-status audit: {partition}',
             'Human labels indicate answer presence, independently of correctness.', '']
    for cohort, results in [('overall', report['overall']), *report['by_cohort'].items()]:
        for method, values in results.items():
            kappa = f'{values["cohen_kappa"]:.3f}' if values['cohen_kappa'] is not None else 'not defined'
            lines.append(f'{cohort} / {method}: n={values["items"]}, agreement={100*values["agreement"]:.2f}%, '
                         f'kappa={kappa}, false no-answer labels={values["false_no_answers"]}, '
                         f'missed no-answer labels={values["missed_no_answers"]}')
    (output/'summary.txt').write_text('\n'.join(lines)+'\n')
    write(output/'decisions.jsonl', selected, rows=True)
    write(output/'disagreements.jsonl', [r for r in selected if r['human_label'] != r['judge_label']], rows=True)
    print(json.dumps(report['overall'], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'import', 'evaluate'))
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--study-dir', type=Path, default=ROOT/'human_validation/routed-40')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=40)
    parser.add_argument('--challenge', action='store_true',
                        help='Target abstention wording (75%%) and uncertainty boundary cases (25%%)')
    parser.add_argument('--exclude-items', type=Path, nargs='+', default=[],
                        help='Existing audit items JSONL files to exclude by question/response')
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--additional-study-dir', type=Path, nargs='+', default=[],
                        help='Import a combined CSV spanning this study and additional sibling studies')
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--partition', choices=('dev', 'evaluation'), default='evaluation')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.results_dir, args.study_dir, args.seed, args.count,
                challenge=args.challenge, exclude_items=args.exclude_items)
    elif args.command == 'import':
        if not args.annotations:
            parser.error('import requires --annotations')
        if args.additional_study_dir:
            import_combined_annotations([args.study_dir, *args.additional_study_dir],
                                        args.annotations, args.partition)
        else:
            import_annotations(args.study_dir, args.annotations, args.partition)
    else:
        if not args.annotations or not args.predictions:
            parser.error('evaluate requires --annotations and --predictions')
        evaluate(args.study_dir, args.annotations, args.predictions, args.partition,
                 args.output_dir or args.study_dir/'results'/args.partition)


if __name__ == '__main__':
    main()
