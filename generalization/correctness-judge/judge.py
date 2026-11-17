"""Judge Granola generations and validate against the human correctness study."""
import argparse
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'generalization.correctness-judge'


ROOT = Path(__file__).resolve().parent
from generalization.benchmark_data import load_cases
from .cache import SQLiteJudgmentCache, content_key
from shared.io import read_rows, write
from generalization.generate import validate_records
from .metrics import summary, score_pairs, split_pairs

import re

def parse_correctness_label(text: str) -> str:
    normalized = " ".join(
        re.sub(r"[`*_#.:]+", " ", text).casefold().split()
    )
    aliases = {
        "correct": "Correct",
        "incorrect": "Incorrect",
        "abstain": "Abstain",
        "abstains": "Abstain",
        "cannot determine": "Cannot determine",
    }
    # Preserve the raw response, but tolerate an explicit initial label followed
    # by a sentence. Never choose among competing labels in the explanation.
    if normalized not in aliases:
        leading = re.match(r"^\s*(Correct|Incorrect|Abstain|Cannot determine)\.\s+(\S.*)$", text, re.I | re.S)
        if leading and not re.search(r"\b(correct|incorrect|abstain|cannot determine)\b", leading.group(2), re.I):
            return aliases[leading.group(1).casefold()]
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(
            f"correctness judge returned an invalid label: {text!r}"
        ) from error


def configuration():
    settings = json.loads((ROOT/'settings.json').read_text())
    prompt = (ROOT/'prompt.txt').read_text()
    if hashlib.sha256(prompt.encode()).hexdigest() != settings['prompt_sha256']:
        raise ValueError('Judge prompt differs from settings.json')
    identity = {'model': settings['model'], 'prompt_sha256': settings['prompt_sha256'],
                'cache_namespace': settings['backend']}
    return settings, prompt, identity


@contextmanager
def writer_lock(path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def judge_rows(rows, cache_path, *, allow_api=False, workers=8, client=None, dry_run=False):
    """Only question, accepted hierarchy and ONE response enter each request."""
    settings, template, identity = configuration()
    jobs, positions = {}, []
    if workers < 1:
        raise ValueError('Positive worker count required')
    for row in rows:
        if row.get('dataset', 'granola') != 'granola':
            raise ValueError('The validated judge applies to Granola only')
        refs = row['reference_answers']
        if not isinstance(refs, list) or not refs or any(not isinstance(a, str) or not a.strip() for a in refs):
            raise ValueError('Nonempty accepted-answer hierarchy required')
        for variant in ('base', 'finetuned'):
            response = row[f'{variant}_response']
            if any(not isinstance(s, str) or not s.strip() for s in (row['question'], response)):
                raise ValueError('Question and exact nonempty answer text required')
            key = content_key(**identity, question=row['question'], reference_answers=refs, response=response)
            jobs[key] = {'question': row['question'], 'reference_answers': refs, 'response': response}
            positions.append((key, variant))
    lock = writer_lock(cache_path) if allow_api and not dry_run else nullcontext()
    with lock:
        cache = SQLiteJudgmentCache(cache_path, identity)
        try:
            missing = {k: r for k, r in jobs.items() if k not in cache}
            counts = {'pairs': len(rows), 'unique_judgments': len(jobs),
                      'cache_hits': len(jobs)-len(missing), 'missing': len(missing)}
            print(json.dumps(counts))
            if dry_run:
                return [], counts
            if missing and not allow_api:
                raise ValueError(f'{len(missing)} judgments missing; no API calls made. Use --allow-api to permit requests.')
            if missing:
                if client is None:
                    from dotenv import load_dotenv
                    from openai import OpenAI
                    load_dotenv(ROOT.parents[1]/'.env')
                    client = OpenAI(base_url=settings['backend'], timeout=120)
                def request(job):
                    prompt = template.format(question=job['question'],
                        reference_answers=json.dumps(job['reference_answers'], ensure_ascii=False),
                        response=job['response'])
                    for attempt in range(3):
                        response = client.responses.create(model=settings['model'], input=prompt)
                        if getattr(response, 'status', 'completed') != 'completed':
                            raise ValueError('Judge response did not complete')
                        raw = response.output_text.strip()
                        try:
                            return parse_correctness_label(raw), raw
                        except ValueError:
                            if attempt == 2:
                                raise
                from tqdm import tqdm
                failures = []
                with ThreadPoolExecutor(max_workers=min(workers, len(missing))) as pool:
                    futures = {pool.submit(request, job): (key, job) for key, job in missing.items()}
                    for future in tqdm(as_completed(futures), total=len(futures), desc='Correctness judge'):
                        key, job = futures[future]
                        try:
                            label, raw = future.result()
                            cache.put_many([{**identity, **job, 'key': key, 'label': label, 'raw_output': raw}])
                        except Exception as error:
                            failures.append((key, type(error).__name__))
                if failures:
                    raise RuntimeError(f'{len(failures)} requests failed. Valid judgments were saved; rerun to retry missing requests.')
            result = []
            for index, row in enumerate(rows):
                judged = dict(row)
                for key, variant in positions[2*index:2*index+2]:
                    record = cache[key]
                    judged[f'judge_{variant}_correct'] = record['label']
                    judged[f'judge_{variant}_raw_output'] = record['raw_output']
                result.append(judged)
            return result, counts
        finally:
            cache.close()


def generation_pairs(models, scopes, directory):
    cases = load_cases('granola')
    result = []
    for model in models:
        records = read_rows(directory/'granola'/f'{model}.jsonl')
        validate_records(records, cases, complete=True)
        index = {(str(r['id']), r['variant']): r for r in records}
        for scope in scopes:
            field = {'first-sentence': 'first_sentence', 'full-answer': 'answer'}[scope]
            for case in cases:
                result.append({'model': model, 'dataset': 'granola', 'source_id': str(case['id']),
                    'qid': 'granola:'+case['qid'], 'question': case['question'], 'relation': case['relation'],
                    'answer_scope': scope, 'reference_answers': case['reference_answers'],
                    **{f'{v}_response': index[(str(case['id']), v)][field] for v in ('base', 'finetuned')}})
    return result


def human_pairs(study_dir):
    human_study = import_module('generalization.human-study.human_study')
    selected = human_study.load_study(study_dir)
    cases = load_cases('granola')
    tasks, keys = human_study.correctness_tasks(selected, cases)
    labels = human_study.read_correctness(tasks, keys, study_dir/'annotations/correctness.csv')
    gold = {(r['pair_id'], r['variant']): r for r in labels}
    references = {r['id']: r for r in cases}
    rows = []
    for pair in selected:
        if pair['dataset'] != 'granola':
            continue
        case = references[pair['source_id']]
        rows.append({'pair_id': pair['pair_id'], 'model': pair['model'], 'dataset': 'granola',
            'qid': 'granola:'+case['qid'], 'question': pair['question'],
            'reference_answers': case['reference_answers'],
            **{f'{v}_response': pair[v]['text'] for v in ('base', 'finetuned')},
            **{f'human_reference_{v}': gold[(pair['pair_id'], v)]['reference_correctness'] for v in ('base', 'finetuned')},
            **{f'human_final_{v}': gold[(pair['pair_id'], v)]['correctness'] for v in ('base', 'finetuned')}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    production = commands.add_parser('score', help='Judge saved Granola generations')
    production.add_argument('--generations-dir', type=Path, default=ROOT.parent/'generations')
    production.add_argument('--models', nargs='+', choices=('lfm', 'qwen', 'gemma', 'ministral'),
                            default=['lfm', 'qwen', 'gemma', 'ministral'])
    production.add_argument('--scopes', nargs='+', choices=('first-sentence', 'full-answer'),
                            default=['first-sentence', 'full-answer'])
    validation = commands.add_parser('validate', help='Compare the judge with human correctness labels')
    validation.add_argument('--study-dir', type=Path, default=ROOT.parent/'human-study')
    validation.add_argument('--partition', choices=('dev', 'evaluation'), default='evaluation')
    for command in (production, validation):
        command.add_argument('--cache', type=Path, default=ROOT/'cache/judgments.sqlite3')
        command.add_argument('--output-dir', type=Path, default=ROOT/'results')
        command.add_argument('--allow-api', action='store_true', help='Permit paid calls for uncached answers')
        command.add_argument('--workers', type=int, default=8)
        command.add_argument('--dry-run', action='store_true', help='Count cached/missing judgments; write nothing')
        command.add_argument('--bootstrap', type=int, default=1000 if command is production else 2000)
        command.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.workers < 1:
        parser.error('Positive bootstrap and worker counts required')
    if args.command == 'score':
        if len(set(args.models)) != len(args.models) or len(set(args.scopes)) != len(args.scopes):
            parser.error('Choose distinct models and scopes')
        rows = generation_pairs(args.models, args.scopes, args.generations_dir)
    else:
        dev, evaluation = split_pairs(human_pairs(args.study_dir), 5, args.seed)
        rows = dev if args.partition == 'dev' else evaluation
    judged, counts = judge_rows(rows, args.cache, allow_api=args.allow_api, workers=args.workers, dry_run=args.dry_run)
    if args.dry_run:
        return
    settings, _, _ = configuration()
    if args.command == 'score':
        for scope in args.scopes:
            group = [r for r in judged if r['answer_scope'] == scope]
            report = {'judge': settings, 'answer_scope': scope,
                'overall': summary(group, args.bootstrap, args.seed),
                'by_model': {m: summary([r for r in group if r['model'] == m], args.bootstrap, args.seed) for m in args.models},
                'by_relation': {rel: summary([r for r in group if r['relation'] == rel], args.bootstrap, args.seed)
                                for rel in sorted({r['relation'] for r in group})}}
            for model in args.models:
                write(args.output_dir/scope/f'{model}.jsonl', [r for r in group if r['model'] == model], rows=True)
            write(args.output_dir/scope/'summary.json', report)
            print(f'{scope}: results written to {args.output_dir/scope}')
    else:
        report = {'judge': settings, 'partition': args.partition, 'pairs': len(judged), 'answers': 2*len(judged),
            'cache': counts,
            'metrics': {field: score_pairs(judged, args.bootstrap, args.seed, gold_field=field)
                        for field in ('reference_correctness', 'correctness')}}
        output = args.output_dir/'validation'/args.partition
        write(output/'predictions.jsonl', judged, rows=True)
        write(output/'summary.json', report)
        print(json.dumps({field: {k: metric[k] for k in ('exact_agreement', 'cohen_kappa')}
                          for field, metric in report['metrics'].items()}, indent=2))


if __name__ == '__main__':
    main()

