"""Matched LFM consensus versus rewritten-greedy self-training data."""
import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace

from shared.io import digest, file_hash, read_rows, write
from parsing.location.rewriter.inference import SETTINGS, format_prompt, validate
from parsing.temporal.response import is_idk_response
from parsing.temporal.values import FIVE_LEVEL_TEMPORAL_LEVELS
from .preparation import build
from .preparation.targets import RELATIONS, evidence
from .preparation.realize import IDK_ANSWER, balanced_markdown, location_eligible, temporal_example

REPO = Path(__file__).resolve().parents[2]
PARENT = REPO / 'data_preparation/vital_persons/datasets/lfm'
MODEL = 'LFM2_5_1_2B'
MODEL_ID = 'LiquidAI/LFM2.5-1.2B-Instruct'
REVISION = '0f604ada3f766f9f257460c4c9f0b5d6f69d431b'
VERSION = 'rewritten-greedy-vital-control-v1'


def read(path):
    return json.loads(path.read_text())


def parent_identity(parent):
    return {name: file_hash(parent / name) for name in
            ('manifest.json', 'train.jsonl', 'dev.jsonl', 'test.jsonl', 'case_targets.jsonl')}


def selected_cases(parent):
    """First stable realization per semantic TRAIN case; never select by gold."""
    from training.schemas import TrainingExample
    targets = read_rows(parent / 'case_targets.jsonl')
    membership = {}
    for row in targets:
        if membership.setdefault(row['qid'], row['split']) != row['split']:
            raise ValueError('Parent person appears in multiple splits')
    chosen = {}
    for row in sorted(read_rows(parent / 'train.jsonl'), key=lambda r: r['case_id']):
        TrainingExample.model_validate(row)
        if row['source_model'] != MODEL or row['split'] != 'train' or membership.get(row['qid']) != 'train' or row['relation'] not in RELATIONS:
            raise ValueError('Requires LFM Vital Persons train cases, not dev/test or Granola')
        chosen.setdefault((row['source_model'], row['qid'], row['relation']), row)
    return sorted(chosen.values(), key=lambda r: r['case_id'])


def check_generation(args):
    manifest = read(args.work_dir / 'generation_manifest.json')
    if manifest['identity']['parent_files'] != parent_identity(args.parent_data_dir):
        raise ValueError('Parent dataset changed; use a new work directory')
    if not manifest['complete']:
        raise ValueError('Generation incomplete; resume generate first')
    for name, expected in manifest['files'].items():
        if file_hash(args.work_dir / name) != expected:
            raise ValueError(f'Frozen generation file changed: {name}')
    return manifest


def generate(args, *, generator=None):
    selected = selected_cases(args.parent_data_dir)
    names = {row['qid']: row['name'] for row in read_rows(args.metadata)}
    requests = [{'case_id': f"{MODEL}:{r['qid']}:{r['relation']}:0", 'parent_case_id': r['case_id'],
                 'qid': r['qid'], 'source_relation': r['relation'], 'prompt': r['prompt'],
                 'prompt_template_id': r['prompt_template_id'], 'name': names[r['qid']]} for r in selected]
    identity = {'version': VERSION, 'parent_files': parent_identity(args.parent_data_dir),
                'requests_sha256': digest(requests), 'model_id': MODEL_ID, 'revision': args.revision,
                'do_sample': False, 'num_beams': 1, 'max_new_tokens': args.max_new_tokens,
                'dtype': args.dtype, 'batch_size': args.batch_size, 'attention': args.attention_implementation,
                'seed': args.seed, 'system_prompt': None, 'training_only': True}
    path = args.work_dir / 'generation_manifest.json'
    if path.exists():
        previous = read(path)
        if previous['identity'] != identity:
            raise ValueError('Generation settings/inputs changed; use a new work directory')
        if previous['complete']:
            check_generation(args)
            return {'questions': len(requests), 'status': 'already complete'}
    elif args.work_dir.exists():
        raise ValueError('Existing work directory has no generation manifest')
    args.work_dir.mkdir(parents=True, exist_ok=True)
    write(path, {'identity': identity, 'complete': False})
    write(args.work_dir / 'requests.json', requests)
    cache = args.work_dir / 'generations.jsonl'
    saved = read_rows(cache) if cache.exists() else []
    answers = {row['case_id']: row for row in saved}
    if len(answers) != len(saved) or answers.keys() - {r['case_id'] for r in requests}:
        raise ValueError('Duplicate or unexpected cached generation')
    pending = [r for r in requests if r['case_id'] not in answers]
    if pending and generator is None:
        from transformers import set_seed
        from shared.generation import generate_greedy
        from parsing.location._inference import load_base
        set_seed(args.seed)
        model, tokenizer = load_base({'base_model': MODEL_ID, 'base_revision': args.revision},
            device=args.device, dtype=args.dtype, attention_implementation=args.attention_implementation)
        resolved = getattr(model.config, '_commit_hash', None)
        if resolved is not None and resolved != args.revision:
            raise ValueError('Resolved LFM revision differs from requested checkpoint')
        tokenizer.padding_side = 'left'
        model.generation_config.num_beams = 1
        generator = lambda prompts: generate_greedy(model, tokenizer, prompts,
            batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    from tqdm.auto import tqdm
    for start in tqdm(range(0, len(pending), args.batch_size), desc='Greedy train targets', unit='batch'):
        batch = pending[start:start + args.batch_size]
        results = generator([r['prompt'] for r in batch])
        if len(results) != len(batch):
            raise ValueError('Generation count mismatch')
        for request, result in zip(batch, results, strict=True):
            payload = asdict(result) if is_dataclass(result) else result
            if not isinstance(payload.get('answer'), str):
                raise ValueError('Generator must return a string answer')
            answers[request['case_id']] = {**payload, 'case_id': request['case_id']}
        write(cache, answers.values(), rows=True)
    inputs = []
    for request in requests:
        answer = answers[request['case_id']]['answer']
        if not answer.strip():
            continue  # Explicit paired rejection during planning, never IDK.
        born = request['source_relation'] in ('birth', 'birthplace')
        question = f"Where {'was' if born else 'did'} {request['name']} {'born' if born else 'die'}?"
        inputs.append({**request, 'source_model': MODEL, 'sample_index': 0, 'question': question,
                       'generation_question': request['prompt'], 'answer': answer})
    write(args.work_dir / 'input.jsonl', inputs, rows=True)
    write(path, {'identity': identity, 'complete': True, 'files': {
        name: file_hash(args.work_dir / name) for name in ('requests.json', 'generations.jsonl', 'input.jsonl')}})
    return {'questions': len(requests), 'empty_answers': len(requests) - len(inputs)}


def preprocess(args):
    """Existing helper CLIs, same order: extraction, safety decisions, normalization."""
    check_generation(args)
    if args.normalization_cache is None:
        raise ValueError('Provide --normalization-cache')
    runtime = ['--batch-size', str(args.batch_size), '--device', args.device, '--dtype', args.dtype,
               '--attention-implementation', args.attention_implementation]
    stages = [
        ('extracted.jsonl', 'input.jsonl', ['-m', 'parsing.location.extractor', *runtime]),
        ('filtered.jsonl', 'extracted.jsonl', ['-m', 'parsing.location.filter', *runtime]),
        ('processed.jsonl', 'filtered.jsonl', ['-m', 'parsing.location.normalization', 'normalize',
            '--cache-db', str(args.normalization_cache), '--workers', str(args.workers)]
            + (['--allow-network', '--user-agent', args.user_agent] if args.allow_network else [])),
    ]
    for output, source, command in stages:
        destination = args.work_dir / output
        metadata = destination.with_suffix('.run.json')
        identity = {'input_sha256': file_hash(args.work_dir / source), 'command': command,
                    'helper_configs': {name: file_hash(REPO / f'parsing/location/{name}/model.json')
                                       for name in ('extractor', 'filter', 'rewriter')}}
        if metadata.exists():
            saved = read(metadata)
            if saved['identity'] != identity or file_hash(destination) != saved['output_sha256']:
                raise ValueError('Preprocessing inputs/settings/output changed')
            continue
        if destination.exists():
            raise ValueError(f'Untracked preprocessing output: {destination}')
        with tempfile.TemporaryDirectory(dir=args.work_dir) as temporary:
            staged = Path(temporary) / output
            subprocess.run([sys.executable, *command, '--input', str(args.work_dir / source), '--output', str(staged)], check=True)
            staged.rename(destination)
        write(metadata, {'identity': identity, 'output_sha256': file_hash(destination)})
    return {'processed': str(args.work_dir / 'processed.jsonl')}


def own_temporal(row):
    value = evidence(row, row['source_relation'])
    if value is None:
        raise ValueError('No event-anchored temporal value; not IDK')
    allowed = []
    for level in FIVE_LEVEL_TEMPORAL_LEVELS:
        try:
            allowed.append(value.to_level(level))
        except ValueError:
            pass
    if not allowed:
        raise ValueError('No supported five-level temporal projection')
    return max(allowed, key=lambda v: v.precision)


def own_location(row):
    reason = location_eligible(row)
    value = evidence(row, row['source_relation'])
    if reason or value is None:
        raise ValueError(f'No safe within-answer location: {reason}; not IDK')
    return value


def plan(args, *, replay_recorded=False):
    generated = check_generation(args)
    identity = {'version': VERSION, 'parent_files': parent_identity(args.parent_data_dir),
                'generation': digest(generated), 'processed': file_hash(args.work_dir / 'processed.jsonl'),
                'rewriter': SETTINGS, 'seed': args.seed,
                'runtime': {k: getattr(args, k) for k in ('device', 'dtype', 'attention_implementation', 'batch_size')},
                'code_sha256': file_hash(Path(__file__))}
    directory = args.work_dir / 'rewrite'
    if directory.exists():
        manifest = read(directory / 'manifest.json')
        if replay_recorded:
            # Existing rewrite results retain the model identity that produced them.
            # Adapter hosting changes do not alter those recorded results.
            identity['rewriter'] = manifest['identity']['rewriter']
        # The version, frozen inputs and runtime define replay compatibility.
        # A whole-file source hash is provenance: renaming a default metadata
        # path must not invalidate the already frozen rewrite tasks.
        replay = lambda value: {k:v for k,v in value.items() if k != 'code_sha256'}
        if replay(manifest['identity']) != replay(identity):
            raise ValueError('Rewrite inputs/settings changed')
        for name, expected in manifest['files'].items():
            if file_hash(directory / name) != expected:
                raise ValueError('Frozen rewrite plan changed')
        return manifest
    parents = {r['case_id']: r for r in selected_cases(args.parent_data_dir)}
    requests = read(args.work_dir / 'requests.json')
    answers = {r['case_id']: r['answer'] for r in read_rows(args.work_dir / 'generations.jsonl')}
    inputs = {r['case_id']: r for r in read_rows(args.work_dir / 'input.jsonl')}
    rows = read_rows(args.work_dir / 'processed.jsonl')
    processed = {r['case_id']: r for r in rows}
    if len(processed) != len(rows) or processed.keys() - answers.keys():
        raise ValueError('Duplicate or unexpected processed answer')
    entries, tasks, rejected = [], {}, []
    for request in requests:
        source = parents[request['parent_case_id']]
        answer = answers[request['case_id']]
        control = {**source, 'case_id': request['case_id'], 'source_answer': answer, 'contrastive_groups': []}
        try:
            if not answer.strip():
                raise ValueError('Empty generation; not IDK')
            if is_idk_response(answer):
                control.update(positive_answer=IDK_ANSWER, is_idk=True, primary_target_level=None, secondary_target_level=None)
                entries.append({'request': request, 'control': control})
                continue
            row = processed.get(request['case_id'])
            if row is None or row['answer'] != answer or any(row[k] != inputs[request['case_id']][k]
                for k in ('qid', 'source_relation', 'question', 'source_model')):
                raise ValueError('Missing/stale processed greedy answer')
            value = own_temporal(row) if row['source_relation'] in ('birth', 'death') else own_location(row)
            control.update(is_idk=False, primary_target_level=value.level.value, secondary_target_level=None)
            entry = {'request': request, 'control': control}
            if row['source_relation'] in ('birth', 'death'):
                control['positive_answer'] = temporal_example(row, value, 'train', request['name'], args.seed)['positive_answer']
            else:
                task = {'question': row['question'], 'answer': answer,
                        'original_location': row['extraction']['location'],
                        'original_components': row['extraction']['components'], 'normalized_value': asdict(value),
                        'target_level': value.level.value, 'replacement_location': value.to_text()}
                key = digest({'rewriter': SETTINGS, 'prompt': format_prompt(task)})
                tasks[key] = task
                entry['cache_key'] = key
            entries.append(entry)
        except ValueError as error:
            rejected.append({**request, 'reason': str(error), 'original_answer': answer})
    with tempfile.TemporaryDirectory(dir=args.work_dir) as temporary:
        stage = Path(temporary) / 'rewrite'
        stage.mkdir()
        for filename, payload in (('entries.json', entries), ('tasks.json', tasks), ('paired_rejections.json', rejected)):
            write(stage / filename, payload)
        manifest = {'identity': identity, 'signature': digest(identity), 'unique_rewrite_tasks': len(tasks),
                    'files': {p.name: file_hash(p) for p in stage.iterdir()}}
        write(stage / 'manifest.json', manifest)
        stage.rename(directory)
    return manifest


def assemble(args):
    from training.schemas import TrainingExample
    manifest = plan(args, replay_recorded=True)
    directory = args.work_dir / 'rewrite'
    entries, tasks = read(directory / 'entries.json'), read(directory / 'tasks.json')
    results = {}
    if (directory / 'rewrites.sqlite3').exists():
        with closing(sqlite3.connect((directory / 'rewrites.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as cache:
            results = {key: json.loads(value) for key, value in cache.execute('SELECT cache_key,result FROM rewrites')}
    if tasks.keys() - results.keys():
        raise ValueError('Rewrites incomplete; run rewrite first')
    parents = {r['case_id']: r for r in selected_cases(args.parent_data_dir)}
    greedy, consensus = [], []
    rejected = read(directory / 'paired_rejections.json')
    for entry in entries:
        row = dict(entry['control'])
        key = entry.get('cache_key')
        if key:
            result = results[key]
            text = result['prediction']
            if result['status'] != 'ok' or validate(tasks[key], text) or not balanced_markdown(text):
                rejected.append({**entry['request'], 'reason': 'Invalid greedy rewrite'})
                continue
            row['positive_answer'] = text
        TrainingExample.model_validate(row)
        greedy.append(row)
        consensus.append({**parents[entry['request']['parent_case_id']], 'contrastive_groups': []})
    if not greedy:
        raise ValueError('No shared train cases survived')
    assert [(r['qid'], r['relation'], r['prompt']) for r in greedy] == [(r['qid'], r['relation'], r['prompt']) for r in consensus]
    targets = read_rows(args.parent_data_dir / 'case_targets.jsonl')
    own = {(r['qid'], r['relation']): r for r in greedy}
    greedy_targets = []
    for target in targets:
        row = own.get((target['qid'], target['relation'])) if target['split'] == 'train' else None
        greedy_targets.append({**target, 'target_status': 'idk' if row['is_idk'] else 'known',
            'target_level': row['primary_target_level'], 'sample_count': 1, 'required_primary_values': 1,
            'available_primary_values': 1} if row else target)
    report = {'version': VERSION, 'signature': digest({'plan': manifest['signature'], 'results': results}),
              'shared_train_cases': len(greedy), 'paired_rejections': len(rejected), 'granola': False,
              'greedy_idk': sum(r['is_idk'] for r in greedy), 'consensus_idk': sum(r['is_idk'] for r in consensus),
              'checkpoint_selection': 'identical original Vital Persons dev examples and targets',
              'weight_policy': 'IDK SFT weight 0.5 in both arms; different IDK labels can change effective weights'}
    if args.output_dir.exists():
        if read(args.output_dir / 'comparison_design.json') != report:
            raise ValueError('Output belongs to a different control; use a new directory')
        for arm in ('consensus', 'greedy'):
            saved = read(args.output_dir / arm / 'manifest.json')
            for item in saved['output_files']:
                if file_hash(args.output_dir / arm / item['path']) != item['sha256']:
                    raise ValueError('Existing dataset changed')
        return report
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temporary:
        stage = Path(temporary) / 'package'
        stage.mkdir()
        for arm, train, labels in (('consensus', consensus, targets), ('greedy', greedy, greedy_targets)):
            output = stage / arm
            write(output / 'train.jsonl', train, rows=True)
            for split in ('dev', 'test'):
                shutil.copy2(args.parent_data_dir / f'{split}.jsonl', output / f'{split}.jsonl')
            write(output / 'case_targets.jsonl', labels, rows=True)
            write(output / 'paired_rejections.json', rejected)
            write(output / 'manifest.json', {'config': {'source_model': MODEL, 'greedy_control_version': VERSION,
                'greedy_control_arm': arm, 'granola': False, 'training_objective': 'sft-only',
                'parent_files': parent_identity(args.parent_data_dir),
                'case_target_train_policy': 'single-greedy-answer' if arm == 'greedy' else 'frozen-parent-consensus'},
                'accepted_by_split': {'train': len(train), 'dev': len(read_rows(output / 'dev.jsonl'))},
                'train_by_relation': dict(Counter(r['relation'] for r in train)),
                'output_files': [{'path': p.name, 'sha256': file_hash(p)} for p in sorted(output.iterdir())]})
        write(stage / 'comparison_design.json', report)
        stage.rename(args.output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'preprocess', 'plan', 'rewrite', 'assemble'))
    parser.add_argument('--parent-data-dir', type=Path, default=PARENT)
    parser.add_argument('--metadata', type=Path, default=REPO / 'data_preparation/vital_persons/metadata/gold_adapted.jsonl')
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--normalization-cache', type=Path)
    parser.add_argument('--allow-network', action='store_true')
    parser.add_argument('--user-agent')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    parser.add_argument('--attention-implementation', default='sdpa')
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.max_new_tokens) < 1 or args.seed < 0:
        parser.error('Batch size/token limits/workers must be positive and seed nonnegative')
    if args.command == 'assemble' and args.output_dir is None:
        parser.error('assemble requires --output-dir')
    if args.allow_network and not args.user_agent:
        parser.error('--allow-network requires a contact --user-agent')
    import fcntl
    args.work_dir.parent.mkdir(parents=True, exist_ok=True)
    with (args.work_dir.parent / f'.{args.work_dir.name}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process is using this work directory') from None
        if args.command == 'generate':
            result = generate(args)
        elif args.command == 'preprocess':
            result = preprocess(args)
        elif args.command == 'plan':
            result = plan(args)
        elif args.command == 'rewrite':
            plan(args)
            runtime = SimpleNamespace(**{**vars(args), 'work_dir': args.work_dir / 'rewrite'})
            build.rewrite_tasks(runtime, read(runtime.work_dir / 'tasks.json'))
            result = {'status': 'rewritten; run assemble next'}
        else:
            result = assemble(args)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
