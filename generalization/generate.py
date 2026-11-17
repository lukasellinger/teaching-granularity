"""Generate paired base/FT answers and score them for the generalization study."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'generalization'

from generalization.benchmark_data import load_cases
from shared.io import read_rows, write

ROOT = Path(__file__).resolve().parent


def validate_records(records, cases, *, complete=False):
    index = {str(r['id']): r for r in cases}
    if not cases or len(index) != len(cases):
        raise ValueError('Questions must be nonempty with unique IDs')
    seen = set()
    for row in records:
        key = (row['variant'], str(row['id']))
        case = index.get(key[1])
        if key in seen or key[0] not in ('base', 'finetuned') or case is None:
            raise ValueError('Duplicate or unexpected generation')
        if any(row.get(k) != case.get(k) for k in ('question', 'qid', 'relation')):
            raise ValueError('Generation does not match its question')
        if row.get('user_prompt') != case['question'] or row.get('system_prompt') is not None:
            raise ValueError('Expected question-only prompt, no system instruction')
        seen.add(key)
    if complete and len(seen) != 2*len(cases):
        raise ValueError('Both base and FT answers required for every question')
    return seen


def load_model(settings, device, adapter_path=None):
    from shared.models import resolve_adapter, generation_tokenizer, load_inference_model
    adapter = resolve_adapter(settings, adapter_path)
    tokenizer = generation_tokenizer(settings, adapter)
    return load_inference_model(settings, device, adapter, dispatch=True), tokenizer


def generate(model, tokenizer, cases, path, settings):
    from tqdm import tqdm
    from shared.generation import generate_greedy_with_logprobs
    records = read_rows(path) if path.exists() else []
    complete = validate_records(records, cases)
    for variant in ('base', 'finetuned'):
        pending = [c for c in cases if (variant, str(c['id'])) not in complete]
        context = model.disable_adapter() if variant == 'base' else nullcontext()
        with context, tqdm(total=len(pending), desc=variant, unit='answer') as progress:
            for start in range(0, len(pending), settings['batch_size']):
                batch = pending[start:start+settings['batch_size']]
                outputs = generate_greedy_with_logprobs(model, tokenizer,
                    [c['question'] for c in batch], batch_size=settings['batch_size'],
                    max_new_tokens=settings['max_new_tokens'], system_prompt=None)
                if len(outputs) != len(batch):
                    raise ValueError('Generation output count mismatch')
                records.extend({**case, **asdict(answer), 'variant': variant,
                    'user_prompt': case['question'], 'system_prompt': None}
                    for case, answer in zip(batch, outputs, strict=True))
                records.sort(key=lambda r: (r['variant'], str(r['id'])))
                write(path, records, rows=True)
                progress.update(len(batch))
    validate_records(records, cases, complete=True)


def score(path, cases, batch_size):
    import math
    import numpy as np
    from granuscore import GranuScore
    records = read_rows(path)
    validate_records(records, cases, complete=True)
    pending = [r for r in records if 'granuscore' not in r or 'first_sentence_granuscore' not in r]
    if not pending:
        print(f'{path}: already scored')
        return
    scorer = GranuScore()
    for field, output in (('answer', 'granuscore'), ('first_sentence', 'first_sentence_granuscore')):
        nonempty = [r for r in pending if r[field].strip()]
        for row in pending:
            row[output] = None
        if nonempty:
            values = np.asarray(scorer([r[field] for r in nonempty], split=True,
                pooling='mean', pooling_scope='sentence', scope_pooling_method='lower_quantile_mean',
                return_percentiles=True, percentile_before_pooling=True, batched=True,
                batch_size=batch_size, show_progress_bar=True), dtype=float).reshape(-1)
            if len(values) != len(nonempty) or not all(math.isfinite(v) for v in values):
                raise ValueError('GranuScore returned invalid values')
            for row, value in zip(nonempty, values, strict=True):
                row[output] = float(value)
    write(path, records, rows=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('lfm', 'qwen', 'gemma', 'ministral'),
                        default=['lfm', 'qwen', 'gemma', 'ministral'])
    parser.add_argument('--datasets', nargs='+', choices=('granola', 'simpleqa', 'truthfulqa'),
                        default=['granola', 'simpleqa', 'truthfulqa'])
    parser.add_argument('--stage', choices=('generate', 'score', 'both'), default='generate')
    parser.add_argument('--settings', type=Path, default=ROOT/'generation_settings.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'generations')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, help='Override recorded per-model generation batch size')
    parser.add_argument('--score-batch-size', type=int, default=128)
    parser.add_argument('--adapter', type=Path, help='Local adapter, when running exactly one model')
    parser.add_argument('--adapter-ref', type=Path, help='Pinned model.json for a released ablation, with exactly one model')
    parser.add_argument('--limit', type=int, help='Smoke test: first N questions; use a separate output directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if (args.adapter or args.adapter_ref) and len(args.models) != 1:
        parser.error('--adapter/--adapter-ref requires exactly one --models value')
    if args.adapter and args.adapter_ref:
        parser.error('Use either --adapter or --adapter-ref')
    if any(n is not None and n < 1 for n in (args.batch_size, args.score_batch_size, args.limit)):
        parser.error('Batch sizes and limit must be positive')
    config = json.loads(args.settings.read_text())
    plans = []
    for name in args.models:
        settings = {**{k: v for k, v in config.items() if k != 'models'}, **config['models'][name]}
        if args.adapter_ref:
            from shared.models import adapter_reference
            settings = adapter_reference(settings, args.adapter_ref)
        if args.batch_size:
            settings['batch_size'] = args.batch_size
        settings['local_adapter'] = str(args.adapter.resolve()) if args.adapter else None
        if settings['do_sample'] or settings['system_prompt'] is not None or settings['user_prompt_template'] != '{question}' or settings['enable_thinking']:
            raise ValueError('This protocol uses greedy question-only generation, thinking disabled')
        for dataset in args.datasets:
            cases = load_cases(dataset)
            if args.limit:
                cases = cases[:args.limit]
            validate_records([], cases)
            path = args.output_dir/dataset/f'{name}.jsonl'
            signature = {'settings': settings, 'device': args.device,
                         'cases_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()}
            print(f'{name}/{dataset}: {len(cases)} questions, batch={settings["batch_size"]}, '
                  f'max_new_tokens={settings["max_new_tokens"]} → {path}')
            plans.append((name, cases, path, signature))
    if args.dry_run:
        return
    # Validate every output before downloading weights or modifying any files.
    for _, cases, path, signature in plans:
        if args.stage in ('generate', 'both'):
            metadata = path.with_suffix('.run.json')
            if metadata.exists():
                if json.loads(metadata.read_text())['signature'] != signature:
                    raise ValueError(f'{path}: settings changed; use a new output directory')
            elif path.exists():
                raise ValueError(f'{path}: existing answers have no matching run metadata; use a new output directory')
        if path.exists():
            validate_records(read_rows(path), cases, complete=args.stage == 'score')
        elif args.stage == 'score':
            raise ValueError(f'{path}: generate answers before scoring')
    if args.stage in ('generate', 'both'):
        import gc
        import torch
        from transformers import set_seed
        current = model = tokenizer = None
        for name, cases, path, signature in plans:
            metadata = path.with_suffix('.run.json')
            if metadata.exists():
                if json.loads(metadata.read_text())['signature'] != signature:
                    raise ValueError(f'{path}: settings changed; use a new output directory')
            elif path.exists():
                raise ValueError(f'{path}: existing answers have no matching run metadata; use a new output directory')
            if path.exists() and len(read_rows(path)) == 2*len(cases):
                print(f'{path}: generation complete')
                continue
            if name != current:
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                set_seed(signature['settings']['seed'])
                model, tokenizer = load_model(signature['settings'], args.device, args.adapter)
                current = name
            write(metadata, {'signature': signature, 'status': 'running',
                             'torch': torch.__version__, 'cuda': torch.version.cuda})
            generate(model, tokenizer, cases, path, signature['settings'])
            write(metadata, {'signature': signature, 'status': 'complete',
                             'torch': torch.__version__, 'cuda': torch.version.cuda})
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage in ('score', 'both'):
        for _, cases, path, _ in plans:
            score(path, cases, args.score_batch_size)


if __name__ == '__main__':
    main()
