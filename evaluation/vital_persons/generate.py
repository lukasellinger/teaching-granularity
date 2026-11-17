"""Resumable greedy answers, with one file per modality and prompt condition."""
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

from shared.io import read_rows, write, file_hash, digest
from .cases import VITAL, REPO
from .prompts import GRANULARITY_INSTRUCTION, IDK_USER_PROMPT_TEMPLATE, build_drag_aggregation_prompt

VARIANTS = ('base_plain', 'base_idk_user_prompt', 'base_granularity_instruction', 'base_drag', 'finetuned')


def variant_prompt(variant, question, samples=None):
    if variant not in VARIANTS:
        raise ValueError('Unknown generation variant')
    if variant == 'base_drag':
        if samples is None or len(samples) != 20 or any(not isinstance(answer, str) for answer in samples):
            raise ValueError('DRAG requires the twenty stored answers, without filtering')
        return build_drag_aggregation_prompt(question, samples), None
    if variant == 'base_idk_user_prompt':
        return IDK_USER_PROMPT_TEMPLATE.format(question=question), None
    return question, GRANULARITY_INSTRUCTION if variant == 'base_granularity_instruction' else None


def drag_samples(model, cases, directory=None):
    directory = Path(directory) if directory else VITAL/'samples'/model
    pools = {}
    for relation in sorted({case.relation.value for case in cases}):
        pool = {}
        for row in read_rows(directory/f'{relation}_samples.jsonl'):
            if row['qid'] in pool:
                raise ValueError('Duplicate DRAG sample-pool person')
            if len(row['samples']) != 20 or any(not isinstance(s, str) for s in row['samples']):
                raise ValueError('DRAG sample pool must have twenty string answers per question')
            pool[row['qid']] = row['samples']
        pools[relation] = pool
    return {case.case_id: pools[case.relation.value][case.qid] for case in cases}


def validate_records(rows, cases, variant, samples=None, *, complete=False):
    expected = {case.case_id: case for case in cases}
    if not expected or len(expected) != len(cases):
        raise ValueError('Cases must be non-empty with unique IDs')
    seen = set()
    for row in rows:
        key = row['case_id']
        if key not in expected or key in seen or row['variant'] != variant:
            raise ValueError('Unexpected or duplicate generation case/variant')
        seen.add(key)
        prompt, system = variant_prompt(variant, expected[key].prompt, samples.get(key) if samples else None)
        if row.get('user_prompt') != prompt or row.get('system_prompt') != system:
            raise ValueError('Generation prompt does not match the evaluation question/condition')
        if not isinstance(row.get('answer'), str):
            raise ValueError('Generation answer must be a string')
        count = row.get('generated_tokens')
        if type(count) is not int or count < 0 or row.get('finish_reason') not in {'eos', 'max_tokens', 'stopped'}:
            raise ValueError('Generation requires a nonnegative token count and finish reason')
    if complete and seen != expected.keys():
        raise ValueError(f'Incomplete {variant} generations: {len(expected)-len(seen)} missing')
    return seen


def settings(model, *, batch_size=None, drag_batch_size=None, max_new_tokens=128, adapter=None):
    config = json.loads((REPO/'generalization/generation_settings.json').read_text())
    result = {**{k: v for k, v in config.items() if k != 'models'}, **config['models'][model]}
    result.update(max_new_tokens=max_new_tokens, batch_size=batch_size or result['batch_size'], drag_batch_size=drag_batch_size or 4)
    if adapter:
        adapter = Path(adapter).resolve()
        if (adapter/'final').is_dir():
            adapter = adapter/'final'
        if not (adapter/'adapter_config.json').is_file():
            raise ValueError('Provide a PEFT adapter directory or completed training run')
        result['local_adapter'] = str(adapter)
        files = sorted(p for p in adapter.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.jinja', '.model'})
        result['local_adapter_sha256'] = {p.name: file_hash(p) for p in files}
    return result


def load_model(config, device, with_adapter):
    from shared.models import resolve_adapter, generation_tokenizer, load_inference_model
    adapter = resolve_adapter(config, config.get('local_adapter')) if with_adapter else None
    tokenizer = generation_tokenizer(config, adapter)
    return load_inference_model(config, device, adapter), tokenizer


def run(cases_by_modality, variants, directory, config, *, device='cuda', samples_by_modality=None):
    plans = []
    for modality, cases in cases_by_modality.items():
        samples = samples_by_modality.get(modality) if samples_by_modality else None
        for variant in variants:
            path = directory/modality/f'{variant}.jsonl'
            metadata = path.with_suffix('.run.json')
            condition = [variant_prompt(variant, case.prompt, samples.get(case.case_id) if samples else None) for case in cases]
            # Gold/consensus changes do not invalidate otherwise identical generations.
            signature = {'settings': config, 'device': device, 'variant': variant,
                         'tokenizer_source': 'adapter' if 'finetuned' in variants else 'base',
                         'questions': digest([(case.case_id, pair) for case, pair in zip(cases, condition, strict=True)])}
            rows = read_rows(path) if path.exists() else []
            if metadata.exists():
                if json.loads(metadata.read_text())['signature'] != signature:
                    raise ValueError(f'{path}: generation settings changed; choose a new directory')
            elif path.exists():
                raise ValueError(f'{path}: existing generations have no run metadata')
            completed = validate_records(rows, cases, variant, samples)
            plans.append((path, metadata, signature, variant, cases, samples, rows, completed))
    # Avoid loading a tokenizer or adapter when every generation is already cached.
    if all(len(plan[-1]) == len(plan[4]) for plan in plans):
        return
    import torch
    from transformers import set_seed
    from tqdm.auto import tqdm
    from shared.generation import generate_greedy
    set_seed(config['seed'])
    model, tokenizer = load_model(config, device, with_adapter='finetuned' in variants)
    for path, metadata, signature, variant, cases, samples, rows, completed in plans:
        pending = [case for case in cases if case.case_id not in completed]
        if not pending:
            continue
        batch_size = config['drag_batch_size'] if variant == 'base_drag' else config['batch_size']
        write(metadata, {'signature': signature, 'status': 'running', 'torch': torch.__version__})
        context = model.disable_adapter() if variant != 'finetuned' and hasattr(model, 'disable_adapter') else nullcontext()
        with context, tqdm(total=len(pending), desc=f'{path.parent.name}/{variant}', unit='answer') as progress:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start+batch_size]
                prompts = [variant_prompt(variant, case.prompt, samples.get(case.case_id) if samples else None) for case in batch]
                outputs = generate_greedy(model, tokenizer, [p for p, _ in prompts], batch_size=batch_size,
                    max_new_tokens=config['max_new_tokens'], system_prompt=prompts[0][1])
                for case, (prompt, system), answer in zip(batch, prompts, outputs, strict=True):
                    rows.append({'schema_version': 2, 'variant': variant, 'case_id': case.case_id,
                                 'user_prompt': prompt, 'system_prompt': system, **asdict(answer)})
                # Save after every batch; resume never re-generates completed answers.
                write(path, sorted(rows, key=lambda row: row['case_id']), rows=True)
                progress.update(len(batch))
        validate_records(rows, cases, variant, samples, complete=True)
        write(metadata, {'signature': signature, 'status': 'complete', 'torch': torch.__version__})
