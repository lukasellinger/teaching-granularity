"""Resume twenty sampled answers per question for the fixed Vital Persons cohort."""
from shared.constants import QUESTIONS as PROMPTS
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from shared.io import read_rows, write

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODELS = ('lfm', 'qwen', 'gemma', 'ministral')


def person_seed(seed, relation, qid):
    return int.from_bytes(hashlib.sha256(f'{seed}:{relation}:{qid}'.encode()).digest()[:4], 'big') % (2**31)


def validate_pool(rows, people, relation, count):
    people = {r['qid']: r for r in people}
    seen = set()
    for row in rows:
        qid = row['qid']
        if qid in seen or qid not in people:
            raise ValueError('Duplicate or unexpected cached person')
        samples = row['samples']
        if not isinstance(samples, list) or len(samples) != count or any(not isinstance(s, str) or not s.strip() for s in samples):
            raise ValueError('Cached question must contain exactly the requested number of nonempty samples')
        question = PROMPTS[relation].format(person=people[qid]['name'])
        if row.get('question', question) != question:
            raise ValueError('Cached generation question changed')
        if 'finish_reasons' in row and (len(row['finish_reasons']) != count or any(v not in {'stop', 'length'} for v in row['finish_reasons'])):
            raise ValueError('Invalid finish reasons')
        seen.add(qid)
    return seen


def decode_samples(tokenizer, sequences, width, eos_ids, max_new_tokens):
    texts, reasons = [], []
    for sequence in sequences:
        ids = sequence[width:]
        eos = next((i for i, token in enumerate(ids) if token in eos_ids), None)
        if eos is None and len(ids) != max_new_tokens:
            raise ValueError('Unexpected early generation stop')
        text = tokenizer.decode(ids if eos is None else ids[:eos], skip_special_tokens=True).strip()
        if not text:
            raise ValueError('Empty generation; refusing to silently drop a consensus sample')
        texts.append(text)
        reasons.append('stop' if eos is not None else 'length')
    return texts, reasons


def load_model(settings, device):
    import torch
    from transformers import AutoTokenizer, MistralCommonBackend
    from shared.models import causal_model_class
    if not device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('New generation requires CUDA; --dry-run checks the frozen data/cache without loading a model')
    ministral = 'Ministral-3' in settings['model_id']
    tokenizer_class = MistralCommonBackend if ministral else AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(settings['model_id'], revision=settings['revision'])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cls = causal_model_class(settings['model_id'])
    model = cls.from_pretrained(settings['model_id'], revision=settings['revision'], dtype=torch.bfloat16,
                               attn_implementation='sdpa', device_map={'': device}).eval()
    return model, tokenizer


def generate_one(model, tokenizer, person, relation, args):
    import torch
    from transformers import GenerationConfig
    question = PROMPTS[relation].format(person=person['name'])
    seed = person_seed(args.seed, relation, person['qid'])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    chat_kwargs = {} if args.model == 'ministral' else {'enable_thinking': False}
    inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
        tokenize=True, add_generation_prompt=True, return_tensors='pt', return_dict=True, **chat_kwargs)
    inputs = {k: v.to(args.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    eos = model.generation_config.eos_token_id
    eos = tokenizer.eos_token_id if eos is None else eos
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    if not eos_ids:
        raise ValueError('Missing EOS token')
    # Other sampling filters come from the
    # pinned checkpoint. The later Ministral run explicitly used unrestricted sampling.
    config = GenerationConfig(top_p=1., top_k=0, repetition_penalty=1., num_beams=1) if args.model == 'ministral' else deepcopy(model.generation_config)
    config.update(do_sample=True, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        eos_token_id=sorted(eos_ids), pad_token_id=min(eos_ids), use_cache=True,
        num_return_sequences=1)
    texts, reasons = [], []
    with torch.inference_mode():
        for start in range(0, args.samples, args.sample_batch_size):
            size = min(args.sample_batch_size, args.samples-start)
            batch = {k: v.repeat(size, *([1]*(v.ndim-1))) for k, v in inputs.items()}
            output = model.generate(**batch, generation_config=config)
            generated, finishes = decode_samples(tokenizer, output.tolist(), inputs['input_ids'].shape[1], eos_ids, args.max_new_tokens)
            texts.extend(generated)
            reasons.extend(finishes)
    return {'qid': person['qid'], 'question': question, 'samples': texts, 'finish_reasons': reasons, 'seed': seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--metadata', type=Path, default=ROOT/'metadata/gold_adapted.jsonl')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--relations', nargs='+', choices=tuple(PROMPTS), default=list(PROMPTS))
    parser.add_argument('--sample-batch-size', type=int)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--max-new-tokens', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, help='Smoke test only; choose a separate output directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.sample_batch_size is None:
        args.sample_batch_size = 16 if args.model == 'ministral' else 20
    if min(args.samples, args.sample_batch_size, args.max_new_tokens) < 1 or not math.isfinite(args.temperature) or args.temperature <= 0 or args.limit is not None and args.limit < 1:
        parser.error('Sample count, batch size, token limit, temperature and limit must be positive')
    if len(set(args.relations)) != len(args.relations):
        parser.error('Choose distinct relations')
    people = read_rows(args.metadata)
    if not people or len({r['qid'] for r in people}) != len(people) or any(not r.get('name') or not r['qid'].startswith('Q') or not r['qid'][1:].isdigit() for r in people):
        raise ValueError('Invalid frozen person metadata')
    if args.limit:
        people = people[:args.limit]
    settings = json.loads((REPO/'generalization/generation_settings.json').read_text())['models'][args.model]
    output = args.output_dir or ROOT/'samples'/args.model
    cohort = [{k: r[k] for k in ('qid', 'name')} for r in people]
    signature = {'version': 1, 'model_id': settings['model_id'], 'revision': settings['revision'],
        'cohort_sha256': hashlib.sha256(json.dumps(cohort, sort_keys=True).encode()).hexdigest(),
        'prompts': PROMPTS, 'samples': args.samples, 'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens, 'sample_batch_size': args.sample_batch_size, 'seed': args.seed,
        'dtype': 'bfloat16', 'attention': 'sdpa', 'thinking': False, 'system_prompt': None}
    pools = {}
    for relation in args.relations:
        path = output/f'{relation}_samples.jsonl'
        rows = read_rows(path) if path.exists() else []
        pools[relation] = validate_pool(rows, people, relation, args.samples)
    missing = sum(len(people)-len(ids) for ids in pools.values())
    print(f'{len(people)} people; {len(people)*len(args.relations)} questions; {missing} missing; {args.samples} samples/question')
    run = output/'run.json'
    if run.exists() and json.loads(run.read_text()) != signature:
        raise ValueError('Generation settings changed; use a new output directory')
    if args.dry_run or not missing:
        return
    if not run.exists() and any(pools.values()):
        raise ValueError('Partial imported pool has no generation settings; regenerate separately rather than mixing protocols')
    write(run, signature)
    model, tokenizer = load_model(settings, args.device)
    from tqdm import tqdm
    for relation in args.relations:
        pending = [r for r in people if r['qid'] not in pools[relation]]
        with (output/f'{relation}_samples.jsonl').open('a', encoding='utf-8') as stream:
            for person in tqdm(pending, desc=relation, unit='question'):
                row = generate_one(model, tokenizer, person, relation, args)
                validate_pool([row], people, relation, args.samples)
                stream.write(json.dumps(row, ensure_ascii=True)+'\n')
                stream.flush()
    print(f'Completed pools: {output}')


if __name__ == '__main__':
    main()
