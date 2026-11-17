"""Cached direct or batch judgments; never modify saved evaluations automatically."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from shared.io import digest, file_hash, read_rows, write
from evaluation.vital_persons.plot_alignment import MODELS, VARIANTS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent/'results'
LABELS = ('Answer', 'No answer')
MODEL = 'gpt-6-luna'


def needs_judge(row):
    """Only missing/failed parses, excluding locally recognized standalone IDK."""
    from parsing.temporal import is_idk_response
    return not is_idk_response(row['answer']) and row['parse_status'] != 'parsed'


def make_job(row, prompt, model=MODEL):
    request = {'model': model, 'instructions': prompt,
               'input': json.dumps({'question': row['question'], 'response': row['answer']}, ensure_ascii=False),
               'store': False, 'reasoning': {'effort': 'none'},
               'text': {'format': {'type': 'json_schema', 'name': 'answer_status', 'strict': True,
                   'schema': {'type': 'object', 'properties': {
                       'label': {'type': 'string', 'enum': list(LABELS)}, 'reason': {'type': 'string'}},
                       'required': ['label', 'reason'], 'additionalProperties': False}}}}
    return digest(request), request


def pool(directory, split, models=MODELS, variants=VARIANTS):
    rows = []
    for model in models:
        for modality in ('temporal', 'location'):
            for variant in variants:
                path = directory/model/split/'generations'/modality/f'{variant}.jsonl'
                # The scored outcome preserves the bare question, unlike user_prompt,
                # which may also contain baseline-specific generation instructions.
                outcomes = read_rows(directory/model/split/'outcomes'/modality/f'{variant}.jsonl')
                metadata = {r['case_id']: r for r in outcomes}
                if len(metadata) != len(outcomes):
                    raise ValueError('Duplicate outcome case IDs')
                seen = set()
                for r in read_rows(path):
                    if r['case_id'] in seen:
                        raise ValueError(f'Duplicate generation: {path}/{r["case_id"]}')
                    seen.add(r['case_id'])
                    saved = metadata.get(r['case_id'])
                    if saved is None or saved['answer'] != r['answer']:
                        raise ValueError('Saved outcome and generation text differ')
                    question = saved['prompt']
                    if not isinstance(question, str) or not question.strip() or not isinstance(r['answer'], str):
                        raise ValueError(f'Question and response required: {path}')
                    rows.append({'item_id': digest([split, model, modality, variant, r['case_id']]),
                        'case_id': r['case_id'], 'qid': saved['qid'], 'model': model, 'modality': modality,
                        'variant': variant, 'split': split, 'question': question, 'answer': r['answer'],
                        'parse_status': saved['parse_status']})
    return rows


def read_cache(path):
    records = {}
    if path.exists():
        for r in read_rows(path):
            if r['label'] not in LABELS or not isinstance(r['reason'], str):
                raise ValueError('Invalid cached status judgment')
            if r['key'] in records and records[r['key']] != r:
                raise ValueError('Conflicting cache records')
            records[r['key']] = r
    return records


@contextmanager
def writer_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix+'.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def parse_body(body):
    if body.get('status') != 'completed':
        raise ValueError('Incomplete judge response')
    texts = [c['text'] for o in body.get('output', []) if o.get('type') == 'message'
             for c in o.get('content', []) if c.get('type') == 'output_text']
    result = json.loads(''.join(texts))
    if set(result) != {'label', 'reason'} or result['label'] not in LABELS or not isinstance(result['reason'], str):
        raise ValueError('Invalid judge label/schema')
    return {**result, 'response_id': body.get('id'), 'model': body.get('model'), 'usage': body.get('usage')}


def prepare_batch(rows, prompt, model, cache_path, directory):
    jobs = dict(make_job(r, prompt, model) for r in rows)
    cached = read_cache(cache_path)
    missing = {key: request for key, request in jobs.items() if key not in cached}
    state_path = directory/'state.json'
    if state_path.exists():
        raise ValueError('Batch directory already has a state; use collect or a new directory')
    if not missing:
        print('All judgments cached; use score.')
        return None
    if len(missing) > 50000:
        raise ValueError('Batch exceeds 50,000 jobs; select a smaller model/variant subset')
    path = directory/'input.jsonl'
    write(path, [{'custom_id': k, 'method': 'POST', 'url': '/v1/responses', 'body': v}
                 for k, v in missing.items()], rows=True)
    if path.stat().st_size > 200_000_000:
        raise ValueError('Batch file exceeds 200 MB')
    state = {'model': model, 'items': len(rows), 'unique_requests': len(missing),
             'input_sha256': file_hash(path), 'prompt_sha256': digest(prompt), 'status': 'prepared'}
    write(state_path, state)
    return state


def submit(directory, client):
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text())
    if state.get('batch_id'):
        print(f'Already submitted: {state["batch_id"]}')
        return
    if state['status'] != 'prepared':
        raise ValueError('Submission may have started; inspect the API before resubmitting')
    path = directory/'input.jsonl'
    if file_hash(path) != state['input_sha256']:
        raise ValueError('Batch input changed')
    state['status'] = 'submitting'
    write(state_path, state)
    with path.open('rb') as stream:
        uploaded = client.files.create(file=stream, purpose='batch')
    state['input_file_id'] = uploaded.id
    write(state_path, state)
    batch = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/responses', completion_window='24h')
    state.update(batch_id=batch.id, status=batch.status)
    write(state_path, state)
    print(f'Submitted {batch.id}: {state["unique_requests"]} requests')


def collect(directory, cache_path, client):
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text())
    path = directory/'input.jsonl'
    if file_hash(path) != state['input_sha256']:
        raise ValueError('Batch input changed')
    requests = {r['custom_id']: r['body'] for r in read_rows(path)}
    if any(digest(body) != key for key, body in requests.items()):
        raise ValueError('Batch IDs do not match request content')
    batch = client.batches.retrieve(state['batch_id'])
    state['status'] = batch.status
    write(state_path, state)
    print(f'{batch.id}: {batch.status}')
    if batch.status not in {'completed', 'failed', 'expired', 'cancelled'}:
        return
    if batch.error_file_id:
        (directory/'errors.jsonl').write_bytes(client.files.content(batch.error_file_id).content)
    if not batch.output_file_id:
        raise ValueError('No batch output; no substitute labels created')
    output = directory/'output.jsonl'
    output.write_bytes(client.files.content(batch.output_file_id).content)
    records = read_rows(output)
    keys = [r['custom_id'] for r in records]
    if len(keys) != len(set(keys)) or not set(keys) <= requests.keys():
        raise ValueError('Duplicate or unknown output IDs')
    with writer_lock(cache_path):
        cached = read_cache(cache_path)
        failures = []
        with cache_path.open('a', encoding='utf-8') as stream:
            for r in records:
                key = r['custom_id']
                if key in cached:
                    continue
                try:
                    if r.get('error') or not r.get('response') or r['response']['status_code'] != 200:
                        raise ValueError('Failed batch request')
                    parsed = parse_body(r['response']['body'])
                except (ValueError, KeyError, TypeError):
                    failures.append(key)
                    continue
                record = {'key': key, **parsed}
                stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                stream.flush()
                cached[key] = record
    print(f'{len(requests.keys()-cached.keys())} missing; {len(failures)} invalid outputs. Successful judgments cached.')


def score(rows, prompt, model, cache_path, output):
    cached = read_cache(cache_path)
    keys = [make_job(r, prompt, model)[0] for r in rows]
    missing = set(keys)-cached.keys()
    if missing:
        raise ValueError(f'{len(missing)} judgments missing; no partial predictions written')
    predictions = [{**r, 'judge': cached[key], 'request_key': key} for r, key in zip(rows, keys, strict=True)]
    write(output/'predictions.jsonl', predictions, rows=True)
    summary = {'model': model, 'prompt_sha256': digest(prompt), 'scope': 'full_answer',
               'items': len(rows), 'unique_requests': len(set(keys)),
               'labels': dict(Counter(r['judge']['label'] for r in predictions))}
    write(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return predictions


def run_direct(rows, prompt, model, cache_path, client, concurrency=8):
    """Direct concurrent requests; cache each success, never substitute failed labels."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if concurrency < 1:
        raise ValueError('Concurrency must be positive')
    jobs = dict(make_job(r, prompt, model) for r in rows)
    with writer_lock(cache_path):
        cached = read_cache(cache_path)
        missing = {k: v for k, v in jobs.items() if k not in cached}
        failures = []
        print(f'{len(jobs)-len(missing)} cached; {len(missing)} direct requests', flush=True)
        def request(body):
            response = client.responses.create(**body)
            return parse_body(response.model_dump())
        with ThreadPoolExecutor(max_workers=concurrency) as executor, cache_path.open('a', encoding='utf-8') as stream:
            futures = {executor.submit(request, body): key for key, body in missing.items()}
            for number, future in enumerate(as_completed(futures), 1):
                key = futures[future]
                try:
                    record = {'key': key, **future.result()}
                except Exception as exc:
                    failures.append({'key': key, 'error_type': type(exc).__name__})
                    print(f'Request failed: {type(exc).__name__}', flush=True)
                else:
                    stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    stream.flush()
                if number % 10 == 0 or number == len(futures):
                    print(f'Judged {number}/{len(futures)}', flush=True)
        if failures:
            write(cache_path.with_suffix('.failures.json'), failures)
            raise ValueError(f'{len(failures)} requests failed; successful judgments cached. Retry run.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'prepare', 'submit', 'collect', 'score'))
    parser.add_argument('--items', type=Path, help='Audit items JSONL; otherwise use the saved generation pool')
    parser.add_argument('--partition', choices=('dev', 'evaluation'), help='Select an audit partition')
    parser.add_argument('--results-dir', type=Path, default=RESULTS)
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--prompt', type=Path, default=ROOT/'prompt.txt')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--cache', type=Path, default=ROOT/'cache/judgments.jsonl')
    parser.add_argument('--batch-dir', type=Path, default=ROOT/'batch/production')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results')
    parser.add_argument('--env-file', type=Path, default=ROOT.parents[2]/'.env')
    parser.add_argument('--allow-api', action='store_true')
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    if args.command in ('submit', 'collect'):
        if not args.allow_api:
            parser.error('--allow-api is required for remote batch operations')
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(args.env_file)
        client = OpenAI(timeout=120, max_retries=2)
        with writer_lock(args.batch_dir/'operation'):
            if args.command == 'submit':
                submit(args.batch_dir, client)
            else:
                collect(args.batch_dir, args.cache, client)
        return
    if len(set(args.models)) != len(args.models) or len(set(args.variants)) != len(args.variants):
        parser.error('Select distinct models/variants')
    if args.partition and not args.items:
        parser.error('--partition requires --items')
    rows = read_rows(args.items) if args.items else pool(args.results_dir, args.split, args.models, args.variants)
    if not args.items:
        rows = [r for r in rows if needs_judge(r)]
    if args.partition:
        rows = [r for r in rows if r['partition'] == args.partition]
    if not rows:
        parser.error('No items selected')
    prompt = args.prompt.read_text()
    if args.command == 'run':
        if not args.allow_api:
            parser.error('--allow-api is required for direct requests')
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(args.env_file)
        run_direct(rows, prompt, args.model, args.cache, OpenAI(timeout=120, max_retries=2), args.concurrency)
        score(rows, prompt, args.model, args.cache, args.output_dir)
        return
    if args.command == 'prepare':
        state = prepare_batch(rows, prompt, args.model, args.cache, args.batch_dir)
        if state:
            print(json.dumps(state, indent=2))
    else:
        score(rows, prompt, args.model, args.cache, args.output_dir)


if __name__ == '__main__':
    main()
