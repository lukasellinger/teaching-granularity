"""Submit missing granularity judgments to OpenAI Batch, then collect to the shared cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'generalization.granularity-judge'


from . import judge


def submit(rows, prompt, labels, model, cache_path, directory, client):
    state_path = directory/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if 'batch_id' not in state:
            raise ValueError('Submission has no recorded batch ID; check the API batch list before retrying to avoid duplicates')
        print(f"Existing batch: {state['batch_id']}. Use collect; no duplicate submitted.")
        return state
    jobs = dict(judge.make_job(row, prompt, labels, model) for row in rows)
    cached = judge.read_cache(cache_path)
    missing = {key: request for key, request in jobs.items() if key not in cached}
    if not missing:
        print('All judgments are cached; run judge.py score.')
        return None
    if len(missing) > 50000:
        raise ValueError('More than 50,000 unique requests; submit smaller --models/--datasets subsets with separate --batch-dir values')
    path = directory/'input.jsonl'
    judge.study.save(path, [{'custom_id': key, 'method': 'POST', 'url': '/v1/responses', 'body': request}
                           for key, request in missing.items()], jsonl=True)
    if path.stat().st_size > 200_000_000:
        raise ValueError('Batch input exceeds 200 MB; choose a smaller subset')
    state = {'model': model, 'pairs': len(rows), 'unique_requests': len(missing),
             'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    # Record submission intent before the remote writes, preventing accidental resubmission.
    judge.study.save(state_path, state)
    with path.open('rb') as stream:
        uploaded = client.files.create(file=stream, purpose='batch')
    state['input_file_id'] = uploaded.id
    judge.study.save(state_path, state)
    batch = client.batches.create(input_file_id=uploaded.id, endpoint='/v1/responses', completion_window='24h')
    state.update(batch_id=batch.id, status=batch.status)
    judge.study.save(state_path, state)
    print(json.dumps(state, indent=2))
    return state


def collect(directory, cache_path, labels, client):
    state_path = directory/'state.json'
    state = json.loads(state_path.read_text())
    path = directory/'input.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != state['input_sha256']:
        raise ValueError('Submitted batch input changed')
    expected = {row['custom_id'] for row in judge.study.read_jsonl(path)}
    batch = client.batches.retrieve(state['batch_id'])
    state['status'] = batch.status
    judge.study.save(state_path, state)
    print(f"{batch.id}: {batch.status}; {batch.request_counts}")
    if batch.status not in {'completed', 'failed', 'expired', 'cancelled'}:
        return False
    if batch.error_file_id:
        (directory/'errors.jsonl').write_bytes(client.files.content(batch.error_file_id).content)
    if not batch.output_file_id:
        print(f'No output file. Batch errors: {batch.errors}')
        return False
    output = directory/'output.jsonl'
    output.write_bytes(client.files.content(batch.output_file_id).content)
    records = judge.study.read_jsonl(output)
    keys = [row['custom_id'] for row in records]
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError('Duplicate or unknown batch output IDs')
    from openai.types.responses import Response
    cache = judge.read_cache(cache_path)
    failures, added = [], 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('a', encoding='utf-8') as stream:
        for row in records:
            key = row['custom_id']
            if key in cache:
                continue
            try:
                response = row['response']
                if row.get('error') or not response or response['status_code'] != 200:
                    raise ValueError('Failed batch request')
                result = judge.parse_response(Response.model_validate(response['body']), labels)
                record = {'key': key, **result}
            except (ValueError, KeyError, TypeError):
                failures.append(key)
                continue
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
            stream.flush()
            cache[key] = record
            added += 1
    missing = expected-cache.keys()
    print(f'{added} judgments added; {len(missing)} still missing; {len(failures)} invalid outputs. No failed labels substituted.')
    return not missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('submit', 'collect'))
    parser.add_argument('--batch-dir', type=Path, default=judge.ROOT/'batch')
    parser.add_argument('--cache', type=Path, default=judge.ROOT/'cache/judgments.jsonl')
    parser.add_argument('--study-dir', type=Path, default=judge.STUDY)
    parser.add_argument('--models', nargs='+', choices=judge.study.MODELS, default=list(judge.study.MODELS))
    parser.add_argument('--datasets', nargs='+', choices=('granola', 'simpleqa', 'truthfulqa'), default=['granola', 'simpleqa', 'truthfulqa'])
    parser.add_argument('--generations-dir', type=Path, default=judge.ROOT.parent/'generations')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--env-file', type=Path, default=judge.ROOT.parents[1]/'.env')
    args = parser.parse_args()
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(args.env_file)
    client = OpenAI(timeout=120)
    prompt, labels = judge.instructions(args.study_dir)
    if args.command == 'submit':
        if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
            parser.error('Choose distinct models and datasets')
        rows = judge.generation_pairs(args.models, args.datasets, args.generations_dir, args.study_dir)
        submit(rows, prompt, labels, args.model, args.cache, args.batch_dir, client)
    else:
        collect(args.batch_dir, args.cache, labels, client)


if __name__ == '__main__':
    main()
