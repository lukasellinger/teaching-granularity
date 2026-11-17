"""Recognize Granola answer levels using the frozen base model, not free generation."""
from shared.constants import GRANOLA_TRAIN_RELATIONS as TRAIN_RELATIONS
from shared.constants import MODEL_NAMES
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from generalization.benchmark_data import dataset_settings, granola_cases
from shared.io import read_rows, write
from .scoring import (ContinuationEncoder, build_candidate_pools, build_distractor_panel,
                      build_hierarchy_conflicts, create_prompt_variants, score_level,
                      summarize_token_logprobs, validate_hierarchy)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def load_data():
    from datasets import load_dataset
    settings = dataset_settings('granola')
    raw = load_dataset(settings['dataset_id'], revision=settings['revision'], split=settings['split'])
    return prepare_rows(raw)


def prepare_rows(raw):
    """Same cleaned pool as evaluation, but retain all relations for distractors."""
    raw = list(raw)
    originals = {str(r['Unnamed: 0']): r for r in raw}
    if len(originals) != len(raw):
        raise ValueError('Duplicate original Granola IDs')
    cases = granola_cases(raw, {r['relation'] for r in raw})
    return [{'id': int(c['id']), 'question': c['question'], 'relation': c['relation'],
             'question_entity': originals[c['id']]['question_entity'],
             'question_entity_qid': originals[c['id']].get('question_entity_qid'),
             'granola_answers': c['reference_answers'],
             'normalized_levels': [1.0] if len(c['reference_answers']) == 1 else
                [1+3*i/(len(c['reference_answers'])-1) for i in range(len(c['reference_answers']))]}
            for c in cases]


def select_level(levels):
    """An uninterrupted coarse-to-fine prefix; isolated finer passes do not count."""
    selected = None
    for level in sorted(levels, key=lambda r: r['level'], reverse=True):
        if not level['passed']:
            break
        selected = level
    return selected


def measure_level(raw):
    correct = [float(r['mean_logprob']) for r in raw['correct']['prompt_scores']]
    distractors = [[float(r['mean_logprob']) for r in d['prompt_scores']] for d in raw['distractors']]
    if len(correct) != 3 or len(distractors) != 20 or any(len(d) != 3 for d in distractors):
        raise ValueError('Expected three prompts and twenty distractors')
    if not all(math.isfinite(v) for scores in [correct, *distractors] for v in scores):
        raise ValueError('Non-finite recognition score')
    margins = [value-max(d[i] for d in distractors) for i, value in enumerate(correct)]
    wins = sum(value > d[i] for i, value in enumerate(correct) for d in distractors)
    return {'prompt_margins': margins, 'rank_fraction': wins/60,
            'passed': wins == 60 and min(margins) >= 0}


def score_row(row, scorer, pools, conflicts, model):
    prompts = create_prompt_variants(row, count=3, seed=42)
    levels = []
    for level, correct in validate_hierarchy(row):
        panel = build_distractor_panel(row=row, fallback_index=0, level=level, correct=correct,
            pool=pools.get((row['relation'], level), []), panel_size=20, seed=42,
            hierarchy_conflicts=conflicts)
        result = {'level': level, 'answer': correct}
        if panel is None:
            result.update(status='INSUFFICIENT_DISTRACTORS', passed=False)
        else:
            raw = score_level(scorer=scorer, prompts=prompts, correct=correct, distractors=panel)
            result.update(status='SCORED', **measure_level(raw),
                correct_mean_logprobs=[s['mean_logprob'] for s in raw['correct']['prompt_scores']],
                distractors=[{'answer': d['answer'], 'mean_logprobs': [s['mean_logprob'] for s in d['prompt_scores']]}
                             for d in raw['distractors']])
        levels.append(result)
    selected = select_level(levels)
    return {'row_index': row['id'], 'entity': row['question_entity'], 'qid': row['question_entity_qid'],
        'question': row['question'], 'relation': row['relation'], 'scoring_model': MODEL_NAMES[model],
        'prompt_variants': prompts, 'level_results': levels,
        'selected_level': selected['level'] if selected else None,
        'selected_answer': selected['answer'] if selected else None, 'no_level_chosen': selected is None}


def fast_tokenizer(model_id, revision):
    from transformers import AutoTokenizer
    from shared.models import load_fast_tokenizer
    if 'Ministral-3' in model_id:
        return load_fast_tokenizer(model_id, revision)
    return AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)


class TransformersScorer:
    def __init__(self, settings, batch_size, device):
        import torch
        from shared.models import causal_model_class
        if not device.startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('Recognition requires CUDA; use --dry-run for the CPU preparation check')
        self.torch, self.batch_size, self.device = torch, batch_size, device
        self.tokenizer = fast_tokenizer(settings['model_id'], settings['revision'])
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = ContinuationEncoder(self.tokenizer, assistant_prefix='')
        cls = causal_model_class(settings['model_id'])
        self.model = cls.from_pretrained(settings['model_id'], revision=settings['revision'],
            dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': device}).eval()

    def score(self, questions, candidates):
        if len(questions) != len(candidates):
            raise ValueError('Misaligned questions and candidates')
        scores = []
        for start in range(0, len(questions), self.batch_size):
            rows = [self.encoder.encode(q, a) for q, a in zip(questions[start:start+self.batch_size], candidates[start:start+self.batch_size], strict=True)]
            if any(len(r['input_ids']) > 2048 for r in rows):
                raise ValueError('Recognition sequence exceeds 2048 tokens; no silent truncation')
            inputs = self.tokenizer.pad({'input_ids': [r['input_ids'] for r in rows],
                'attention_mask': [[1]*len(r['input_ids']) for r in rows]}, padding=True, return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits
                for i, row in enumerate(rows):
                    positions = self.torch.tensor(row['candidate_positions'], device=logits.device)
                    values = logits[i, positions-1].float()
                    ids = inputs['input_ids'][i, positions]
                    logprobs = values.gather(1, ids[:, None]).squeeze(1)-self.torch.logsumexp(values, dim=-1)
                    scores.append(summarize_token_logprobs(logprobs.cpu().tolist()))
            del logits
        return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--batch-size', type=int, default=8, help='Candidate sequences per forward pass')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int, help='Smoke test; use a separate output directory')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit is not None and args.limit < 1:
        parser.error('batch-size and limit must be positive')
    settings = json.loads((REPO/'generalization/generation_settings.json').read_text())['models'][args.model]
    rows = load_data()
    relations = TRAIN_RELATIONS | set(dataset_settings('granola')['held_out_relations'])
    eligible = [r for r in rows if r['relation'] in relations]
    if args.limit:
        eligible = eligible[:args.limit]
    output = args.output_dir or ROOT/'recognition'/args.model
    identity = {'model_id': settings['model_id'], 'revision': settings['revision'],
        'dataset': dataset_settings('granola'), 'prompt_sha256': hashlib.sha256((ROOT/'prompts.json').read_bytes()).hexdigest(),
        'protocol': {'version': 1, 'prompts': 3, 'distractors': 20, 'seed': 42, 'minimum_win_rate': 1.0,
            'margin_threshold': 0.0, 'dtype': 'bfloat16', 'attention': 'sdpa', 'limit': args.limit}}
    print(f'{len(rows)} cleaned rows; {len(eligible)} non-location rows; {settings["model_id"]}')
    if args.dry_run:
        print(json.dumps(identity, indent=2))
        return
    run = output/'run.json'
    if run.exists() and json.loads(run.read_text()) != identity:
        raise ValueError('Recognition settings changed; use a new output directory')
    path = output/'results.jsonl'
    if path.exists() and not run.exists():
        raise ValueError('Recognition cache has no settings; use a new output directory')
    write(run, identity)
    cached = read_rows(path) if path.exists() else []
    inputs = {str(r['id']): r for r in eligible}
    seen = set()
    for row in cached:
        key = str(row['row_index'])
        source = inputs.get(key)
        if key in seen or source is None or row['scoring_model'] != MODEL_NAMES[args.model] or any(row[k] != source[k] for k in ('question', 'relation')):
            raise ValueError('Duplicate or mismatched recognition cache row')
        seen.add(key)
    pending = [r for r in eligible if str(r['id']) not in seen]
    print(f'{len(cached)} cached; {len(pending)} scheduled')
    if pending:
        from tqdm import tqdm
        pools, conflicts = build_candidate_pools(rows), build_hierarchy_conflicts(rows)
        scorer = TransformersScorer(settings, args.batch_size, args.device)
        with path.open('a', encoding='utf-8') as stream:
            for row in tqdm(pending, desc='Granola recognition', unit='row'):
                result = score_row(row, scorer, pools, conflicts, args.model)
                stream.write(json.dumps(result, ensure_ascii=False)+'\n')
                stream.flush()
                cached.append(result)
    selected = [{k: r[k] for k in ('row_index', 'entity', 'qid', 'question', 'relation', 'scoring_model', 'selected_level', 'selected_answer')}
                for r in cached if r['selected_level'] is not None]
    write(output/'selected.jsonl', selected, rows=True)
    print(json.dumps({'rows': len(cached), 'selected': len(selected),
                      'selected_by_relation': dict(Counter(r['relation'] for r in selected))}, indent=2))


if __name__ == '__main__':
    main()
