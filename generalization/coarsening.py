"""Confidence versus coarser/not-coarser outcomes; no absolute-level cutoff."""
import argparse
from importlib import import_module
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'generalization'

import warnings

import numpy as np
import pandas as pd

from generalization.benchmark_data import load_cases
from shared.io import read_rows, write
from generalization.generate import validate_records
from generalization.preference import direction, question_key
from generalization.truthfulqa_grouping import domain_records, grouping_metadata

ROOT = Path(__file__).resolve().parent
human_study = import_module('generalization.human-study.human_study')

ORDERABLE = {'base_more_specific', 'equal', 'finetuned_more_specific'}


def generation_features(models, datasets, directory, study):
    calibration = {question_key(r['question']) for r in read_rows(study/'data/pairs.jsonl')}
    rows = []
    for dataset in datasets:
        cases = load_cases(dataset)
        for model in models:
            generations = read_rows(directory/dataset/f'{model}.jsonl')
            validate_records(generations, cases, complete=True)
            index = {(str(r['id']), r['variant']): r for r in generations}
            for case in cases:
                base, ft = (index[(str(case['id']), v)] for v in ('base', 'finetuned'))
                tokens = base['first_sentence_tokens']
                confidence = base['first_sentence_mean_token_logprob']
                if tokens and (confidence is None or not math.isfinite(confidence) or
                    not math.isclose(confidence, base['first_sentence_logprob_sum']/tokens, abs_tol=1e-7)):
                    raise ValueError('Base confidence is invalid or inconsistent with the token sum/count')
                rows.append({'model': model, 'dataset': dataset, 'source_id': str(case['id']),
                    'qid': dataset+':'+str(case['qid']), 'question': case['question'],
                    'relation': case['relation'], 'original_relation': case['relation'],
                    'base_response': base['first_sentence'], 'finetuned_response': ft['first_sentence'],
                    'base_granuscore': base['first_sentence_granuscore'],
                    'finetuned_granuscore': ft['first_sentence_granuscore'],
                    'base_words': len(base['first_sentence'].split()), 'base_logprob': confidence,
                    'base_answer_tokens': tokens, 'calibration_question': question_key(case['question']) in calibration})
    return rows


def label_combined(rows, parameters):
    eligible, excluded = [], []
    for row in rows:
        reason = None
        texts = (row['base_response'], row['finetuned_response'])
        if not all(t.strip() for t in texts):
            reason = 'empty_answer'
        elif any(human_study.abstention(t) for t in texts):
            reason = 'standalone_abstention'
        elif not row['base_answer_tokens'] or row['base_logprob'] is None:
            reason = 'missing_confidence'
        elif any(v is None or not math.isfinite(v) for v in (row['base_granuscore'], row['finetuned_granuscore'])):
            reason = 'missing_granuscore'
        if reason:
            excluded.append({**row, 'exclusion': reason})
            continue
        label, score = direction(parameters, row['base_granuscore'], row['finetuned_granuscore'], *texts)
        eligible.append({**row, 'direction': label, 'combined_shift': score,
                         'coarser': int(label == 'base_more_specific')})
    return eligible, excluded


def label_human(features, study):
    pairs, labels = human_study.adjudicated_study(study)
    index = {(r['model'], r['dataset'], r['source_id']): r for r in features}
    eligible, excluded, all_sampled = [], [], []
    for pair in pairs:
        key = pair['model'], pair['dataset'], pair['source_id']
        if key not in index:
            continue
        row = index[key]
        if row['question'] != pair['question'] or any(row[f'{v}_response'] != pair[v]['text'] for v in ('base', 'finetuned')):
            raise ValueError('Human pair and base-confidence generation do not match')
        label = labels[pair['pair_id']]
        record = {**row, 'pair_id': pair['pair_id'], 'direction': label}
        if not row['base_answer_tokens'] or row['base_logprob'] is None:
            excluded.append({**record, 'exclusion': 'missing_confidence'})
            continue
        all_sampled.append(record)
        if label not in ORDERABLE:
            excluded.append({**record, 'exclusion': label})
        else:
            eligible.append({**record, 'coarser': int(label == 'base_more_specific')})
    return eligible, excluded, all_sampled


def label_judge(features, predictions):
    index = {}
    for row in predictions:
        key = row['model'], row['dataset'], row['source_id']
        if key in index:
            raise ValueError('Duplicate granularity-judge pair')
        index[key] = row
    eligible, excluded = [], []
    for row in features:
        prediction = index.get((row['model'], row['dataset'], row['source_id']))
        if prediction is None:
            raise ValueError('Missing granularity-judge pair; finish judging before analysis')
        if any(prediction[k] != row[k] for k in ('question', 'base_response', 'finetuned_response')):
            raise ValueError('Granularity-judge and confidence generation texts differ')
        label = prediction['direction']
        if label not in ORDERABLE | {'abstention_present', 'not_comparable'}:
            raise ValueError('Invalid granularity-judge direction')
        record = {**row, 'direction': label}
        if label not in ORDERABLE:
            excluded.append({**record, 'exclusion': label})
        elif not row['base_answer_tokens'] or row['base_logprob'] is None:
            excluded.append({**record, 'exclusion': 'missing_confidence'})
        else:
            eligible.append({**record, 'coarser': int(label == 'base_more_specific')})
    return eligible, excluded


def attach_quartiles(rows, population, fields):
    """Set cutoffs once; human cutoffs precede non-orderable-label exclusions."""
    grouped = defaultdict(list)
    for row in population:
        grouped[tuple(row[k] for k in fields)].append(row['base_logprob'])
    cutoffs = {key: np.quantile(values, [.25, .75]) for key, values in grouped.items()}
    result = []
    for row in rows:
        q25, q75 = cutoffs[tuple(row[k] for k in fields)]
        confidence = row['base_logprob']
        low = (1 if confidence <= q25 else 0 if confidence >= q75 else None) if q25 < q75 else None
        result.append({**row, 'q25': float(q25), 'q75': float(q75), 'low_confidence': low})
    return result


def matched_outcomes(features, predictions, parameters):
    """One judge-orderable cohort and one set of cutoffs for both outcomes."""
    judged, excluded = label_judge(features, predictions)
    for row in judged:
        if not all(row[k].strip() for k in ('base_response', 'finetuned_response')) or any(
                row[k] is None or not math.isfinite(row[k]) for k in ('base_granuscore', 'finetuned_granuscore', 'base_logprob')):
            raise ValueError('Judge-orderable pair lacks inputs required for the matched comparison')
    judged = attach_quartiles(judged, judged, ('model', 'dataset', 'relation'))
    combined = []
    for row in judged:
        label, score = direction(parameters, row['base_granuscore'], row['finetuned_granuscore'],
                                 row['base_response'], row['finetuned_response'])
        combined.append({**row, 'direction': label, 'combined_shift': score,
                         'coarser': int(label == 'base_more_specific')})
    return combined, judged, excluded


def logistic(rows, *, human=False, adjustment='relation-only'):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    frame = pd.DataFrame([r for r in rows if r['low_confidence'] is not None])
    n = len(frame)
    result = {'cases': n, 'positive_cases': int(frame['coarser'].sum()) if n else 0,
              'effect': 'bottom versus top base-confidence quartile',
              'qid_clusters': int(frame['qid'].nunique()) if n else 0}
    if n < (20 if human else 40):
        return {**result, 'status': 'not_estimable', 'reason': 'too_few_cases'}
    if frame['coarser'].nunique() < 2 or frame['low_confidence'].nunique() < 2:
        return {**result, 'status': 'not_estimable', 'reason': 'no_outcome_or_confidence_variation'}
    frame['dataset_relation'] = frame['dataset']+'/'+frame['relation']
    terms = ['low_confidence']
    if frame['model'].nunique() > 1:
        terms.append('C(model)')
    group = 'dataset' if human else 'dataset_relation'
    if frame[group].nunique() > 1:
        terms.append(f'C({group})')
    if adjustment != 'relation-only' and frame['base_granuscore'].nunique() > 1:
        if adjustment == 'linear':
            terms.append('base_granuscore')
        elif adjustment == 'shared-spline' or human:
            terms.append('bs(base_granuscore, df=3)')
        else:
            terms.append('C(dataset_relation):bs(base_granuscore, df=3)')
    if adjustment != 'relation-only' and frame['base_words'].nunique() > 1:
        terms.append('base_words')
    formula = 'coarser ~ '+' + '.join(terms)
    result['formula'] = formula
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': frame['qid']})
        if any('separation' in str(w.message).lower() for w in captured):
            raise ValueError('separation: unregularized estimates unreliable')
        if not model.converged or np.linalg.matrix_rank(model.model.exog) < model.model.exog.shape[1]:
            raise ValueError('nonconvergence or rank-deficient design')
        coefficient, se, p = (float(x['low_confidence']) for x in (model.params, model.bse, model.pvalues))
        lower, upper = model.conf_int().loc['low_confidence'].to_numpy(dtype=float)
        if not np.isfinite([coefficient, se, p, lower, upper]).all() or se <= 0 or upper <= lower:
            raise ValueError('invalid uncertainty estimate')
        return {**result, 'status': 'estimated', 'log_odds_coefficient': coefficient,
                'cluster_robust_standard_error': se, 'odds_ratio': math.exp(coefficient),
                'odds_ratio_ci_lower': math.exp(lower), 'odds_ratio_ci_upper': math.exp(upper), 'p_value': p}
    except Exception as error:
        return {**result, 'status': 'not_estimable', 'reason': f'{type(error).__name__}: {error}'}


def analyze(rows, *, human=False, adjustment='relation-only'):
    rates = {}
    for name, value in (('low_confidence', 1), ('high_confidence', 0)):
        subset = [r for r in rows if r['low_confidence'] == value]
        rates[name] = {'pairs': len(subset), 'coarser': sum(r['coarser'] for r in subset),
                       'coarser_rate': sum(r['coarser'] for r in subset)/len(subset) if subset else None}
    return {'eligible_pairs': len(rows), 'directions': dict(Counter(r['direction'] for r in rows)),
            'confidence_extremes': rates,
            'coarser_vs_not_coarser': logistic(rows, human=human, adjustment=adjustment)}


def breakdown(rows, *, human=False, adjustment='relation-only'):
    def fit(group):
        return analyze(group, human=human, adjustment=adjustment)
    return {'overall': fit(rows),
        'by_model': {m: fit([r for r in rows if r['model'] == m]) for m in sorted({r['model'] for r in rows})},
        'by_dataset': {d: {'overall': fit([r for r in rows if r['dataset'] == d]),
            'by_model': {m: fit([r for r in rows if r['dataset'] == d and r['model'] == m]) for m in sorted({r['model'] for r in rows})},
            'by_relation': {rel: fit([r for r in rows if r['dataset'] == d and r['relation'] == rel])
                            for rel in sorted({r['relation'] for r in rows if r['dataset'] == d})}}
            for d in sorted({r['dataset'] for r in rows})}}


def text_summary(report):
    lines = ['Coarser versus not coarser: no absolute GranuScore filter.',
             'Coarser = 1; equal/finer = 0. Odds ratio > 1 supports more coarsening at lower confidence.',
             '95% cluster-robust Wald intervals. Associations, not causal effects.', '']
    lines += ['Adjustment: '+('relation/category only (plus model for pooled-model fits); no starting GranuScore or word-count controls.'
        if report['adjustment'] == 'relation-only' else report['adjustment']+' starting-GranuScore adjustment, starting word count and relation/category.'), '']
    if report.get('matched_judge_orderable'):
        lines += ['MATCHED COHORT: identical judge-orderable pairs, confidence cutoffs and regression controls.',
                  'Only the coarser/equal/finer outcome differs between measures.', '']
    for source in ('combined', 'judge', 'human'):
        if source not in report:
            continue
        lines += [('GranuScore + length' if source == 'combined' else source.upper()), 'Dataset / model         Eligible  Extremes  Odds ratio [95% CI]']
        groups = [('all / all', report[source]['overall'])]
        if source == 'human':
            groups += [(f'all / {m}', group) for m, group in report[source]['by_model'].items()]
            groups += [(f'{d} / all', group['overall']) for d, group in report[source]['by_dataset'].items()]
        groups += [(f'{d} / {m}', v) for d, results in report[source]['by_dataset'].items()
                   for m, v in results['by_model'].items()]
        for name, group in groups:
            result = group['coarser_vs_not_coarser']
            shown = (f'{result["odds_ratio"]:.2f} [{result["odds_ratio_ci_lower"]:.2f}, {result["odds_ratio_ci_upper"]:.2f}]'
                     if result['status'] == 'estimated' else 'not estimable: '+result['reason'])
            lines.append(f'{name:<25}{group["eligible_pairs"]:>7} {result["cases"]:>9}  {shown}')
        lines.append('')
    if report.get('matched_judge_orderable'):
        lines += ['Both measures exclude the same judge-labeled abstention/not-comparable pairs.',
                  'This comparison conditions on judge orderability; it is not an independent validation of that selection.']
    else:
        lines += ['Human model×dataset cells are small; failed fits are not treated as null effects.',
                  'GranuScore + length results exclude only recognizable standalone abstentions, not all implicit abstentions.']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=('combined', 'judge', 'human', 'both', 'all', 'matched'), default='both')
    parser.add_argument('--models', nargs='+', choices=('lfm', 'qwen', 'gemma', 'ministral'), default=['lfm', 'qwen', 'gemma', 'ministral'])
    parser.add_argument('--datasets', nargs='+', choices=('granola', 'simpleqa', 'truthfulqa'), default=['granola', 'simpleqa', 'truthfulqa'])
    parser.add_argument('--generations-dir', type=Path, default=ROOT/'generations')
    parser.add_argument('--study-dir', type=Path, default=ROOT/'human-study')
    parser.add_argument('--metric', type=Path, default=ROOT/'human-study/results/metric.json')
    parser.add_argument('--judge-predictions', type=Path, default=ROOT/'granularity-judge/results/first-sentence/predictions.jsonl')
    parser.add_argument('--adjustment', choices=('relation-only', 'linear', 'shared-spline', 'relation-specific-spline'), default='relation-only')
    parser.add_argument('--truthfulqa-grouping', choices=('category', 'domains'), default='domains')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    name = 'coarsening-matched' if args.source == 'matched' else 'coarsening'
    if args.adjustment != 'relation-only':
        name += '-fully-adjusted' if args.adjustment == 'linear' else '-'+args.adjustment
    args.output_dir = args.output_dir or ROOT/'results'/name
    if len(set(args.models)) != len(args.models) or len(set(args.datasets)) != len(args.datasets):
        parser.error('Choose distinct models and datasets')
    features = generation_features(args.models, args.datasets, args.generations_dir, args.study_dir)
    if args.truthfulqa_grouping == 'domains':
        features = [r for r in features if r['dataset'] != 'truthfulqa']+domain_records([r for r in features if r['dataset'] == 'truthfulqa'])
    report = {'outcome': 'coarser versus equal/finer', 'scope': 'first_sentence', 'absolute_level_filter': None,
        'confidence': 'Base first-sentence mean emitted-token log probability; EOS/padding excluded.',
        'adjustment': args.adjustment, 'starting_controls': [] if args.adjustment == 'relation-only' else ['base_granuscore', 'base_words'],
        'truthfulqa_grouping': grouping_metadata() if args.truthfulqa_grouping == 'domains' else 'original categories',
        'limitations': ['No known absolute starting-level risk set; no GranuScore-98 filter.',
            'Only standalone canonical abstentions can be excluded in unjudged pools; implicit abstentions/non-comparability may remain.',
            'Human results condition on the sampled nonidentical pairs; model×dataset fits have low power.',
            'Starting controls are automatic GranuScore/word-count proxies, not human-annotated absolute levels.']}
    details, exclusions = [], []
    if args.adjustment == 'relation-only':
        report['limitations'] = [v for v in report['limitations'] if not v.startswith('Starting controls')]
        report['limitations'].append('Descriptive within-relation association; does not hold starting answer granularity or length fixed.')
    if args.source == 'matched':
        metric = json.loads(args.metric.read_text())
        if metric['scope'] != 'first_sentence':
            raise ValueError('Expected a first-sentence calibrated metric')
        combined, judged, excluded = matched_outcomes(features, read_rows(args.judge_predictions), metric['parameters'])
        report['matched_judge_orderable'] = True
        report['limitations'] = ['Conditions on LLM-judge orderability, not independent of the judge.',
            'Associations are not causal effects or correctness gains.',
            'Starting answer granularity and length are not held fixed.' if args.adjustment == 'relation-only' else
            'Starting controls are automatic GranuScore/word-count proxies, not absolute granularity.']
        report['outcome_agreement'] = human_study.agreement([r['direction'] for r in judged], [r['direction'] for r in combined])
        for source, rows in (('combined', combined), ('judge', judged)):
            report[source] = {**breakdown(rows, adjustment=args.adjustment),
                'exclusions': dict(Counter(r['exclusion'] for r in excluded)),
                'confidence_quartiles_within': ['model', 'dataset', 'relation'],
                'excluding_calibration_questions': breakdown([r for r in rows if not r['calibration_question']], adjustment=args.adjustment)}
            details.extend({**r, 'source': source} for r in rows)
        report['combined']['parameters'] = metric['parameters']
        exclusions.extend({**r, 'source': 'shared'} for r in excluded)
    if args.source in ('combined', 'both', 'all'):
        metric = json.loads(args.metric.read_text())
        if metric['scope'] != 'first_sentence':
            raise ValueError('Expected a first-sentence calibrated metric')
        eligible, excluded = label_combined(features, metric['parameters'])
        eligible = attach_quartiles(eligible, eligible, ('model', 'dataset', 'relation'))
        report['combined'] = {**breakdown(eligible, adjustment=args.adjustment),
            'parameters': metric['parameters'], 'exclusions': dict(Counter(r['exclusion'] for r in excluded)),
            'confidence_quartiles_within': ['model', 'dataset', 'relation'],
            'excluding_calibration_questions': breakdown([r for r in eligible if not r['calibration_question']], adjustment=args.adjustment)}
        details.extend({**r, 'source': 'combined'} for r in eligible)
        exclusions.extend({**r, 'source': 'combined'} for r in excluded)
    if args.source in ('judge', 'all'):
        eligible, excluded = label_judge(features, read_rows(args.judge_predictions))
        eligible = attach_quartiles(eligible, eligible, ('model', 'dataset', 'relation'))
        report['judge'] = {**breakdown(eligible, adjustment=args.adjustment),
            'exclusions': dict(Counter(r['exclusion'] for r in excluded)),
            'confidence_quartiles_within': ['model', 'dataset', 'relation']}
        details.extend({**r, 'source': 'judge'} for r in eligible)
        exclusions.extend({**r, 'source': 'judge'} for r in excluded)
    if args.source in ('human', 'both', 'all'):
        eligible, excluded, population = label_human(features, args.study_dir)
        eligible = attach_quartiles(eligible, population, ('model', 'dataset'))
        report['human'] = {**breakdown(eligible, human=True, adjustment=args.adjustment),
            'exclusions': dict(Counter(r['exclusion'] for r in excluded)),
            'confidence_quartiles_within': ['model', 'dataset'], 'cutoffs_precede_label_exclusions': True}
        details.extend({**r, 'source': 'human'} for r in eligible)
        exclusions.extend({**r, 'source': 'human'} for r in excluded)
    write(args.output_dir/'summary.json', report)
    write(args.output_dir/'pairs.jsonl', details, rows=True)
    write(args.output_dir/'excluded.jsonl', exclusions, rows=True)
    (args.output_dir/'summary.txt').write_text(text_summary(report), encoding='utf-8')
    print(text_summary(report))


if __name__ == '__main__':
    main()
