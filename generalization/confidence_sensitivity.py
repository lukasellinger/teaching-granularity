"""Confidence-cutoff and continuous-confidence sensitivity on saved matched pairs.

uv run --locked --extra analysis python -m generalization.confidence_sensitivity
No generations, judge calls, dataset downloads or metric refitting are needed.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from shared.io import read_rows, write, file_hash
from .coarsening import logistic

ROOT = Path(__file__).resolve().parent
TAILS = (0.20, 0.25, 0.30, 0.35)
MODELS = ('lfm', 'ministral', 'qwen', 'gemma')
DATASETS = ('granola', 'simpleqa', 'truthfulqa')
NAMES = {'lfm': 'LFM', 'ministral': 'Ministral', 'qwen': 'Qwen', 'gemma': 'Gemma'}


def continuous(rows, scales, adjustment='relation-only'):
    selected = []
    for row in rows:
        mean, sd = scales[row['relation']]
        if sd > 0:
            selected.append({**row, 'lower_confidence_sd': -(row['base_logprob']-mean)/sd})
    frame = pd.DataFrame(selected)
    result = {'cases': len(frame), 'zero_variance_exclusions': len(rows)-len(frame),
              'effect': 'one within-relation population SD decrease in confidence'}
    if len(frame) < 40 or frame['coarser'].nunique() < 2:
        return {**result, 'status': 'not_estimable', 'reason': 'insufficient cases or outcome variation'}
    result['positive_cases'] = int(frame['coarser'].sum())
    result['qid_clusters'] = int(frame['qid'].nunique())
    formula = 'coarser ~ lower_confidence_sd'
    if frame['relation'].nunique() > 1:
        formula += ' + C(relation)'
    if adjustment == 'linear':
        for control in ('base_granuscore', 'base_words'):
            if frame[control].nunique() > 1:
                formula += ' + ' + control
    result['formula'] = formula
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            fit = smf.glm(formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type='cluster', cov_kwds={'groups': frame['qid']})
        if any('separation' in str(w.message).lower() for w in captured):
            raise ValueError('Separation: unregularized estimate unreliable')
        if not fit.converged or np.linalg.matrix_rank(fit.model.exog) < fit.model.exog.shape[1]:
            raise ValueError('Nonconvergence or rank-deficient design')
        coefficient = float(fit.params['lower_confidence_sd'])
        se = float(fit.bse['lower_confidence_sd'])
        p = float(fit.pvalues['lower_confidence_sd'])
        lo, hi = map(float, fit.conf_int().loc['lower_confidence_sd'])
        if not np.isfinite([coefficient, se, p, lo, hi]).all() or se <= 0 or hi <= lo:
            raise ValueError('Invalid uncertainty estimate')
        return {**result, 'status': 'estimated', 'log_odds_coefficient': coefficient,
                'cluster_robust_standard_error': se, 'odds_ratio': math.exp(coefficient),
                'odds_ratio_ci_lower': math.exp(lo), 'odds_ratio_ci_upper': math.exp(hi),
                'p_value': p}
    except Exception as error:
        return {**result, 'status': 'not_estimable', 'reason': str(error)}


def format_estimate(record):
    if record['status'] != 'estimated':
        return 'Not estimable'
    def endpoint(value):
        return f"{value:.3f}" if f"{value:.2f}" == '1.00' and value != 1 else f"{value:.2f}"
    value = (f"{record['odds_ratio']:.2f} "
             f"[{endpoint(record['odds_ratio_ci_lower'])}, {endpoint(record['odds_ratio_ci_upper'])}]")
    return value


def write_table(results, output, adjustment='relation-only'):
    specs = [f'tails_{int(100*t)}' for t in TAILS] + ['continuous']
    indexed = {(r['source'], r['dataset'], r['model']): r for r in results}
    dataset_names = {'granola': r'\textsc{GranolaEQ}',
                     'simpleqa': r'\textsc{SimpleQA}',
                     'truthfulqa': r'\textsc{TruthfulQA}'}
    tex = [r'\begin{table*}[t]', r'\scriptsize', r'\centering',
           r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{@{}lrrrrr@{}}',
           r'\toprule',
           r'Model & 20\% tails & 25\% tails & 30\% tails & 35\% tails & Continuous \\']
    for dataset in DATASETS:
        tex.extend([r'\midrule',
                    rf"\multicolumn{{6}}{{@{{}}l}}{{{dataset_names[dataset]}}} \\"])
        for source, label in [('combined', 'Combined metric'), ('judge', 'LLM judge')]:
            tex.extend([r'\addlinespace[3pt]',
                        rf'\multicolumn{{6}}{{@{{}}l}}{{\quad\textit{{{label}}}}} \\'])
            for model in MODELS:
                record = indexed[source, dataset, model]
                tex.append(r'\quad\quad '+NAMES[model]+' & '+
                           ' & '.join(format_estimate(record[spec]) for spec in specs)+r' \\')
    adjustment_note = (r'All fits additionally adjust for base-answer \textsc{GranuScore} and word count.'
                       if adjustment == 'linear' else '')
    tex += [r'\bottomrule', r'\end{tabular}',
            r'\caption{Sensitivity of relation-adjusted coarsening odds ratios to confidence specification.',
            r'Brackets show 95\% cluster-robust confidence intervals. Tail specifications compare the lowest and highest',
            r'confidence fractions within each model--dataset--relation group. Continuous estimates use all eligible pairs',
            r'and describe a one-standard-deviation decrease in confidence within these groups.',
            adjustment_note + '}',
            r'\label{tab:confidence-sensitivity-adjusted}' if adjustment == 'linear' else r'\label{tab:confidence-sensitivity}', r'\end{table*}']
    (output/'table.tex').write_text('\n'.join(tex)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT/'results/coarsening-matched/pairs.jsonl')
    parser.add_argument('--adjustment', choices=('relation-only', 'linear'), default='relation-only')
    parser.add_argument('--primary-summary', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    adjusted = args.adjustment == 'linear'
    args.primary_summary = args.primary_summary or ROOT / ('results/coarsening-matched-fully-adjusted/summary.json' if adjusted else 'results/coarsening-matched/summary.json')
    args.output_dir = args.output_dir or ROOT / ('results/coarsening-confidence-sensitivity-adjusted' if adjusted else 'results/coarsening-confidence-sensitivity')
    rows = read_rows(args.pairs)
    primary = json.loads(args.primary_summary.read_text())
    grouped = defaultdict(list)
    for row in rows:
        assert row['source'] in {'combined', 'judge'}
        assert row['coarser'] in (0, 1) and math.isfinite(row['base_logprob'])
        grouped[row['source'], row['dataset'], row['model']].append(row)
    assert len(grouped) == 24
    results = []
    group_metadata = []
    for dataset in DATASETS:
        for model in MODELS:
            paired = {}
            for source in ('combined', 'judge'):
                group = grouped[source, dataset, model]
                keyed = {(r['source_id'], r['qid'], r['question']): r for r in group}
                assert len(keyed) == len(group), 'Duplicate matched observation'
                paired[source] = keyed
            assert paired['combined'].keys() == paired['judge'].keys(), 'Different metric cohorts'
            for key, row in paired['combined'].items():
                other = paired['judge'][key]
                assert all(row[k] == other[k] for k in ('base_logprob', 'relation', 'q25', 'q75', 'low_confidence'))
            by_relation = defaultdict(list)
            for row in paired['combined'].values():
                by_relation[row['relation']].append(row['base_logprob'])
            scales = {r: (float(np.mean(v)), float(np.std(v, ddof=0))) for r, v in by_relation.items()}
            cutoffs = {tail: {r: tuple(map(float, np.quantile(v, [tail, 1-tail])))
                             for r, v in by_relation.items()} for tail in TAILS}
            for relation, values in by_relation.items():
                group_metadata.append({'model': model, 'dataset': dataset, 'relation': relation,
                    'n': len(values), 'mean': scales[relation][0], 'population_sd': scales[relation][1],
                    'cutoffs': {str(t): list(cutoffs[t][relation]) for t in TAILS}})
            for source in ('combined', 'judge'):
                group = list(paired[source].values())
                record = {'source': source, 'dataset': dataset, 'model': model, 'eligible_pairs': len(group)}
                for tail in TAILS:
                    selected = []
                    for row in group:
                        lo, hi = cutoffs[tail][row['relation']]
                        low = (1 if row['base_logprob'] <= lo else 0 if row['base_logprob'] >= hi else None) if lo < hi else None
                        if tail == .25:
                            assert np.allclose([lo, hi], [row['q25'], row['q75']], atol=1e-12, rtol=0)
                            assert low == row['low_confidence'], 'Primary tail assignment changed'
                        selected.append({**row, 'low_confidence': low})
                    fit = logistic(selected, adjustment=args.adjustment)
                    fit['effect'] = f'bottom versus top {100*tail:g}% of eligible confidence within relation'
                    fit['tail_fraction'] = tail
                    fit['low_pairs'] = sum(r['low_confidence'] == 1 for r in selected)
                    fit['high_pairs'] = sum(r['low_confidence'] == 0 for r in selected)
                    fit['low_coarser_rate'] = sum(r['coarser'] for r in selected if r['low_confidence'] == 1)/fit['low_pairs'] if fit['low_pairs'] else None
                    fit['high_coarser_rate'] = sum(r['coarser'] for r in selected if r['low_confidence'] == 0)/fit['high_pairs'] if fit['high_pairs'] else None
                    if tail == .25:
                        saved = primary[source]['by_dataset'][dataset]['by_model'][model]['coarser_vs_not_coarser']
                        assert fit['status'] == saved['status']
                        if fit['status'] == 'estimated':
                            for key in ('odds_ratio', 'odds_ratio_ci_lower', 'odds_ratio_ci_upper'):
                                assert math.isclose(fit[key], saved[key], rel_tol=1e-8, abs_tol=1e-10), f'Primary estimate mismatch: {key}'
                        assert fit['cases'] == saved['cases']
                    record[f'tails_{int(100*tail)}'] = fit
                record['continuous'] = continuous(group, scales, args.adjustment)
                results.append(record)
            print(f'Completed {dataset} / {model}: both measures, five specifications', flush=True)
    specs = [f'tails_{int(100*t)}' for t in TAILS]+['continuous']
    counts = {source: {spec: {
        'estimated': sum(r[spec]['status'] == 'estimated' for r in results if r['source'] == source),
        'positive': sum(r[spec]['status'] == 'estimated' and r[spec]['odds_ratio'] > 1 for r in results if r['source'] == source),
        'ci_above_one': sum(r[spec]['status'] == 'estimated' and r[spec]['odds_ratio_ci_lower'] > 1 for r in results if r['source'] == source),
    } for spec in specs} for source in ('combined', 'judge')}
    metadata = {
        'input': str(args.pairs.relative_to(ROOT.parent)) if args.pairs.is_relative_to(ROOT.parent) else str(args.pairs),
        'input_sha256': file_hash(args.pairs), 'primary_summary_sha256': file_hash(args.primary_summary),
        'population': 'Identical saved judge-orderable eligible pairs for both measures; includes middle-confidence pairs.',
        'adjustment': args.adjustment,
        'starting_controls': ['base_granuscore', 'base_words'] if adjusted else [],
        'uncertainty': '95% Wald intervals, same qid-cluster-robust covariance and finite-sample correction as primary.',
        'continuous_scaling': 'Negative confidence z-score within model/dataset/relation, using eligible-pair population SD (ddof=0). Zero-SD groups excluded.',
        'continuous_assumption': 'Linear confidence effect on log odds; not a test of non-linear confidence effects.',
        'tail_ties': 'Include values at cutoff; exclude groups whose lower and upper cutoffs coincide.',
        'validation': 'All 24 quartile fits and their case assignments reproduce the saved '+('base-answer-adjusted' if adjusted else 'primary')+' estimates and intervals.',
    }
    write(args.output_dir/'summary.json', {'metadata': metadata, 'counts': counts, 'results': results})
    write(args.output_dir/'groups.json', group_metadata)
    lines = ['# Confidence specification sensitivity', '',
             'Odds ratios [95% cluster-robust CI]. All tail comparisons use the same eligible population; only the retained extremes change.',
             'Continuous: all eligible pairs with nonzero within-group confidence variance; OR per one within-relation SD decrease in confidence.',
             'Continuous ORs have a different unit and should not be compared numerically with tail-comparison ORs.', '',
             'The 25% column reproduces the saved '+('base-answer-adjusted' if adjusted else 'primary')+' analysis, including all 24 estimates and intervals.',
             'Controls: relation/category'+(' plus base GranuScore and base word count.' if adjusted else ' only.'), '']
    for source in ('combined', 'judge'):
        lines += [f'## {source.capitalize()}', '',
                  '| Dataset / model | 20% tails | 25% tails | 30% tails | 35% tails | Continuous (1 SD lower) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for record in results:
            if record['source'] == source:
                lines.append('| '+record['dataset']+' / '+NAMES[record['model']]+' | '+' | '.join(format_estimate(record[s]) for s in specs)+' |')
        lines += ['', 'CI entirely above one: '+', '.join(f"{spec}: {counts[source][spec]['ci_above_one']}/12" for spec in specs)+'.', '']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    write_table(results, args.output_dir, args.adjustment)
    print(json.dumps(counts, indent=2), flush=True)
    print(args.output_dir/'summary.md', flush=True)


if __name__ == '__main__':
    main()
