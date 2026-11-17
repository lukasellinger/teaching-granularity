"""Paired person-cluster bootstrap intervals for FT utility differences.

uv run --locked python -m evaluation.vital_persons.utility_uncertainty
Keeps the original point estimates and cap selection unchanged. A separate
sensitivity calculation reselects the utility-maximizing cap in each resample.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from shared.io import read_rows, write
from parsing.location.normalization.values import LocationValue
from .table import (ROOT, MODELS, METHODS, LEVELS, analyze, project_prediction,
                    temporal_value, temporal_correct, location_correct,
                    temporal_informativeness, location_informativeness)


def cap_utilities(rows, modality, policy):
    values = []
    for row in rows:
        if row['gold'] is None:
            values.append(None)
            continue
        prediction, status = project_prediction(row, modality, policy)
        if status == 'idk':
            values.append(0.)
        elif prediction is None:
            values.append(-1.)
        else:
            gold = temporal_value(row['gold']) if modality == 'temporal' else LocationValue.from_dict(row['gold'])
            correct = temporal_correct(prediction, gold) if modality == 'temporal' else location_correct(prediction, gold)
            gain = temporal_informativeness if modality == 'temporal' else location_informativeness
            values.append(gain(prediction.level) if correct else -1.)
    return values


def interval(point, draws):
    low, high = np.quantile(draws, [.025, .975])
    return {'difference': 100*float(point), 'ci_lower': 100*float(low),
            'ci_upper': 100*float(high), 'excludes_zero': bool(low > 0 or high < 0)}


def analyze_uncertainty(results_dir, models, split, resamples, seed):
    points, hashes = analyze(results_dir, models, split)
    result = {}
    for model in models:
        result[model] = {}
        for modality, levels in LEVELS.items():
            directory = results_dir/model/split/'outcomes'/modality
            base = sorted(read_rows(directory/'base_plain.jsonl'), key=lambda r:r['case_id'])
            qids = sorted({r['qid'] for r in base})
            qid_index = {qid:i for i,qid in enumerate(qids)}
            raw = {}
            for method in METHODS:
                rows = sorted(read_rows(directory/f'{method}.jsonl'), key=lambda r:r['case_id'])
                assert [(r['case_id'],r['qid'],r['gold']) for r in rows] == [(r['case_id'],r['qid'],r['gold']) for r in base]
                raw[method] = [r['utility_uniform_wrong_penalty'] for r in rows]
            policies = [f'cap-{level}' for level in levels]
            raw.update({policy:cap_utilities(base, modality, policy) for policy in policies})
            names = list(raw)
            mask = np.array([r['gold'] is not None for r in base])
            totals = np.zeros((len(qids), len(names)))
            counts = np.zeros(len(qids))
            for j,name in enumerate(names):
                assert np.array_equal([v is not None for v in raw[name]], mask)
                for row,value in zip(base,raw[name],strict=True):
                    if value is not None:
                        totals[qid_index[row['qid']],j] += value
                expected = (points[model][modality]['methods'][name] if name in METHODS
                            else points[model][modality]['controls'][name])['uniform_error_utility']
                assert np.isclose(totals[:,j].sum()/mask.sum(), expected, rtol=0, atol=1e-12)
            for row,eligible in zip(base,mask,strict=True):
                counts[qid_index[row['qid']]] += int(eligible)
            means = totals.sum(axis=0)/counts.sum()
            # Resample people, retaining both event questions for every drawn person.
            rng = np.random.default_rng(seed)
            sampled = rng.integers(0,len(qids),size=(resamples,len(qids)))
            denominator = counts[sampled].sum(axis=1)
            valid = denominator > 0
            boot = totals[sampled[valid]].sum(axis=1)/denominator[valid,None]
            ft_index = names.index('finetuned')
            comparisons = {method:interval(means[ft_index]-means[names.index(method)],
                            boot[:,ft_index]-boot[:,names.index(method)])
                           for method in METHODS if method != 'finetuned'}
            cap_indices = [names.index(policy) for policy in policies]
            chosen = points[model][modality]['best_fixed_cap_oracle']['policy']
            chosen_index = names.index(chosen)
            cap_point = means[ft_index]-means[chosen_index]
            comparisons['best_fixed_cap'] = interval(cap_point,boot[:,ft_index]-boot[:,chosen_index])
            selected = np.argmax(boot[:,cap_indices],axis=1)
            comparisons['best_cap_reselected'] = interval(cap_point,boot[:,ft_index]-boot[:,cap_indices].max(axis=1))
            result[model][modality] = {'gold_questions':int(counts.sum()), 'person_clusters':len(qids),
                'persons_with_gold':int((counts>0).sum()), 'valid_resamples':int(valid.sum()),
                'selected_cap':chosen, 'comparisons':comparisons,
                'bootstrap_cap_selection_counts':dict(Counter(policies[i] for i in selected))}
    return {'method':'paired_person_cluster_percentile', 'confidence_level':.95,
        'num_resamples':resamples, 'seed':seed, 'split':split,
        'interval_scope':'pointwise; no multiple-comparison adjustment',
        'cap_interpretation':'Primary intervals hold the test-selected cap fixed; sensitivity reselects the best cap in every resample.',
        'inputs_sha256':hashes, 'models':result}


def shown(record):
    return f"{record['difference']:+.2f} [{record['ci_lower']:+.2f}, {record['ci_upper']:+.2f}]"


def render(report, output):
    lines = ['# FT utility differences with 95% person-cluster bootstrap intervals', '',
        f"{report['num_resamples']:,} paired resamples; seed {report['seed']}. All values use utility ×100.",
        'Each resampled person retains both event questions. Gold-available questions are the same for every method.',
        'Intervals are pointwise. The fixed-cap interval conditions on the cap chosen on the original test set; the reselected-cap column is a selection sensitivity analysis.', '',
        '| Model | Domain | Cap | FT − fixed cap [95% CI] | FT − reselected cap [95% CI] |',
        '| --- | --- | --- | ---: | ---: |']
    tex = [r'\begin{table}[t]',r'\small',r'\centering',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}llrrr@{}}',r'\toprule',
        r'Model & Domain & $\Delta$ Utility & \multicolumn{2}{c}{95\% CI} \\',r'\midrule']
    for model,blocks in report['models'].items():
        for domain,block in blocks.items():
            record = block['comparisons']['best_fixed_cap']
            sensitivity = block['comparisons']['best_cap_reselected']
            lines.append(f"| {MODELS[model]} | {domain} | {block['selected_cap']} | {shown(record)} | {shown(sensitivity)} |")
            tex.append(f"{MODELS[model]} & {'Temp.' if domain=='temporal' else 'Geo.'} & {record['difference']:+.2f} & {record['ci_lower']:+.2f} & {record['ci_upper']:+.2f}"+r' \\')
    lines += ['', '## All baseline comparisons', '',
        '| Model | Domain | Comparator | FT − comparator [95% CI] |', '| --- | --- | --- | ---: |']
    for model,blocks in report['models'].items():
        for domain,block in blocks.items():
            for comparator in METHODS:
                if comparator == 'finetuned':continue
                lines.append(f"| {MODELS[model]} | {domain} | {METHODS[comparator]} | {shown(block['comparisons'][comparator])} |")
    tex += [r'\bottomrule',r'\end{tabular}',
        r'\caption{FT minus the test-selected best fixed cap in utility',
        rf'($\times100$). Pointwise 95\% percentile intervals use {report["num_resamples"]:,} paired',
        r'person-cluster bootstrap resamples, retaining birth and death questions',
        r'together. The selected cap is held fixed across resamples.}',
        r'\label{tab:fixed-cap-uncertainty}',r'\end{table}']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    (output/'table.tex').write_text('\n'.join(tex)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=ROOT/'results-corrected')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--split',choices=('dev','test'),default='test')
    parser.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    parser.add_argument('--resamples',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.resamples<1 or len(set(args.models))!=len(args.models):parser.error('Use a positive resample count and distinct models')
    report = analyze_uncertainty(args.results_dir,args.models,args.split,args.resamples,args.seed)
    output = args.output_dir or ROOT/'results-utility-uncertainty'/args.split
    write(output/'summary.json',report)
    render(report,output)
    print((output/'summary.md').read_text().split('## All baseline comparisons')[0])


if __name__ == '__main__':
    main()
