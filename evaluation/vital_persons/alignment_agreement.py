"""Correctness-aware target agreement, conditional on correct answers."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from .plot_alignment import collect_comparison
from .table import (ROOT, MODELS, LEVELS, analyze, project_prediction,
                    temporal_correct, location_correct, temporal_value)
from .cases import temporal_level_for_taxonomy, TemporalTaxonomy
from parsing.location.normalization.values import LocationValue
from shared.io import read_rows, write


def control_row(row, modality, policy):
    prediction, status = project_prediction(row, modality, policy)
    result = dict(row, parse_status=status)
    if prediction is not None:
        result['prediction_level_for_target_metrics'] = (
            temporal_level_for_taxonomy(prediction.level, TemporalTaxonomy.FIVE_LEVEL).value
            if modality == 'temporal' else prediction.level.value)
        gold = temporal_value(row['gold']) if modality == 'temporal' else LocationValue.from_dict(row['gold'])
        result['truth_correct'] = bool((temporal_correct if modality == 'temporal' else location_correct)(prediction, gold))
    else:
        result['truth_correct'] = False
    return result


def agreement(rows, modality):
    levels = LEVELS[modality]
    target, actual, folded = [], [], []
    outcomes = Counter()
    by_target = {level: Counter() for level in levels}
    for row in rows:
        t = levels.index(row['target_level'])
        if row['parse_status'] == 'idk':
            outcome = 'abstain'
        elif row['parse_status'] != 'parsed':
            raise ValueError('Agreement requires the common assessable cohort')
        elif not row['truth_correct']:
            outcome = 'wrong'
        else:
            p = levels.index(row.get('prediction_level_for_target_metrics') or row['prediction']['level'])
            target.append(t); actual.append(p); folded.append(max(t, p))
            outcome = 'correct_at_target_or_finer' if p <= t else 'correct_coarser'
        outcomes[outcome] += 1
        by_target[levels[t]][outcome] += 1
    def kappa(predictions, weights=None):
        # Undefined when both variables have the same single category.
        if not target or len(set(target + predictions)) < 2:
            return None
        observed = np.zeros((len(levels), len(levels)))
        for t, p in zip(target, predictions, strict=True):
            observed[t, p] += 1
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))/len(target)
        distance = np.abs(np.subtract.outer(np.arange(len(levels)), np.arange(len(levels))))
        disagreement = distance if weights == 'linear' else (distance != 0)
        denominator = float((expected * disagreement).sum())
        return 1 - float((observed * disagreement).sum())/denominator if denominator else None
    n = len(rows)
    per_level = {level: {'n': sum(counts.values()),
                  'success_rate': counts['correct_at_target_or_finer']/sum(counts.values())}
                 for level, counts in by_target.items() if counts}
    return {'n': n, 'correct_n': len(target),
            'counts': {key: outcomes[key] for key in ('correct_at_target_or_finer', 'correct_coarser', 'wrong', 'abstain')},
            'correct_at_target_or_finer_rate': outcomes['correct_at_target_or_finer']/n if n else None,
            'macro_correct_at_target_or_finer_rate': float(np.mean([v['success_rate'] for v in per_level.values()])) if per_level else None,
            'by_target': per_level,
            'correct_only_kappa': kappa(folded),
            'correct_only_linear_weighted_kappa': kappa(folded, 'linear'),
            'correct_only_unfolded_kappa': kappa(actual)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results-corrected')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    args = parser.parse_args()
    cohort = collect_comparison(args.results_dir, list(MODELS), args.split)
    tables, hashes = analyze(args.results_dir, list(MODELS), args.split)
    result = {'split': args.split, 'definition': 'On correct answers only, map finer generated levels to the target before computing unweighted and linear-weighted Cohen kappa. This asymmetric transformation measures sufficient informativeness, not exact ordinal agreement. Wrong and abstaining answers are excluded from kappa, but retained in success-rate denominators and reported separately. Method-specific correct subsets differ; do not compare kappa without coverage and all-case success.',
              'cohort': cohort['paired_cohorts'], 'inputs_sha256': hashes, 'models': {}}
    lines = ['# Correctness-aware target agreement', '',
             result['definition'], '',
             'All methods use the same known-target, gold-available cases assessable for both Base and FT. IDK targets are outside this level-alignment analysis. Best fixed cap is selected for utility on the full gold-available split, as in the main table; all individual caps are retained in summary.json.', '',
             '| Model / domain | Method | N | Correct | Correct ≥ target (%) | Macro (%) | κ (correct only) | Linear-weighted κ | Wrong | Abstain |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(value, percent=False):
        if value is None:
            return '—'
        return f'{100*value:.2f}' if percent else f'{value:.3f}'
    for model in MODELS:
        result['models'][model] = {}
        for modality in LEVELS:
            ids = set(cohort['paired_cohorts'][model][modality]['retained_case_ids'])
            directory = args.results_dir/model/args.split/'outcomes'/modality
            base = [r for r in read_rows(directory/'base_plain.jsonl') if r['case_id'] in ids]
            ft = [r for r in read_rows(directory/'finetuned.jsonl') if r['case_id'] in ids]
            policies = ['cap-'+level for level in LEVELS[modality]]
            methods = {'Base': agreement(base, modality), 'FT (ours)': agreement(ft, modality)}
            controls = {policy: agreement([control_row(r, modality, policy) for r in base], modality) for policy in policies}
            best = tables[model][modality]['best_fixed_cap_oracle']['policy']
            methods['Best fixed cap ('+best.removeprefix('cap-').lower()+')'] = controls[best]
            result['models'][model][modality] = {'methods': methods, 'fixed_caps': controls, 'selected_policy': best}
            for method, values in methods.items():
                c = values['counts']
                cells = [MODELS[model]+' / '+modality, method, str(values['n']), str(values['correct_n']),
                         number(values['correct_at_target_or_finer_rate'], True), number(values['macro_correct_at_target_or_finer_rate'], True),
                         number(values['correct_only_kappa']), number(values['correct_only_linear_weighted_kappa']), str(c['wrong']), str(c['abstain'])]
                lines.append('| '+' | '.join(cells)+' |')
    output = args.output_dir or ROOT/'results-alignment-agreement'/args.split
    write(output/'summary.json', result)
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    ft_lines = ['# FT correctness-aware linearly weighted kappa', '',
                'Correct finer answers are folded onto the target. Linear weights penalize remaining disagreements by the number of granularity steps. Kappa is conditional on correct answers, not a measure of overall correctness or proof of selective target following. Known non-IDK targets with gold; same matched assessable cohort as the Base/FT figures.', '',
                '| Model | Temporal weighted κ | Correct / eligible | Location weighted κ | Correct / eligible |',
                '| --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        cells = [MODELS[model]]
        for modality in LEVELS:
            values = result['models'][model][modality]['methods']['FT (ours)']
            cells.extend([number(values['correct_only_linear_weighted_kappa']),
                          f"{values['correct_n']} / {values['n']}"])
        ft_lines.append('| '+' | '.join(cells)+' |')
    (output/'ft-weighted.md').write_text('\n'.join(ft_lines)+'\n')
    print('\n'.join(ft_lines))
    print(f'\nSaved to {output}')


if __name__ == '__main__':
    main()
