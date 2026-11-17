"""Per-relation and leave-one-relation-out confidence/coarsening checks.

uv run --locked --extra analysis python -m generalization.relation_sensitivity
Uses the fixed eligible cohort and confidence assignments from the paper analysis.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from shared.io import read_rows, write, file_hash
from .coarsening import logistic
from .confidence_sensitivity import continuous, format_estimate, MODELS, DATASETS, NAMES

ROOT = Path(__file__).resolve().parent
RELATION_NAMES = {'P127': 'Owner', 'P26': 'Spouse', 'P264': 'Record label', 'P50': 'Author'}


def rates(rows):
    result = {'eligible_pairs': len(rows)}
    for label, indicator in [('low', 1), ('high', 0)]:
        selected = [r for r in rows if r['low_confidence'] == indicator]
        result[label+'_pairs'] = len(selected)
        result[label+'_coarser'] = sum(r['coarser'] for r in selected)
        result[label+'_rate'] = result[label+'_coarser']/len(selected) if selected else None
    result['rate_difference_pp'] = 100*(result['low_rate']-result['high_rate']) if result['low_rate'] is not None and result['high_rate'] is not None else None
    return result


def analyze(rows, scales):
    summary = rates(rows)
    single_relation = len({r['relation'] for r in rows}) == 1
    extremes = [r for r in rows if r['low_confidence'] is not None]
    cells = [sum(r['low_confidence'] == confidence and r['coarser'] == outcome
                 for r in extremes) for confidence in (0, 1) for outcome in (0, 1)]
    if single_relation and len(extremes) >= 40 and min(cells) == 0:
        quartiles = {'cases': len(extremes), 'status': 'not_estimable',
                     'reason': 'zero contingency cell: ordinary subgroup logit has an unbounded estimate',
                     'contingency_cells_high0_high1_low0_low1': cells}
    else:
        quartiles = logistic(rows, adjustment='relation-only')
    lower = [r['base_logprob'] for r in rows if r['coarser'] == 1]
    other = [r['base_logprob'] for r in rows if r['coarser'] == 0]
    separated = (single_relation and lower and other and
                 (min(lower) >= max(other) or min(other) >= max(lower)))
    if separated:
        continuous_fit = {'cases': len(rows), 'status': 'not_estimable',
                          'reason': 'separated one-dimensional subgroup confidence values'}
    else:
        continuous_fit = continuous(rows, scales)
    return {**summary, 'quartiles': quartiles, 'continuous': continuous_fit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, default=ROOT/'results/coarsening-matched/pairs.jsonl')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results/coarsening-by-relation')
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in read_rows(args.pairs):
        groups[row['source'], row['dataset'], row['model']].append(row)
    assert len(groups) == 24
    reports = []
    for dataset in DATASETS:
        for model in MODELS:
            first = groups['combined', dataset, model]
            evidence = defaultdict(list)
            for row in first:
                evidence[row['relation']].append(row['base_logprob'])
            scales = {r: (float(np.mean(v)), float(np.std(v, ddof=0))) for r, v in evidence.items()}
            relations = sorted(evidence)
            for source in ('combined', 'judge'):
                rows = groups[source, dataset, model]
                record = {'source': source, 'dataset': dataset, 'model': model,
                          'overall': analyze(rows, scales), 'by_relation': {}, 'leave_one_out': {}}
                for relation in relations:
                    subset = [r for r in rows if r['relation'] == relation]
                    retained = [r for r in rows if r['relation'] != relation]
                    record['by_relation'][relation] = analyze(subset, scales)
                    record['leave_one_out'][relation] = analyze(retained, scales)
                reports.append(record)
            print(f'Completed {dataset} / {model}', flush=True)
    # All overall fits must reproduce the previously verified sensitivity analysis.
    baseline = json.loads((ROOT/'results/coarsening-confidence-sensitivity/summary.json').read_text())
    indexed = {(r['source'], r['dataset'], r['model']): r for r in baseline['results']}
    for r in reports:
        for key, original in [('quartiles', 'tails_25'), ('continuous', 'continuous')]:
            a, b = r['overall'][key], indexed[r['source'], r['dataset'], r['model']][original]
            assert a['status'] == b['status'] and a['cases'] == b['cases']
            if a['status'] == 'estimated':
                for field in ('odds_ratio', 'odds_ratio_ci_lower', 'odds_ratio_ci_upper'):
                    assert math.isclose(a[field], b[field], rel_tol=1e-8, abs_tol=1e-10)
    counts = {}
    for source in ('combined', 'judge'):
        counts[source] = {}
        for scope in ('by_relation', 'leave_one_out'):
            counts[source][scope] = {}
            for spec in ('quartiles', 'continuous'):
                values = [v[spec] for r in reports if r['source'] == source for v in r[scope].values()]
                counts[source][scope][spec] = {
                    'total': len(values),
                    'estimated': sum(v['status']=='estimated' for v in values),
                    'positive': sum(v['status']=='estimated' and v['odds_ratio']>1 for v in values),
                    'ci_above_one': sum(v['status']=='estimated' and v['odds_ratio_ci_lower']>1 for v in values),
                }
    write(args.output_dir/'summary.json', {'metadata': {
        'input_sha256': file_hash(args.pairs),
        'analysis': 'Per-model per-relation estimates and leave-one-relation-out refits on the fixed judge-orderable cohort.',
        'scopes': 'GranolaEQ relations; SimpleQA topics; TruthfulQA seven grouped domains.',
        'quartiles': 'Original within-model/dataset/relation confidence cutoffs, unchanged after excluding another relation.',
        'continuous': 'Negative within-group standardized confidence; same eligible-group population scales as the preceding sensitivity analysis.',
        'inference': 'Unadjusted 95% qid-cluster-robust Wald confidence intervals; exploratory subgroup checks, no multiplicity correction.',
        'small_groups': 'At least 40 pairs, outcome/confidence variation, no separation or rank deficiency. Single-relation quartile fits also require all four contingency cells to be nonzero; continuous subgroup fits require overlapping confidence ranges across outcomes.',
        'validation': 'All 48 overall fits reproduce previously verified estimates and confidence intervals.',
    }, 'counts': counts, 'results': reports})
    lines = ['# Coarsening by relation and model', '',
        'All analyses use the same matched judge-orderable cohort. Quartile comparisons retain the original within-relation cutoffs.',
        'Continuous confidence uses all eligible observations; OR per one within-relation SD lower confidence.',
        'GranolaEQ rows are relations, SimpleQA rows topics, and TruthfulQA rows the seven grouped domains.',
        'Intervals are exploratory, unadjusted 95% cluster-robust Wald intervals. Small or separated groups may be not estimable.', '']
    for source in ('combined', 'judge'):
        lines += [f'## {source.capitalize()}: individual relations', '',
                  '| Dataset / model / relation | Eligible | Low / high N | Low / high coarser (%) | Difference (pp) | Quartile OR [95% CI] | Continuous OR [95% CI] |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in reports:
            if r['source'] != source:
                continue
            for relation, v in r['by_relation'].items():
                rates_text = ' / '.join('—' if v[k] is None else f'{100*v[k]:.1f}' for k in ('low_rate','high_rate'))
                diff = '—' if v['rate_difference_pp'] is None else f"{v['rate_difference_pp']:.1f}"
                lines.append(f"| {r['dataset']} / {NAMES[r['model']]} / {RELATION_NAMES.get(relation, relation)} | {v['eligible_pairs']} | {v['low_pairs']} / {v['high_pairs']} | {rates_text} | {diff} | {format_estimate(v['quartiles'])} | {format_estimate(v['continuous'])} |")
        lines += ['', f'## {source.capitalize()}: leave-one-relation-out', '',
                  '| Dataset / model | Overall quartile OR | Range after omitting one relation | Positive / estimated | CI above one / estimated | Overall continuous OR | Continuous range after omission | CI above one / estimated |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in reports:
            if r['source'] != source:
                continue
            summary = {}
            for spec in ('quartiles','continuous'):
                values = [v[spec] for v in r['leave_one_out'].values() if v[spec]['status']=='estimated']
                summary[spec] = {'range': f"{min(v['odds_ratio'] for v in values):.2f}–{max(v['odds_ratio'] for v in values):.2f}" if values else '—',
                                 'positive': sum(v['odds_ratio']>1 for v in values),
                                 'ci': sum(v['odds_ratio_ci_lower']>1 for v in values), 'n': len(values)}
            q,c=summary['quartiles'],summary['continuous']
            lines.append(f"| {r['dataset']} / {NAMES[r['model']]} | {format_estimate(r['overall']['quartiles'])} | {q['range']} | {q['positive']}/{q['n']} | {q['ci']}/{q['n']} | {format_estimate(r['overall']['continuous'])} | {c['range']} | {c['ci']}/{c['n']} |")
        lines += ['']
    (args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(counts, indent=2), flush=True)
    print(args.output_dir/'summary.md', flush=True)


if __name__ == '__main__':
    main()
