"""Post-hoc consensus-target sensitivity; no model training or generation.

Run from the repository root:
    uv run --locked python -m ablations.consensus_threshold
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from evaluation.vital_persons.data import load_cases
from evaluation.vital_persons.cases import VITAL
from evaluation.vital_persons.consensus_utility import values_for, infer, score_target
from shared.io import file_hash, write

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'lfm': 'LFM', 'ministral': 'Ministral', 'qwen': 'Qwen', 'gemma': 'Gemma'}
THRESHOLDS = (0.7, 0.8, 0.9, 1.0)
DOMAINS = {'temporal': 'Temporal', 'location': 'Geographic'}


def summarize(rows):
    n = len(rows)
    gold = [r for r in rows if r['gold_available']]
    evaluable = [r for r in gold if r['status'] != 'unavailable']
    known = [r for r in evaluable if r['status'] == 'known']
    mean = lambda values: sum(values) / len(values) if values else None
    return {
        'n': n, 'statuses': {s: sum(r['status'] == s for r in rows)
                           for s in ('known', 'idk', 'unavailable')},
        'status_percentages': {s: 100 * sum(r['status'] == s for r in rows) / n if n else None
                              for s in ('known', 'idk', 'unavailable')},
        'target_levels': dict(Counter(r['target']['level'] for r in rows if r['target'])),
        'gold_available_n': len(gold), 'scored_n': len(evaluable), 'known_with_gold_n': len(known),
        'coverage_percent': 100 * len(evaluable) / len(gold) if gold else None,
        'accuracy_percent': mean([100 * r['correct'] for r in evaluable]),
        'accuracy_among_known_percent': mean([100 * r['correct'] for r in known]),
        'utility_x100': mean([100 * r['utility'] for r in evaluable]),
    }


def run(output):
    report = {
        'split': 'dev', 'thresholds': THRESHOLDS, 'inputs_sha256': {}, 'models': {},
        'definition': 'Target construction only, not fine-tuned model performance. Each model uses its own frozen development partition. The table includes all gold-available questions. Coverage is (known + IDK) / gold-available N; utility is averaged only over covered questions, with IDK scored zero. Unavailable targets are excluded from utility, never treated as IDK. Covered questions can change across thresholds. Accuracy includes IDK as non-correct; conditional accuracy among known targets is reported separately.',
    }
    details = []
    for model, label in MODELS.items():
        directory = VITAL / 'datasets' / model
        path = directory / 'consensus.json'
        records = json.loads(path.read_text())
        index = {(r['qid'], r['relation']): r for r in records}
        assert len(index) == len(records) == 6560
        report['inputs_sha256'][str(path.relative_to(ROOT))] = file_hash(path)
        for filename in ('case_targets.jsonl', 'split_assignments.json'):
            report['inputs_sha256'][str((directory / filename).relative_to(ROOT))] = file_hash(directory / filename)
        # Reconstruct all frozen 80% targets, not just the selected dev cohort.
        for record in records:
            modality = 'temporal' if record['relation'] in {'birth', 'death'} else 'location'
            values = values_for(record, modality)
            assert sum(v is not None for v in values) == record['available_values']
            target, status = infer(values, modality, 0.8)
            if status != record['target_status'] or (json.loads(json.dumps(asdict(target))) if target else None) != record['target']:
                raise ValueError(f'Frozen 80% target mismatch: {model}/{record["qid"]}/{record["relation"]}')
        block = {'label': label, 'reconstructed80_n': len(records), 'domains': {}}
        for modality, cases in load_cases(model, 'dev').items():
            all_rows = {str(t): [] for t in THRESHOLDS}
            gold_rows = {str(t): [] for t in THRESHOLDS}
            gold_ids = [case.case_id for case in cases if case.gold is not None]
            for case in cases:
                record = index[case.qid, case.relation]
                values = values_for(record, modality)
                for threshold in THRESHOLDS:
                    target, status = infer(values, modality, threshold)
                    utility = score_target(target, status, case.gold, modality) if case.gold is not None else None
                    correct = (bool(utility is not None and utility >= 0 and status == 'known')
                               if case.gold is not None and status != 'unavailable' else None)
                    row = {'model': model, 'modality': modality, 'case_id': case.case_id,
                           'qid': case.qid, 'relation': case.relation, 'threshold': threshold,
                           'usable_values': record['available_values'], 'status': status,
                           'target': json.loads(json.dumps(asdict(target))) if target else None,
                           'gold_available': case.gold is not None,
                           'correct': correct, 'utility': utility}
                    all_rows[str(threshold)].append(row)
                    if case.gold is not None:
                        gold_rows[str(threshold)].append(row)
                    details.append(row)
            block['domains'][modality] = {
                'development_n': len(cases), 'gold_available_n': len(gold_ids),
                'gold_available_case_ids': gold_ids,
                'gold_available': {t: summarize(rows) for t, rows in gold_rows.items()},
                'full_development': {t: summarize(rows) for t, rows in all_rows.items()},
            }
        report['models'][model] = block
    report['overall'] = overall_means(report)
    report['overall_definition'] = 'Arithmetic means with equal weight for each of the four models and each of the two domains; utility remains conditional on available targets.'
    metadata = ROOT / 'data_preparation/vital_persons/metadata/gold_adapted.jsonl'
    report['inputs_sha256'][str(metadata.relative_to(ROOT))] = file_hash(metadata)
    output.mkdir(parents=True, exist_ok=True)
    write(output / 'summary.json', report)
    write(output / 'targets.jsonl', details, rows=True)
    export_table(report, output)
    export_markdown(report, output)
    print(output / 'table.tex')
    print((output / 'summary.md').read_text())


def overall_means(report):
    overall = {}
    for threshold in THRESHOLDS:
        cells = [report['models'][model]['domains'][domain]['gold_available'][str(threshold)]
                 for model in MODELS for domain in DOMAINS]
        if any(cell['utility_x100'] is None or cell['coverage_percent'] is None for cell in cells):
            raise ValueError('Overall mean requires results for every model/domain')
        overall[str(threshold)] = {
            field: sum(cell[field] for cell in cells) / len(cells)
            for field in ('utility_x100', 'coverage_percent')}
    return overall


def export_table(report, output):
    lines = [r'\begin{table*}[t]', r'\small', r'\centering',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrrrrrr@{}}',
             r'\toprule',
             r'& \multicolumn{2}{c}{70\%} & \multicolumn{2}{c}{80\%} & \multicolumn{2}{c}{90\%} & \multicolumn{2}{c}{100\%} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
             r'Model ($N$) & Utility & Cov. & Utility & Cov. & Utility & Cov. & Utility & Cov. \\']
    for domain, label in DOMAINS.items():
        lines += [r'\midrule', rf'\multicolumn{{9}}{{@{{}}l}}{{\textit{{{label}}}}} \\',
                  r'\addlinespace[2pt]']
        for model, name in MODELS.items():
            block = report['models'][model]['domains'][domain]
            cells = []
            for t in THRESHOLDS:
                stats = block['gold_available'][str(t)]
                utility, coverage = stats['utility_x100'], stats['coverage_percent']
                cells += ['---' if utility is None else f'{utility:.2f}',
                          '---' if coverage is None else f'{coverage:.1f}']
            lines.append(f'{name} ({block["gold_available_n"]}) & ' + ' & '.join(cells) + r' \\')
    overall = report['overall']
    cells = []
    for threshold in THRESHOLDS:
        cell = overall[str(threshold)]
        value = f'{cell["utility_x100"]:.2f}'
        cells.extend([value, f'{cell["coverage_percent"]:.1f}'])
    lines.extend([r'\midrule', 'Overall & ' + ' & '.join(cells) + r' \\'])
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Consensus-target utility ($\times100$) and coverage (\%) on',
              r'model-specific Vital Persons development sets ($N$ gold-available',
              r'questions). Coverage includes common-value and IDK targets;',
              r'utility averages over covered questions, which vary by threshold.',
              r'Overall is the macro-average across the eight model--domain results.}',
              r'\label{tab:consensus-threshold-sensitivity}', r'\end{table*}']
    (output / 'table.tex').write_text('\n'.join(lines) + '\n')


def export_markdown(report, output):
    lines = ['# Consensus-threshold sensitivity', '', report['definition'], '',
             'All 26,240 original 80% targets are reconstructed exactly.', '']
    lines += ['## Overall mean', '', report['overall_definition'], '',
              '| Threshold | Utility ×100 | Coverage (%) |', '| --- | ---: | ---: |']
    for t in THRESHOLDS:
        cell = report['overall'][str(t)]
        lines.append(f'| {t:.0%} | {cell["utility_x100"]:.2f} | {cell["coverage_percent"]:.1f} |')
    lines.append('')
    for domain, label in DOMAINS.items():
        lines += [f'## {label}: all gold-available development questions', '',
                  'Each cell is utility ×100 / target coverage (%). Utility denominators vary; N is the full gold-available denominator.', '',
                  '| Model | N | 70% | 80% | 90% | 100% |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model, name in MODELS.items():
            block = report['models'][model]['domains'][domain]
            cells = []
            for t in THRESHOLDS:
                stats = block['gold_available'][str(t)]
                u, c = stats['utility_x100'], stats['coverage_percent']
                cells.append(('—' if u is None else f'{u:.2f}') + ' / ' + ('—' if c is None else f'{c:.1f}'))
            lines.append(f'| {name} | {block["gold_available_n"]} | ' + ' | '.join(cells) + ' |')
        lines += ['', f'## {label}: gold-available target counts and accuracy', '',
                  '| Model | Threshold | N | Common value | IDK | Unavailable | Scored N | Accuracy (%) | Utility ×100 | Target levels (counts) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for model, name in MODELS.items():
            for t in THRESHOLDS:
                stats = report['models'][model]['domains'][domain]['gold_available'][str(t)]
                counts = stats['statuses']
                fmt = lambda value: '—' if value is None else f'{value:.2f}'
                lines.append(f'| {name} | {t:.0%} | {stats["n"]} | {counts["known"]} | {counts["idk"]} | {counts["unavailable"]} | {stats["scored_n"]} | {fmt(stats["accuracy_percent"])} | {fmt(stats["utility_x100"])} | {stats["target_levels"]} |')
        lines += ['', f'## {label}: full development coverage (including questions without gold)', '',
                  '| Model | Threshold | N | Common value | IDK | Unavailable |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for model, name in MODELS.items():
            for t in THRESHOLDS:
                stats = report['models'][model]['domains'][domain]['full_development'][str(t)]
                counts = stats['statuses']
                lines.append(f'| {name} | {t:.0%} | {stats["n"]} | {counts["known"]} | {counts["idk"]} | {counts["unavailable"]} |')
        lines.append('')
    (output / 'summary.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'ablations/consensus-threshold/results/dev')
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == '__main__':
    main()
