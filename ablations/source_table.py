"""Export LFM source-ablation results from saved test outcomes and judgments.

Run: uv run --locked python -m ablations.source_table
Vital Persons uses refreshed scoring from ablations.source_evaluation.
"""
from collections import defaultdict
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'ablations/lfm/sources/results'
CONDITIONS = ('consistency-only', 'recognition-only', 'mixed')
LABELS = {'base': 'Base', 'consistency-only': 'Consistency only',
          'recognition-only': 'Recognition only', 'mixed': 'Both (main)'}


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def main():
    grouped = defaultdict(dict)
    for row in read_rows(ROOT / 'ablations/lfm/sources/correctness/predictions.jsonl'):
        if row['answer_scope'] != 'first_sentence':
            continue
        key = row['source_id'], row['question']
        if row['condition'] in grouped[key]:
            raise ValueError(f'Duplicate correctness judgment: {key}')
        grouped[key][row['condition']] = row
    for group in grouped.values():
        assert set(group) == set(CONDITIONS)
        assert len({r['base_response'] for r in group.values()}) == 1
        assert len({r['judge_base_correct'] for r in group.values()}) == 1
        assert len({json.dumps(r['reference_answers']) for r in group.values()}) == 1
    common = [g for g in grouped.values() if all(
        r[field] != 'Cannot determine' for r in g.values()
        for field in ('judge_base_correct', 'judge_finetuned_correct'))]
    if not common:
        raise ValueError('Empty common correctness cohort')

    results = {}
    gold = {}
    for condition in ('base', *CONDITIONS):
        directory = OUT / 'rescored' / condition
        variant = 'base_plain' if condition == 'base' else 'finetuned'
        record = {}
        for modality in ('temporal', 'location'):
            rows = read_rows(directory / 'outcomes' / modality / f'{variant}.jsonl')
            by_case = {r['case_id']: r['gold'] for r in rows}
            assert len(by_case) == len(rows)
            if modality in gold:
                assert by_case == gold[modality], 'Different test cases or gold references'
            else:
                gold[modality] = by_case
            eligible = [r for r in rows if r['gold'] is not None]
            record[modality] = {
                'n': len(eligible),
                'accuracy': 100 * sum(r['truth_correct'] for r in eligible) / len(eligible),
                'utility': 100 * sum(r['utility_uniform_wrong_penalty'] for r in eligible) / len(eligible),
            }
        labels = [g['mixed']['judge_base_correct'] if condition == 'base'
                  else g[condition]['judge_finetuned_correct'] for g in common]
        record['granola'] = {'n': len(labels),
            'accuracy': 100 * labels.count('Correct') / len(labels),
            'abstention': 100 * labels.count('Abstain') / len(labels)}
        results[condition] = record

    columns = [('temporal', 'accuracy'), ('location', 'accuracy'),
               ('temporal', 'utility'), ('location', 'utility')]
    best = [max(r[m][metric] for r in results.values()) for m, metric in columns]
    lines = [r'\begin{table}[t]', r'\small', r'\centering',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrr@{}}',
        r'\toprule',
        r'& \multicolumn{2}{c}{Accuracy $\uparrow$} & \multicolumn{2}{c}{Utility $\uparrow$} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'Supervision & Temp. & Geo. & Temp. & Geo. \\', r'\midrule']
    for condition, record in results.items():
        values = [record[m][metric] for m, metric in columns]
        cells = [rf'\textbf{{{v:.2f}}}' if f'{v:.2f}' == f'{maximum:.2f}' else f'{v:.2f}'
                 for v, maximum in zip(values, best)]
        lines.append(LABELS[condition] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
        r'\caption{LFM supervision-source ablations on the Vital Persons',
        r'test set (accuracy in \%; utility $\times100$).',
        r'Consistency-only and mixed-source training use up to 3,000 steps;',
        r'recognition-only uses 20 epochs. Checkpoints are selected by',
        r'Vital Persons development loss. Bold indicates the best value per column.}',
        r'\label{tab:source-ablations}', r'\end{table}']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'table.tex').write_text('\n'.join(lines) + '\n')
    summary = {'results': results, 'granola_excluded': len(grouped)-len(common),
        'vital_persons_status': 'refreshed with reviewed gold, corrections and the production no-answer judge',
        'training_budgets': 'Consistency-only and main: max 3000 steps; recognition-only: 20 epochs. All select on Vital Persons dev loss.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
