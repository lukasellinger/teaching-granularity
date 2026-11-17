"""Refresh and export LFM objective, contrastive-weight, balancing and IDK-weight ablations.

uv run --locked python -m ablations.objective_table prepare
uv run --locked python -m evaluation.vital_persons.answer_status.judge run \
  --items ablations/lfm/short/results/objectives/answer-status/items.jsonl \
  --output-dir ablations/lfm/short/results/objectives/answer-status --allow-api
uv run --locked python -m ablations.objective_table score
uv run --locked python -m ablations.objective_table table
"""
import argparse
import json
from pathlib import Path
from shared.io import read_rows, write
from ablations.source_evaluation import refresh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ablations/lfm/short'
OUT = BASE / 'results/objectives'
CONDITIONS = (
    ('contrastive', 'Contrastive'),
    ('sft', r'\quad + $w_{\mathrm{IDK}}=1$'),
    ('sft-idk-0.5', r'\quad + $w_{\mathrm{IDK}}=0.5$'),
    ('sft-idk-0', r'\quad + $w_{\mathrm{IDK}}=0$'),
    ('combined-weight-1', r'\quad + $\lambda=1$'),
    ('combined-weight-0.5', r'\quad + $\lambda=0.5$'),
    ('combined-balanced', r'\quad\quad + balancing'),
    ('combined-balanced-idk-0.5', r'\quad\quad\quad + $w_{\mathrm{IDK}}=0.5$'),
    ('combined-balanced-idk-0', r'\quad\quad\quad + $w_{\mathrm{IDK}}=0$'),
    ('combined-weight-2', r'\quad + $\lambda=2$'),
)


def table():
    results = {}
    gold = {}
    for name, label in CONDITIONS:
        record = {}
        for modality in ('temporal', 'location'):
            rows = read_rows(OUT / 'rescored' / name / 'outcomes' / modality / 'finetuned.jsonl')
            by_case = {r['case_id']: r['gold'] for r in rows}
            assert len(by_case) == len(rows)
            if modality in gold:
                assert by_case == gold[modality], 'Mismatched development cases or gold'
            gold[modality] = by_case
            eligible = [r for r in rows if r['gold'] is not None]
            record[modality] = {
                'n': len(eligible),
                'accuracy': 100 * sum(r['truth_correct'] for r in eligible) / len(eligible),
                'utility': 100 * sum(r['utility_uniform_wrong_penalty'] for r in eligible) / len(eligible),
            }
        training = json.loads((BASE / name / 'training.json').read_text())
        config = json.loads((BASE / name / 'config.json').read_text())
        assert config['granola_data_dir'] is None
        assert config['max_steps'] == 600 and config['early_stopping_patience'] is None
        record['best_checkpoint_step'] = training['best_checkpoint_step']
        record['contrastive_weight'] = config['contrastive_weight'] if config['objective'] == 'sft_contrastive' else None
        record['rank_balance'] = config['rank_balance']
        record['idk_weight'] = config['idk_sft_weight'] if config['objective'] != 'contrastive' else None
        results[name] = record
    columns = [('temporal', 'accuracy'), ('location', 'accuracy'),
               ('temporal', 'utility'), ('location', 'utility')]
    best = [max(record[m][metric] for record in results.values()) for m, metric in columns]
    lines = [r'\begin{table}[t]', r'\small', r'\centering',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Accuracy $\uparrow$} & \multicolumn{2}{c}{Utility $\uparrow$} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Objective & Temp. & Geo. & Temp. & Geo. \\', r'\midrule']
    for name, label in CONDITIONS:
        values = [results[name][m][metric] for m, metric in columns]
        cells = [rf'\textbf{{{v:.2f}}}' if f'{v:.2f}' == f'{maximum:.2f}' else f'{v:.2f}'
                 for v, maximum in zip(values, best)]
        if name in {'sft', 'combined-weight-1'}:
            lines.append(r'\midrule')
            heading = 'SFT' if name == 'sft' else 'Combined'
            lines.append(heading + r' & & & & \\')
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{LFM ablations on the Vital Persons',
              r'development set (accuracy in \%; utility $\times100$).',
              r'All runs use Vital Persons supervision and a 600-step budget.',
              r'Indented rows modify their parent configuration.',
              r'$\lambda$ weights the contrastive term; $w_{\mathrm{IDK}}$ weights',
              r'abstention examples in the SFT loss and defaults to 1.',
              r'Checkpoints are selected by development objective loss.',
              r'Bold indicates the best displayed value per column, including ties.}',
              r'\label{tab:objective-ablations}', r'\end{table}']
    write(OUT / 'summary.json', results)
    (OUT / 'table.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score', 'table'))
    args = parser.parse_args()
    if args.stage == 'table':
        table()
    else:
        entries = [(name, BASE / name / 'dev', 'finetuned') for name, _ in CONDITIONS]
        refresh(args.stage, 'dev', entries, OUT)


if __name__ == '__main__':
    main()
