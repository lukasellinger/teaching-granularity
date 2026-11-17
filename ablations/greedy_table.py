"""Refresh and export the matched consensus versus rewritten-greedy control.

uv run --locked python -m ablations.greedy_table prepare
uv run --locked python -m evaluation.vital_persons.answer_status.judge run \
  --items ablations/lfm/greedy-control/results/answer-status/items.jsonl \
  --output-dir ablations/lfm/greedy-control/results/answer-status --allow-api
uv run --locked python -m ablations.greedy_table score
uv run --locked python -m ablations.greedy_table table
"""
import argparse
import json
from pathlib import Path
from shared.io import read_rows, write
from ablations.source_evaluation import refresh

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'ablations/lfm/greedy-control'
OUT = BASE / 'results'
LABELS = {'base': 'Base', 'greedy': 'Rewritten-greedy', 'consensus': 'Consensus'}


def entries():
    for name in LABELS:
        directory = (ROOT / 'evaluation/vital_persons/results/lfm/test' if name == 'base'
                     else BASE / name / 'vital_persons/test')
        yield name, directory, 'base_plain' if name == 'base' else 'finetuned'


def table():
    training = {}
    prompts = None
    for name in ('greedy', 'consensus'):
        rows = read_rows(BASE / name / 'data/train.jsonl')
        by_case = {(r['qid'], r['relation']): r['prompt'] for r in rows}
        assert len(by_case) == len(rows), 'Expected one realization per matched question'
        if prompts is not None:
            assert by_case == prompts, 'Training questions or prompts differ between arms'
        prompts = by_case
        config = json.loads((BASE / name / 'config.json').read_text())
        assert config['idk_sft_weight'] == 0.5 and config['granola_data_dir'] is None
        assert config['max_steps'] == 600 and config['eval_steps'] == 20
        assert config['early_stopping_patience'] == 5
        run = json.loads((BASE / name / 'training.json').read_text())
        training[name] = {'questions': len(rows), 'idk_targets': sum(r['is_idk'] for r in rows),
                          'best_checkpoint_step': run['best_checkpoint_step'],
                          'completed_global_step': run['completed_global_step']}
    results = {}
    gold = {}
    for name, _, variant in entries():
        record = {}
        for modality in ('temporal', 'location'):
            rows = read_rows(OUT / 'rescored' / name / 'outcomes' / modality / f'{variant}.jsonl')
            by_case = {r['case_id']: r['gold'] for r in rows}
            assert len(by_case) == len(rows)
            if modality in gold:
                assert by_case == gold[modality], 'Different test cases or gold references'
            gold[modality] = by_case
            eligible = [r for r in rows if r['gold'] is not None]
            record[modality] = {
                'n': len(eligible),
                'accuracy': 100 * sum(r['truth_correct'] for r in eligible) / len(eligible),
                'utility': 100 * sum(r['utility_uniform_wrong_penalty'] for r in eligible) / len(eligible),
            }
        results[name] = record
    columns = [('temporal', 'accuracy'), ('location', 'accuracy'),
               ('temporal', 'utility'), ('location', 'utility')]
    best = [max(record[m][metric] for record in results.values()) for m, metric in columns]
    lines = [r'\begin{table}[t]', r'\small', r'\centering',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Accuracy $\uparrow$} & \multicolumn{2}{c}{Utility $\uparrow$} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Training targets & Temp. & Geo. & Temp. & Geo. \\', r'\midrule']
    for name, label in LABELS.items():
        values = [results[name][m][metric] for m, metric in columns]
        cells = [rf'\textbf{{{v:.2f}}}' if f'{v:.2f}' == f'{maximum:.2f}' else f'{v:.2f}'
                 for v, maximum in zip(values, best)]
        lines.append(label + ' & ' + ' & '.join(cells) + r' \\')
    n = training['consensus']['questions']
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Matched LFM self-training controls on the Vital Persons',
              r'test set (accuracy in \%; utility $\times100$).',
              f'Both trained arms use the same {n:,} questions, one target per question,',
              r'IDK weight 0.5, and no \textsc{GranolaEQ} supervision.',
              r'Training uses up to 600 steps with early stopping and checkpoint',
              r'selection on shared development data. Bold indicates the best value per column.}',
              r'\label{tab:matched-greedy-control}', r'\end{table}']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'table.tex').write_text('\n'.join(lines)+'\n')
    summary = {'results': results, 'training': training}
    write(OUT / 'summary.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'score', 'table'))
    args = parser.parse_args()
    if args.stage == 'table':
        table()
    else:
        refresh(args.stage, 'test', entries(), OUT)


if __name__ == '__main__':
    main()
