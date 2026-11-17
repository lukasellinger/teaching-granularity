"""Compare our full-hierarchy utility with incorrect-answer scores -1 and 0.

Run: uv run --locked python -m evaluation.vital_persons.metric_sensitivity
Uses saved corrected test results; no generation, judging or level collapsing.
The appendix table compares Base and FT for each model and domain.
"""
import argparse
from pathlib import Path

from shared.io import file_hash, write
from .table import ROOT, MODELS, analyze

METRICS = (('ig_penalty', -1), ('ig_zero', 0))


def utility_record(record):
    return {'n': record['gold_available'], 'accuracy': record['truth_accuracy'],
            'scores': {'ig_penalty': record['uniform_error_utility'],
                       'ig_zero': record['positive_information_reward']}}


def analyze_metrics(results_dir):
    original, hashes = analyze(results_dir, list(MODELS), 'test')
    report = {}
    for model, blocks in original.items():
        report[model] = {}
        for modality, block in blocks.items():
            methods = {name: utility_record(record) for name, record in block['methods'].items()}
            assert len({record['n'] for record in methods.values()}) == 1
            report[model][modality] = {'methods': methods}
    return report, hashes


def render(report):
    latex = [r'\begin{table}[t]', r'\small', r'\centering',
             r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
             r'& \multicolumn{2}{c}{Error score $-1$} & \multicolumn{2}{c}{Error score $0$} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
             r'Method & Temp. & Geo. & Temp. & Geo. \\', r'\midrule']
    markdown = ['# Utility with and without an error penalty: Vital Persons test', '',
        'Scores are multiplied by 100. Both variants use the original full hierarchy, correctness decisions, no-answer labels and gold-available cases. Correct answers receive the same information gain; no-answer responses receive zero.', '',
        '| Method | Error -1: Temp. | Geo. | Error 0: Temp. | Geo. |',
        '| --- | ---: | ---: | ---: | ---: |']
    columns = [(domain, key) for key, _ in METRICS for domain in ('temporal', 'location')]
    for index, (model, blocks) in enumerate(report.items()):
        if index:
            latex.append(r'\midrule')
        data = []
        for method in ('base_plain', 'finetuned'):
            values = [blocks[domain]['methods'][method]['scores'][key] for domain, key in columns]
            data.append((method, values))
        maxima = [max(values[col] for _, values in data) for col in range(len(columns))]
        for method, values in data:
            cells = [rf'\textbf{{{100*v:.2f}}}' if f'{100*v:.2f}' == f'{100*best:.2f}' else f'{100*v:.2f}'
                     for v, best in zip(values, maxima)]
            label = MODELS[model]+' (base)' if method == 'base_plain' else r'FT (ours)'
            latex.append(label+' & '+' & '.join(cells)+r' \\')
            plain_label = MODELS[model]+' '+{'base_plain': 'Base', 'finetuned': 'FT'}[method]
            markdown.append('| '+' | '.join([plain_label, *(f'{100*v:.2f}' for v in values)])+' |')
    latex += [r'\bottomrule', r'\end{tabular}', r'\caption{',
        r'Base and fine-tuned utility on the Vital Persons test set with error scores $-1$ and $0$ (values $\times100$).',
        r'Both variants use the full evaluation hierarchy and identical correctness decisions;',
        r'abstentions receive zero.',
        r'Bold indicates the best displayed value within each model block, including ties.',
        r'}', r'\label{tab:metric-sensitivity}', r'\end{table}']
    return '\n'.join(latex)+'\n', '\n'.join(markdown)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results-corrected')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results-metric-sensitivity/test')
    args = parser.parse_args()
    report, hashes = analyze_metrics(args.results_dir)
    latex, markdown = render(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'table.tex').write_text(latex)
    (args.output_dir/'summary.md').write_text(markdown)
    write(args.output_dir/'summary.json', {
        'split': 'test', 'metrics': [{'key': key, 'error_score': penalty} for key, penalty in METRICS],
        'models': report, 'inputs_sha256': hashes,
        'geographic_reward_parameters_sha256': file_hash(ROOT/'location_informativeness.json'),
        'protocol': 'Use the original full evaluation hierarchy and saved correctness/no-answer decisions. Correct answers receive information gain and no-answer outputs receive zero. Other incorrect or non-evaluable outputs receive -1 or 0.'})
    print(f'Utility comparison written to {args.output_dir}')


if __name__ == '__main__':
    main()
