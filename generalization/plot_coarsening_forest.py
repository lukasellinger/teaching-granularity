"""Paper forest plot: matched GranuScore + length and LLM-judge estimates."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
MODELS = [('lfm', 'LFM'), ('ministral', 'Ministral'), ('qwen', 'Qwen'), ('gemma', 'Gemma')]
DATASETS = [('granola', 'Granola-EQ'), ('simpleqa', 'SimpleQA'), ('truthfulqa', 'TruthfulQA')]
METHODS = [('combined', 'Granu + Length', '#0072B2', 'o', .12),
           ('judge', 'LLM judge', '#D55E00', 's', -.12)]


def plot(report, output):
    if not report.get('matched_judge_orderable') or report.get('absolute_level_filter') is not None:
        raise ValueError('Use the matched-cohort analysis without the old absolute-level filter')
    if report.get('adjustment') not in ('relation-only', 'linear') or report.get('scope') != 'first_sentence':
        raise ValueError('Expected first-sentence relation-only or fully adjusted linear estimates')
    relation_only = report['adjustment'] == 'relation-only'
    expected_controls = [] if relation_only else ['base_granuscore', 'base_words']
    if report.get('starting_controls') != expected_controls:
        raise ValueError('Regression-control metadata contradicts the requested specification')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), sharex=True, sharey=True)
    fig.subplots_adjust(left=.12, right=.985, bottom=.25, top=.87, wspace=.18)
    rows = []
    for ax, (dataset, title) in zip(axes, DATASETS, strict=True):
        for index, (model, label) in enumerate(MODELS):
            matched = [report[method]['by_dataset'][dataset]['by_model'][model] for method, *_ in METHODS]
            a, b = [group['coarser_vs_not_coarser'] for group in matched]
            if matched[0]['eligible_pairs'] != matched[1]['eligible_pairs'] or any(
                    a[key] != b[key] for key in ('cases', 'qid_clusters', 'formula')):
                raise ValueError('Methods do not use matching cases and regression specifications')
            for method, _, color, marker, offset in METHODS:
                group = report[method]['by_dataset'][dataset]['by_model'][model]
                result = group['coarser_vs_not_coarser']
                if result['status'] != 'estimated':
                    raise ValueError(f'Non-estimable result: {dataset}/{model}/{method}')
                value, lower, upper = [result[key] for key in ('odds_ratio', 'odds_ratio_ci_lower', 'odds_ratio_ci_upper')]
                if not all(math.isfinite(v) for v in (value, lower, upper)) or not 0 < lower <= value <= upper:
                    raise ValueError('Invalid odds ratio/confidence interval')
                ax.errorbar(value, 3-index+offset, xerr=[[value-lower], [upper-value]],
                    fmt=marker, color=color, markersize=4.5, capsize=2, elinewidth=1.1,
                    markeredgecolor='white', markeredgewidth=.35, zorder=3)
                rows.append({'dataset': dataset, 'model': model, 'method': method,
                    'adjustment': report['adjustment'],
                    'eligible_pairs': group['eligible_pairs'], 'regression_pairs': result['cases'],
                    'odds_ratio': value, 'lower': lower, 'upper': upper})
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.axvline(1, color='#666666', linestyle=(0, (4, 3)), linewidth=.9, zorder=2)
        ax.set_xscale('log')
        ax.set_xlim(.45, 20)
        ax.set_ylim(-.55, 3.55)
        ticks = [.5, 1, 2, 4, 8, 16]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter(['0.5', '1', '2', '4', '8', '16']))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis='x', color='#E8E8E8', linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', labelsize=8, length=3, color='#BBBBBB')
        ax.tick_params(axis='y', length=0, pad=7)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#BBBBBB')
    axes[0].set_yticks([3, 2, 1, 0], [label for _, label in MODELS])
    handles = [Line2D([], [], color=color, marker=marker, linestyle='-', markersize=5,
                      linewidth=1.1, label=label) for _, label, color, marker, _ in METHODS]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(.985, .02),
               ncol=2, frameon=False, fontsize=8.5, borderaxespad=0, borderpad=0,
               columnspacing=1.3, handlelength=2)
    first_panel = axes[0].get_position()
    fig.supxlabel('Odds ratio (log scale)', fontsize=9,
                  x=(first_panel.x0+first_panel.x1)/2, y=.02, ha='center')
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(output/f'coarser_at_risk_forest.{extension}', dpi=300, facecolor='white')
    (output/'coarser_at_risk_forest_data.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/coarsening-matched/summary.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    plot(json.loads(args.input.read_text()), args.output_dir)
    print(args.output_dir/'coarser_at_risk_forest.pdf')


if __name__ == '__main__':
    main()
