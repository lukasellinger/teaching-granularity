"""Relation-by-model overview of confidence-associated coarsening.

uv run --locked --extra plotting python -m generalization.plot_relation_sensitivity
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from .confidence_sensitivity import DATASETS, MODELS, NAMES
from .relation_sensitivity import RELATION_NAMES

ROOT = Path(__file__).resolve().parent
DATASET_NAMES = {'granola': 'Granola-EQ', 'simpleqa': 'SimpleQA', 'truthfulqa': 'TruthfulQA'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'results/coarsening-by-relation/summary.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/coarsening-by-relation')
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    records = {(r['source'], r['dataset'], r['model']): r for r in report['results']}
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9, 'pdf.fonttype': 42})
    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#EEEEEE')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        relations = list(records['judge', dataset, 'lfm']['by_relation'])
        for spec, description in [('quartiles', 'Lowest vs. highest confidence quartile'),
                                  ('continuous', 'One within-group SD lower confidence')]:
            height = max(3.2, .40*len(relations)+1.35)
            fig, axes = plt.subplots(1, 2, figsize=(8.2, height), sharey=True)
            left = .30 if dataset == 'truthfulqa' else .23 if dataset == 'simpleqa' else .16
            fig.subplots_adjust(left=left, right=.98, bottom=.24, top=.82, wspace=.10)
            for ax, source, title in zip(axes, ('combined','judge'), ('Combined metric','LLM judge')):
                colors = np.full((len(relations), len(MODELS)), np.nan)
                for i, relation in enumerate(relations):
                    for j, model in enumerate(MODELS):
                        fit = records[source, dataset, model]['by_relation'][relation][spec]
                        if fit['status']=='estimated':
                            colors[i,j] = np.clip(math.log2(fit['odds_ratio']), -3, 3)
                im = ax.imshow(colors, cmap=cmap, norm=norm, aspect='auto', interpolation='none')
                for i, relation in enumerate(relations):
                    for j, model in enumerate(MODELS):
                        fit = records[source, dataset, model]['by_relation'][relation][spec]
                        if fit['status']=='estimated':
                            mark = '*' if fit['odds_ratio_ci_lower']>1 or fit['odds_ratio_ci_upper']<1 else ''
                            value = fit['odds_ratio']
                            label = (f'{value:.1f}' if value>=10 else f'{value:.2f}')+mark
                            color = 'white' if abs(colors[i,j])>1.7 else '#222222'
                        else:
                            label, color = 'NE', '#666666'
                        ax.text(j,i,f"{label}\nN={fit['cases']}",ha='center',va='center',fontsize=7.5,color=color)
                ax.set_title(title,fontsize=10,pad=7)
                ax.set_xticks(range(4), [NAMES[m] for m in MODELS])
                ax.set_yticks(range(len(relations)), [RELATION_NAMES.get(r,r) for r in relations])
                ax.tick_params(length=0,pad=6)
                ax.set_xticks(np.arange(-.5,4,1),minor=True)
                ax.set_yticks(np.arange(-.5,len(relations),1),minor=True)
                ax.grid(which='minor',color='white',linewidth=1)
                ax.tick_params(which='minor',length=0)
                for spine in ax.spines.values():spine.set_visible(False)
            fig.suptitle(DATASET_NAMES[dataset]+' — '+description,fontsize=11,y=.965)
            cax=fig.add_axes([max(left,.30),.12,.55,.025])
            bar=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-3,-2,-1,0,1,2,3],extend='both')
            bar.ax.set_xticklabels(['1/8','1/4','1/2','1','2','4','8'])
            bar.ax.tick_params(labelsize=7,length=2)
            bar.set_label('Odds ratio (log scale)',fontsize=8,labelpad=2)
            fig.text(.5,.015,'* Unadjusted 95% CI excludes 1; NE: not estimable. N: observations used in the regression.',
                     ha='center',fontsize=7.5)
            for ext in ('pdf','png'):
                fig.savefig(args.output_dir/f'{dataset}-{spec}.{ext}',dpi=220,facecolor='white')
            plt.close(fig)
    print(args.output_dir)


if __name__ == '__main__':
    main()
