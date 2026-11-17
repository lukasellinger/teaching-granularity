"""Frozen consensus utility and post-hoc threshold sensitivity on Vital Persons test."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .table import ROOT, MODELS, temporal_value, temporal_correct, location_correct, temporal_informativeness, location_informativeness, analyze
from .cases import VITAL
from data_preparation.vital_persons.preparation.targets import temporal_consensus
from parsing.location.normalization.values import LocationValue, location_consensus, consensus_count
from shared.io import read_rows, write, file_hash

THRESHOLDS = [i/20 for i in range(10,21)]


def values_for(record, modality):
    raw = record['evidence']
    if modality == 'location':
        raw = [item.get('value') if item else None for item in raw]
    if len(raw) != 20:
        raise ValueError('Expected 20 original evidence slots, including missing values')
    parse = temporal_value if modality == 'temporal' else LocationValue.from_dict
    return [parse(item) if item else None for item in raw]


def infer(values, modality, threshold):
    target = (temporal_consensus if modality == 'temporal' else location_consensus)(values, threshold)
    available = sum(value is not None for value in values)
    status = 'known' if target is not None else 'idk' if available >= consensus_count(len(values), threshold) else 'unavailable'
    return target, status


def score_target(target, status, gold, modality):
    if status == 'unavailable':
        return None
    if status == 'idk':
        return 0.
    correct = (temporal_correct if modality == 'temporal' else location_correct)(target, gold)
    return (temporal_informativeness if modality == 'temporal' else location_informativeness)(target.level) if correct else -1.


def stats(records):
    utilities = [r['utility'] for r in records if r['utility'] is not None]
    return {'n': len(records), 'utility_n': len(utilities),
            'utility': float(np.mean(utilities)) if utilities else None,
            'statuses': dict(Counter(r['status'] for r in records)),
            'utility_lower_bound': float(np.mean([r['utility'] if r['utility'] is not None else -1. for r in records])) if records else None,
            'utility_upper_bound': float(np.mean([r['utility'] if r['utility'] is not None else 1. for r in records])) if records else None}


def run(output):
    # Check corrected outcomes against current adapted gold before using any scores.
    _, hashes = analyze(ROOT/'results-corrected', list(MODELS), 'test')
    records, outcomes, targets = {}, {}, {}
    report = {'split': 'test', 'thresholds': THRESHOLDS, 'inputs_sha256': hashes,
              'definition': 'Utility on a shared person/event cohort across all four models. Explicit IDK targets score zero; unavailable targets are never treated as IDK. Frozen 80% comparison excludes target-unavailable cases for any model. Sweep is post-hoc and does not evaluate retrained models.', 'domains': {}}
    for model in MODELS:
        path = VITAL/'datasets'/model/'consensus.json'
        report['inputs_sha256'][str(path)] = file_hash(path)
        records[model] = {(r['qid'],r['relation']): r for r in json.loads(path.read_text())}
        outcomes[model], targets[model] = {}, {}
        for modality in ('temporal','location'):
            methods = {}
            for method in ('base_plain','finetuned'):
                rows = read_rows(ROOT/'results-corrected'/model/'test'/'outcomes'/modality/(method+'.jsonl'))
                methods[method] = {(r['qid'],r['relation']):r for r in rows}
            outcomes[model][modality] = methods
            targets[model][modality] = {}
            for key,row in methods['base_plain'].items():
                if row['gold'] is None:
                    continue
                record = records[model][key]
                values = values_for(record, modality)
                gold = (temporal_value if modality == 'temporal' else LocationValue.from_dict)(row['gold'])
                frozen = (temporal_value if modality == 'temporal' else LocationValue.from_dict)(record['target']) if record['target'] else None
                recomputed, status = infer(values,modality,.8)
                if recomputed != frozen or status != record['target_status'] or status != row['target_status']:
                    raise ValueError(f'80% reconstruction differs from frozen target: {model}/{key}')
                thresholds = {}
                for threshold in THRESHOLDS:
                    target,status = infer(values,modality,threshold)
                    thresholds[str(threshold)] = {'status':status,'target':asdict(target) if target else None,
                        'utility':score_target(target,status,gold,modality)}
                targets[model][modality][key] = thresholds
    lines = ['# Consensus-target utility on the test set', '',
             'Utility ×100. Gold is the adapted external gold. All four models and all three methods use identical person/event questions within each domain. Base/FT pipeline failures remain −1; abstentions and explicit consensus IDK score zero. Consensus-unavailable cases are excluded jointly, not imputed as IDK.', '',
             '| Model | Temporal Base | FT | 80% target | Location Base | FT | 80% target |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for modality in ('temporal','location'):
        gold_keys = set.intersection(*(set(targets[m][modality]) for m in MODELS))
        for key in gold_keys:
            golds = [outcomes[m][modality]['base_plain'][key]['gold'] for m in MODELS]
            if any(g != golds[0] for g in golds):
                raise ValueError('Models must use identical external gold')
        shared80 = sorted(k for k in gold_keys if all(targets[m][modality][k]['0.8']['status'] != 'unavailable' for m in MODELS))
        complete = sorted(k for k in gold_keys if all(targets[m][modality][k][str(t)]['status'] != 'unavailable' for m in MODELS for t in THRESHOLDS))
        block = {'gold_available_n':len(gold_keys), 'comparison80_n':len(shared80),
                 'excluded_unavailable80_n':len(gold_keys)-len(shared80), 'comparison80_keys':shared80,
                 'fixed_sweep_n':len(complete),'fixed_sweep_keys':complete,'models':{}}
        for model in MODELS:
            methods = {}
            for method in ('base_plain','finetuned'):
                methods[method] = float(np.mean([outcomes[model][modality][method][k]['utility_uniform_wrong_penalty'] for k in shared80])) if shared80 else None
            methods['consensus80'] = stats([targets[model][modality][k]['0.8'] for k in shared80])
            fixed = {str(t):stats([targets[model][modality][k][str(t)] for k in complete]) for t in THRESHOLDS}
            # Keep the larger gold cohort too, with availability and missing-value bounds.
            all_gold = {str(t):stats([targets[model][modality][k][str(t)] for k in sorted(gold_keys)]) for t in THRESHOLDS}
            own_complete = sorted(k for k in gold_keys if all(targets[model][modality][k][str(t)]['status'] != 'unavailable' for t in THRESHOLDS))
            own_sweep = {str(t):stats([targets[model][modality][k][str(t)] for k in own_complete]) for t in THRESHOLDS}
            block['models'][model] = {'comparison80':methods,'fixed_cohort_sweep':fixed,'all_gold_sweep':all_gold,
                                     'model_fixed_sweep_n':len(own_complete), 'model_fixed_sweep_keys':own_complete, 'model_fixed_cohort_sweep':own_sweep}
        report['domains'][modality] = block
    for model in MODELS:
        cells = [MODELS[model]]
        for modality in ('temporal','location'):
            v = report['domains'][modality]['models'][model]['comparison80']
            cells.extend(f'{100*x:.2f}' for x in (v['base_plain'],v['finetuned'],v['consensus80']['utility']))
        lines.append('| '+' | '.join(cells)+' |')
    for modality, block in report['domains'].items():
        lines.extend(['',f"{modality.title()}: {block['comparison80_n']} shared questions out of {block['gold_available_n']} with gold; {block['excluded_unavailable80_n']} excluded because at least one model has an unavailable 80% target."])
    sweep = ['# Post-hoc consensus threshold sensitivity', '',
             'This is target utility, not FT performance after retraining. Utility ×100. The primary sweep fixes both questions and denominator across thresholds and models. It retains only questions with an available target (known or explicit IDK) at every threshold for every model. This selects evidence-complete cases and is not representative of the full test set. JSON also reports the full gold cohort at each threshold, with known/IDK/unavailable counts, observed utility, and bounds for unavailable values; those observed means have changing denominators.', '']
    for modality,block in report['domains'].items():
        sweep += [f"## {modality.title()} (fixed N = {block['fixed_sweep_n']})", '', '| Threshold | LFM | Ministral | Qwen | Gemma |', '| --- | ---: | ---: | ---: | ---: |']
        for threshold in THRESHOLDS:
            vals=[block['models'][m]['fixed_cohort_sweep'][str(threshold)]['utility'] for m in MODELS]
            sweep.append('| '+str(round(100*threshold))+'% | '+' | '.join('—' if v is None else f'{100*v:.2f}' for v in vals)+' |')
        sweep += ['', 'Post-hoc maximizing thresholds (ties retained):']
        for model in MODELS:
            values=block['models'][model]['fixed_cohort_sweep']
            available=[v['utility'] for v in values.values() if v['utility'] is not None]
            if available:
                best=max(available)
                winners=[f'{100*float(t):.0f}%' for t,v in values.items() if abs(v['utility']-best)<1e-12]
                sweep.append(f"- {MODELS[model]}: {', '.join(winners)} ({100*best:.2f}).")
        sweep.append('')
    sweep += ['## Model-specific fixed cohorts', '',
              'Each model uses a fixed set of gold-available questions with available targets throughout the sweep. Cohorts differ between models; compare thresholds within a model only. These cohorts also select evidence-complete cases.', '',
              '| Domain / model | N | 80% utility | 100% utility | Post-hoc best threshold | Best utility |',
              '| --- | ---: | ---: | ---: | --- | ---: |']
    for modality,block in report['domains'].items():
        for model,v in block['models'].items():
            values=v['model_fixed_cohort_sweep']
            available=[r['utility'] for r in values.values() if r['utility'] is not None]
            if not available:
                continue
            best=max(available)
            winners=[f'{100*float(t):.0f}%' for t,r in values.items() if abs(r['utility']-best)<1e-12]
            sweep.append(f"| {modality} / {MODELS[model]} | {v['model_fixed_sweep_n']} | {100*values['0.8']['utility']:.2f} | {100*values['1.0']['utility']:.2f} | {', '.join(winners)} | {100*best:.2f} |")
    write(output/'summary.json',report)
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    (output/'threshold-sweep.md').write_text('\n'.join(sweep)+'\n')
    (output/'threshold-sweep-caption.txt').write_text('Post-hoc consensus-target utility sensitivity on the test set. Each curve uses its own fixed gold-available cohort with available targets at every threshold; model cohorts differ and select evidence-complete cases. The dashed line marks the fixed training threshold of 80%. These are target utilities, not outcomes of retraining. See summary.json for the shared-cohort sweep and full-cohort availability.\n')
    plot_sweep(report,output)
    print('\n'.join(lines)); print('\n'+'\n'.join(sweep))


def plot_sweep(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(7,2.8),sharex=True)
    for ax,(modality,block) in zip(axes,report['domains'].items(),strict=True):
        for model in MODELS:
            vals=[block['models'][model]['model_fixed_cohort_sweep'][str(t)]['utility'] for t in THRESHOLDS]
            ax.plot([100*t for t in THRESHOLDS],[100*v if v is not None else np.nan for v in vals],marker='.',label=MODELS[model]+f" (N={block['models'][model]['model_fixed_sweep_n']})")
        ax.axvline(80,color='grey',linestyle='--',linewidth=.8)
        ax.set(title=modality.title(),xlabel='Consensus threshold (%)',ylabel='Target utility ×100')
        ax.spines[['top','right']].set_visible(False)
    for ax in axes:
        ax.legend(loc='best',fontsize=8,frameon=False)
    fig.tight_layout()
    for extension in ('pdf','png'):
        fig.savefig(output/('threshold-sweep.'+extension),dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results-consensus-utility'/'test')
    args=p.parse_args();run(args.output_dir)

if __name__ == '__main__':main()
