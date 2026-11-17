"""Refresh only location utility fields from saved outcomes; preserve factual decisions."""
import argparse
import json
from pathlib import Path
import numpy as np
from parsing.location.normalization.values import LocationLevel
from shared.io import read_rows, write, file_hash
from shared.bootstrap import paired_qid_bootstrap
from .location import location_informativeness, aggregate_location_outcomes, _LOCATION_INFORMATION
from .report import headline
from .table import ROOT


def refresh_row(row):
    result = dict(row)
    if row['gold'] is None:
        return result
    prediction = row.get('prediction')
    gain = location_informativeness(LocationLevel(prediction['level'])) if prediction else 0.
    for prefix in ('', 'strict_context_'):
        if prefix+'truth_correct' not in row:
            continue
        correct = row[prefix+'truth_correct']
        changes = {prefix+'utility':gain*(1 if correct else -1) if prediction else 0.,
                   prefix+'utility_positive_informativeness':gain if correct and prediction else 0.,
                   prefix+'utility_uniform_wrong_penalty':0. if row['parse_status']=='idk' else gain if correct and prediction else -1.}
        result.update({key:value for key,value in changes.items() if key in row})
    return result


def update_utility_fields(saved, fresh):
    for key in saved:
        if key in fresh:
            if 'utility' in key:
                saved[key] = fresh[key]
            elif isinstance(saved[key],dict) and isinstance(fresh[key],dict):
                update_utility_fields(saved[key],fresh[key])



def refresh_metric_file(directory, variant, fresh_metrics):
    """Synchronize existing utility aggregates without changing factual metrics."""
    path = directory / 'metrics/location' / f'{variant}.json'
    if not path.is_file():
        return False  # Rescored reports embed metrics in summary.json instead.
    saved = json.loads(path.read_text())
    before = json.dumps(saved['metrics'], sort_keys=True)
    update_utility_fields(saved['metrics'], fresh_metrics)
    if json.dumps(saved['metrics'], sort_keys=True) == before:
        return False
    write(path, saved)
    return True


def utility_means(rows):
    result = {}
    for prefix in ('','strict_context_'):
        for name in ('utility','utility_uniform_wrong_penalty','utility_accuracy','utility_positive_informativeness'):
            key=prefix+name
            values=[r[key] for r in rows if r.get(key) is not None]
            result[prefix+'location_'+name]=float(np.mean(values)) if values else None
    return result


def refresh(directory):
    report_path=directory/'summary.json'
    report=json.loads(report_path.read_text())
    block=report['modalities']['location']
    changed=[]; outcomes={}
    for path in sorted((directory/'outcomes/location').glob('*.jsonl')):
        before=read_rows(path); after=[refresh_row(row) for row in before]
        for a,b in zip(before,after,strict=True):
            assert {k:v for k,v in a.items() if 'utility' not in k} == {k:v for k,v in b.items() if 'utility' not in k}
        outcomes[path.stem]=after
        record=block['variants'][path.stem]
        fresh=aggregate_location_outcomes(after)
        refresh_metric_file(directory, path.stem, fresh)
        if 'metrics' in record:
            update_utility_fields(record['metrics'],fresh)
            newhead=headline(fresh,after,'location')
            for key in ('positive_information_reward','uniform_error_utility','error_penalty'):
                record['headline'][key]=newhead[key]
        else:
            for section in ('guardrails','primary','secondary'):
                update_utility_fields(record[section],fresh)
        oldutility=utility_means(before)['location_utility_uniform_wrong_penalty']
        newutility=utility_means(after)['location_utility_uniform_wrong_penalty']
        changed.append({'variant':path.stem,'before_utility':oldutility,'after_utility':newutility})
        write(path,after,rows=True)
    comparisons=block.get('comparisons',block.get('finetuned_minus_baselines',{}))
    for baseline,old in comparisons.items():
        if baseline not in outcomes or 'finetuned' not in outcomes:
            raise ValueError('Missing outcome data for existing bootstrap')
        names=[key for key in old['metrics'] if 'utility' in key]
        if names:
            fresh=paired_qid_bootstrap(outcomes[baseline],outcomes['finetuned'],metric_names=names,
                compute_metrics=utility_means,num_resamples=old['num_resamples'],confidence_level=old['confidence_level'],seed=old['seed'])
            old['metrics'].update(fresh['metrics'])
    report['location_informativeness']=_LOCATION_INFORMATION
    write(report_path,report)
    text=directory/'summary.txt'
    if text.exists() and 'inputs' in report:
        lines=['Vital Persons — percentages / percentage points',
               'Modality  Variant  Truth  Target  Macro-target  Reward  Penalty  Utility']
        for modality,section in report['modalities'].items():
            for variant,entry in section['variants'].items():
                head=entry['headline'];keys=('truth_accuracy','target_match_accuracy','macro_target_match_accuracy','positive_information_reward','error_penalty','uniform_error_utility')
                lines.append(f'{modality}  {variant}  '+'  '.join('—' if head[k] is None else f'{100*head[k]:.2f}' for k in keys))
        text.write_text('\n'.join(lines)+'\n')
    return {'directory':str(directory),'summary_sha256':file_hash(report_path),'changes':changed}


def refresh_exports():
    import csv
    import io
    # Preserve the existing ablation table and update only location utility columns.
    names={'combined':'combined-weight-1','combined_lambda_0.5':'combined-weight-0.5',
           'combined_lambda_0.5_balanced_cap_8':'combined-balanced',
           'combined_lambda_0.5_idk_weight_0.5_balanced_cap_8':'combined-balanced-idk-0.5',
           'combined_lambda_0.5_idk_weight_0.5_granola_balanced_cap_8':'combined-balanced-idk-0.5-granola',
           'combined_lambda_0.5_idk_weight_0_balanced_cap_8':'combined-balanced-idk-0',
           'combined_lambda_2':'combined-weight-2','contrastive_only':'contrastive',
           'sft':'sft','sft_idk_weight_0':'sft-idk-0','sft_idk_weight_0.5':'sft-idk-0.5',
           'sft_idk_weight_0.5_granola':'sft-idk-0.5-granola'}
    path=ROOT.parents[1]/'ablations/lfm/short/results.csv'
    reader=csv.DictReader(path.open()); fields=reader.fieldnames; rows=list(reader)
    for row in rows:
        report=json.loads((path.parent/names[row['variant']]/'dev/summary.json').read_text())
        groups=report['modalities']['location']['variants']['finetuned']
        for field in fields:
            if field.startswith('location_') and 'utility' in field:
                key=field[len('location_'):]
                matches=[group[key] for group in groups.values() if isinstance(group,dict) and key in group]
                if len(matches)==1: row[field]=matches[0]
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    path.write_text(stream.getvalue())
    path=ROOT/'normalization-comparison.json'
    if path.exists():
        report=json.loads(path.read_text())
        for row in report['rows']:
            for condition in ('uncorrected','corrected'):
                source=json.loads((ROOT/('results-'+condition)/row['model']/row['split']/'summary.json').read_text())
                row[condition]=source['modalities'][row['modality']]['variants'][row['variant']]['headline']
            for key in row['difference']:
                a,b=row['uncorrected'][key],row['corrected'][key]
                row['difference'][key]=None if a is None or b is None else b-a
        report['location_informativeness']=_LOCATION_INFORMATION
        write(path,report)
        md=path.with_suffix('.md');text=md.read_text();prefix=text[:text.index('## Test location results')]
        lines=[prefix.rstrip(),'']
        for split in ('test','dev'):
            lines += [f'## {split.title()} location results','',
                '| Model | Method | Accuracy without | Accuracy with | Utility without | Utility with |',
                '| --- | --- | ---: | ---: | ---: | ---: |']
            for row in report['rows']:
                if row['split']==split and row['modality']=='location':
                    values=[row[c][k] for k in ('truth_accuracy','uniform_error_utility') for c in ('uncorrected','corrected')]
                    lines.append('| '+' | '.join([row['model'],row['variant'],*[f'{100*v:.2f}' for v in values]])+' |')
            lines.append('')
        md.write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-ablations',action='store_true')
    args=parser.parse_args()
    directories=[]
    for name in ('results-corrected','results-uncorrected'):
        directories.extend(sorted((ROOT/name).glob('*/*/summary.json')))
    if args.include_ablations:
        directories.extend(sorted((ROOT.parents[1]/'ablations').rglob('summary.json')))
    changes=[]
    for path in directories:
        if not (path.parent/'outcomes/location').is_dir():
            continue
        print('Refreshing '+str(path.parent),flush=True)
        changes.append(refresh(path.parent))
    if args.include_ablations:
        refresh_exports()
    write(ROOT/'results-informativeness'/'rescore.json',{'parameters':_LOCATION_INFORMATION,'runs':changes})


if __name__=='__main__':main()
