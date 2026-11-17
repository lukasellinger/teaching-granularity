"""Accuracy and utility alongside gold-blind fixed-backoff controls."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from parsing.temporal.values import TemporalValue, TemporalLevel, TemporalModifier, Season
from parsing.location.normalization.values import LocationValue, LocationLevel
from shared.io import digest, file_hash, read_rows, write
from .cases import temporal_level_for_taxonomy, TemporalTaxonomy
from .data import load_cases
from .temporal import correct_at_prediction_level as temporal_correct, temporal_informativeness
from .location import correct_at_prediction_level as location_correct, location_informativeness

ROOT = Path(__file__).resolve().parent
MODELS = {'lfm':'LFM', 'ministral':'Ministral', 'qwen':'Qwen', 'gemma':'Gemma'}
METHODS = {'base_plain':'Base', 'base_idk_user_prompt':'IDK',
           'base_granularity_instruction':'GAP', 'base_drag':'DRAG', 'finetuned':'FT'}
LEVELS = {'temporal':['DAY','MONTH','YEAR','DECADE','CENTURY'],
          'location':['CITY','REGION','COUNTRY','CONTINENT']}


def temporal_value(raw):
    values = dict(raw); values['level'] = TemporalLevel(values['level'])
    for field in ('decade_modifier','year_modifier','century_modifier'):
        if values.get(field):values[field]=TemporalModifier(values[field])
    if values.get('season'):values['season']=Season(values['season'])
    return TemporalValue(**values)


def project_prediction(row, modality, policy):
    """This decision never reads gold, correctness, or the model-derived target."""
    if policy == 'always-idk':
        return None, 'idk'
    status = row['parse_status']
    if status != 'parsed':
        return None, status
    value = temporal_value(row['prediction']) if modality == 'temporal' else LocationValue.from_dict(row['prediction'])
    levels = LEVELS[modality]
    desired = policy.removeprefix('cap-')
    if desired not in levels:raise ValueError('Unknown fixed-backoff policy')
    target = TemporalLevel(desired) if modality == 'temporal' else LocationLevel(desired)
    try:return value.to_level(target), 'parsed'
    except ValueError:return value, 'parsed'  # Never refine or invent missing geography.


def evaluate_control(rows, modality, policy):
    scored=[]
    for row in rows:
        prediction,status=project_prediction(row,modality,policy)
        level=(temporal_level_for_taxonomy(prediction.level,TemporalTaxonomy.FIVE_LEVEL).value
               if modality=='temporal' else prediction.level.value) if prediction else None
        gold_raw=row['gold']
        correct=reward=utility=None
        if gold_raw is not None:
            gold=temporal_value(gold_raw) if modality=='temporal' else LocationValue.from_dict(gold_raw)
            correct=bool(prediction is not None and (temporal_correct(prediction,gold) if modality=='temporal' else location_correct(prediction,gold)))
            reward=(temporal_informativeness(prediction.level) if modality=='temporal' else location_informativeness(prediction.level)) if correct else 0.
            utility=0. if status=='idk' else reward if correct else -1.
        match=(None if row['target_status']=='unavailable' else status=='idk' if row['target_status']=='idk' else level==row['target_level'])
        scored.append({'case_id':row['case_id'],'qid':row['qid'],'target_category':row['target_category'],
                       'target_match':match,'truth_correct':correct,'utility_positive_informativeness':reward,
                       'utility_uniform_wrong_penalty':utility,'parse_status':status})
    return summarize(scored)


def summarize(rows):
    mean=lambda key: float(np.mean([r[key] for r in rows if r[key] is not None])) if any(r[key] is not None for r in rows) else None
    groups=defaultdict(list)
    for row in rows:
        if row['target_match'] is not None:groups[row['target_category']].append(float(row['target_match']))
    reward=mean('utility_positive_informativeness');utility=mean('utility_uniform_wrong_penalty')
    return {'cases':len(rows),'gold_available':sum(r['truth_correct'] is not None for r in rows),
            'target_available':sum(r['target_match'] is not None for r in rows),
            'macro_target_match_accuracy':float(np.mean([np.mean(v) for v in groups.values()])) if groups else None,
            'target_match_by_category':{k:{'cases':len(v),'accuracy':float(np.mean(v))} for k,v in sorted(groups.items())},
            'truth_accuracy':mean('truth_correct'),'positive_information_reward':reward,
            'error_penalty':utility-reward if utility is not None else None,'uniform_error_utility':utility,
            'parse_status_counts':dict(Counter(r['parse_status'] for r in rows))}


def analyze(results_dir, models, split):
    result={};hashes={}
    for model in models:
        directory=results_dir/model/split
        report=json.loads((directory/'summary.json').read_text())
        cases=load_cases(model,split)
        if report['inputs']['cases_sha256']!=digest({k:[asdict(c) for c in rows] for k,rows in cases.items()}):
            raise ValueError('Rescore with the adapted gold first')
        result[model]={}
        for modality in LEVELS:
            methods={};outcomes={}
            for method in METHODS:
                path=directory/'outcomes'/modality/f'{method}.jsonl';rows=read_rows(path)
                ids=[r['case_id'] for r in rows]
                if len(set(ids))!=len(ids) or set(ids)!={c.case_id for c in cases[modality]}:
                    raise ValueError('Missing/duplicate outcome cases')
                outcomes[method]=rows;hashes[str(path)]=file_hash(path)
                methods[method]=summarize(rows)
                expected=report['modalities'][modality]['variants'][method]['headline']
                for key in ('macro_target_match_accuracy','truth_accuracy','uniform_error_utility'):
                    if methods[method][key]!=expected[key]:raise ValueError('Saved headline and outcomes differ')
            base=outcomes['base_plain']
            # These are global diagnostic caps, not selected per question.
            controls={f'cap-{level}':evaluate_control(base,modality,f'cap-{level}') for level in LEVELS[modality]}
            controls['always-idk']=evaluate_control(base,modality,'always-idk')
            finest=controls['cap-'+LEVELS[modality][0]]
            for key in ('macro_target_match_accuracy','truth_accuracy','uniform_error_utility'):
                if finest[key]!=methods['base_plain'][key]:raise ValueError('Finest cap must reproduce Base')
            caps=[key for key in controls if key.startswith('cap-')]
            best=max(caps,key=lambda key:controls[key]['uniform_error_utility'])
            result[model][modality]={'methods':methods,'controls':controls,
                'best_fixed_cap_oracle':{'policy':best,**controls[best]}}
    return result,hashes


def display(value):return '—' if value is None else f'{100*value:.2f}'


def render(result,split,all_controls=False):
    keys=('truth_accuracy','uniform_error_utility')
    lines=[f'# Vital Persons — {split}', '',
        'Values are percentages / percentage points. Accuracy and utility use gold-available cases. Fixed controls never refine or repair base predictions. The oracle reference selects one global cap per model/domain for utility on this split; it is not a per-question oracle.', '',
        '| Method | Accuracy: Temp. | Geo. | Utility: Temp. | Geo. |',
        '| --- | ---: | ---: | ---: | ---: |']
    latex=[r'\begin{table*}[t]',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\centering',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'& \multicolumn{2}{c}{Accuracy $\uparrow$}',
           r'& \multicolumn{2}{c}{Utility $\uparrow$} \\',
           r'\cmidrule(lr){2-3}',r'\cmidrule(lr){4-5}',
           r'Method & Temp. & Geo. & Temp. & Geo. \\',r'\midrule']
    cap_notes=[]
    for model,blocks in result.items():
        name=MODELS[model]
        if all_controls:
            entries=[]
            for modality in LEVELS:
                for policy,record in blocks[modality]['controls'].items():
                    entries.append((name+' + '+('Temp. ' if modality=='temporal' else 'Geo. ')+policy,
                                    {m:record if m==modality else None for m in LEVELS}))
        else:
            entries=[]
            for key,label in METHODS.items():
                shown=name+' (base)' if key=='base_plain' else 'FT (ours)' if key=='finetuned' else '+ '+label
                if key != 'finetuned':
                    entries.append((shown,{m:blocks[m]['methods'][key] for m in LEVELS}))
            entries += [('+ Best fixed cap',{m:blocks[m]['best_fixed_cap_oracle'] for m in LEVELS}),
                        ('FT (ours)',{m:blocks[m]['methods']['finetuned'] for m in LEVELS})]
            caps='/'.join(blocks[m]['best_fixed_cap_oracle']['policy'].removeprefix('cap-').lower() for m in LEVELS)
            cap_notes.append(name+': '+caps)
        raw=[[records[m][key] if records[m] is not None else None for key in keys for m in LEVELS] for _,records in entries]
        # Ties at displayed precision share the same emphasis.
        ranks=[sorted({round(100*row[c],2) for row in raw if row[c] is not None},reverse=True)[:2] for c in range(2*len(keys))]
        for (label,_),values in zip(entries,raw,strict=True):
            plain=[display(value) for value in values];tex=[];md=[]
            for c,(value,text) in enumerate(zip(values,plain,strict=True)):
                rank=ranks[c].index(round(100*value,2)) if value is not None and round(100*value,2) in ranks[c] else None
                tex.append(r'\textbf{'+text+'}' if rank==0 else r'\textit{'+text+'}' if rank==1 else text.replace('—','---'))
                md.append('**'+text+'**' if rank==0 else '*'+text+'*' if rank==1 else text)
            lines.append('| '+' | '.join([label,*md])+' |')
            shown=label if all_controls or label==name+' (base)' else r'\quad '+label
            latex.append(shown+' & '+' & '.join(tex)+r' \\')
        latex.append(r'\midrule')
    latex[-1]=r'\bottomrule'
    caption=(r'Results on the Vital Persons '+split+r' set. Accuracy values are percentages; utility values are multiplied by 100. '
        r'Accuracy measures factual compatibility with the external gold at the expressed granularity. '
        r'Utility rewards correct information, assigns $-1$ to incorrect or non-evaluable non-abstaining answers, and $0$ to recognized no-answer responses. '
        r'Cases without gold are excluded from both metrics. '
        r'GAP denotes granularity-aware prompting. All methods use one greedy response except DRAG, which aggregates 20 samples. '
        r'Backoff controls coarsen the saved plain-base prediction, retaining coarser/unprojectable answers, failures and abstentions. '
        r'The oracle reference selects one fixed cap per model and domain to maximize utility on this split, not per question. ')
    if cap_notes:
        chosen='; '.join(cap_notes)+'.'
        lines += ['', 'Selected oracle caps (Temp./Geo.): '+chosen, '']
        caption+='Oracle caps (Temp./Geo.): '+chosen+' '
    caption+=r'Best and second-best displayed results within each model block are bold and italic, including ties and controls.'
    label='tab:vital_persons_results' if not all_controls and split=='test' else 'tab:vital-'+('fixed-controls-' if all_controls else 'results-')+split
    latex += [r'\end{tabular}',r'\caption{',caption,r'}',r'\label{'+label+'}',r'\end{table*}']
    return '\n'.join(lines)+'\n','\n'.join(latex)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=ROOT/'results-corrected')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--split',choices=('dev','test'),default='test')
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    args=p.parse_args()
    if len(set(args.models))!=len(args.models):p.error('Choose distinct models')
    result,hashes=analyze(args.results_dir,args.models,args.split)
    output=args.output_dir or ROOT/'results-table'/args.split
    write(output/'summary.json',{'split':args.split,'models':result,'inputs_sha256':hashes,
        'control_definition':'Gold-blind projection of the saved plain-base prediction; preserve coarser values, missing fields, failures and abstentions. Best cap is a post-hoc global reference, not a per-case oracle.'})
    for full,name in [(False,'table'),(True,'fixed-controls')]:
        md,tex=render(result,args.split,full)
        (output/(name+'.md')).write_text(md);(output/(name+'.tex')).write_text(tex)
    print(f'Table and fixed-level controls written to {output}')


if __name__=='__main__':main()
