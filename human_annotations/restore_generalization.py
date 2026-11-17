"""Restore local annotation tasks from released IDs/labels and upstream questions."""
import argparse
import csv
import hashlib
from importlib import import_module
import json
from pathlib import Path
import shutil

from shared.io import read_rows, write, file_hash
from generalization.benchmark_data import load_cases, granola_cases, held_out_relations

REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO/'human_annotations/generalization'


def restore(data_dir, output_dir, cases):
    if output_dir.exists():
        raise ValueError('Use a new output directory; original study files are not overwritten')
    manifest = json.loads((data_dir/'manifest.json').read_text())
    for name, expected in manifest['files_sha256'].items():
        if file_hash(data_dir/name) != expected:
            raise ValueError('Public annotations changed: '+name)
    study = import_module('generalization.human-study.human_study')
    pairs = read_rows(data_dir/'pairs.jsonl')
    labels = {r['pair_id']: r for r in read_rows(data_dir/'granularity.jsonl')}
    correct = {(r['pair_id'], r['variant']): r for r in read_rows(data_dir/'correctness.jsonl')}
    index = {name: {str(r['id']): r for r in group} for name, group in cases.items()}
    rows = []
    for pair in pairs:
        question = index[pair['dataset']][pair['source_id']]['question']
        if hashlib.sha256(question.encode()).hexdigest() != pair['question_sha256']:
            raise ValueError('Reconstructed question differs from the study: '+pair['pair_id'])
        rows.append({**{k:v for k,v in pair.items() if k not in ('question_sha256','question_group')},
                     'question': question})
    output_dir.mkdir(parents=True)
    write(output_dir/'data/pairs.jsonl', rows, rows=True)
    def save_csv(name, records):
        path = output_dir/'annotations'/name; path.parent.mkdir(exist_ok=True)
        fields = list(dict.fromkeys(k for r in records for k in r))
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)
    independent = {(r['pair_id'], a): labels[r['pair_id']][a] for r in rows for a in ('annotator_1','annotator_2')}
    for number, annotator in enumerate(('annotator_1','annotator_2'),1):
        records=[]
        for row in rows:
            assignment=row['assignments'][annotator]
            records.append({**study.task_data(row,assignment), 'annotation_id': annotator+'-'+row['pair_id'],
                            'annotator': annotator, 'specificity': study.displayed(labels[row['pair_id']][annotator],assignment)})
        save_csv(f'annotator-{number}.csv',records)
    discussed=study.discussion_tasks(rows,independent)
    by_task={r['adjudication_task_uid']:r for r in rows}
    save_csv('adjudication.csv',[{**task['data'], 'annotation_id':'adjudication-'+task['data']['task_uid'],
             'annotator':'adjudication', 'final_specificity':study.displayed(labels[by_task[task['data']['task_uid']]['pair_id']]['final'],
                                                 by_task[task['data']['task_uid']]['assignments']['annotator_1'])} for task in discussed])
    tasks,keys=study.correctness_tasks(rows,cases['granola'])
    for key in keys.values():
        expected=correct[key['pair_id'],key['variant']]['reference_answers_sha256']
        actual=hashlib.sha256(json.dumps(key['reference_answers'],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if actual != expected:
            raise ValueError('Reconstructed reference hierarchy differs from the annotation task: '+key['pair_id'])
    records=[]
    for task in tasks:
        key=keys[task['data']['task_uid']];human=correct[key['pair_id'],key['variant']]
        records.append({**task['data'], **{k:v for k,v in human.items() if k not in ('pair_id','model','dataset','variant')},
                        'annotation_id':'correctness-'+task['data']['task_uid'], 'annotator':'correctness_annotator'})
    save_csv('correctness.csv',records)
    write(output_dir/'label_studio/correctness.json',tasks)
    study.prepare(output_dir,rows)
    write(output_dir/'label_studio/adjudication.json',discussed)
    for path in (REPO/'generalization/human-study/label_studio').glob('*.xml'):
        shutil.copyfile(path,output_dir/'label_studio'/path.name)
    write(output_dir/'restoration.json',{'public_manifest_sha256':file_hash(data_dir/'manifest.json'),
          'datasets':manifest['datasets'],'scope':'Locally reconstructed questions/references; released response texts, labels and task identities unchanged. CSV annotation identity fields are anonymous reconstruction markers, not original platform user IDs.'})
    print(f'Restored {len(rows)} local pairs and {len(records)} correctness judgments in {output_dir}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=PUBLIC)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--granola-jsonl',type=Path,help='Optional locally reconstructed Granola JSONL, before the notebook\'s optional split experiment')
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error('Use a new output directory')
    cases={name:load_cases(name) for name in ('simpleqa','truthfulqa')}
    if args.granola_jsonl:
        raw=read_rows(args.granola_jsonl)
        raw=[{**r,'Unnamed: 0':r.get('Unnamed: 0',r.get('id'))} for r in raw]
        if any(r['Unnamed: 0'] is None for r in raw):
            raise ValueError('Local Granola rows must retain original source IDs')
        cases['granola']=granola_cases(raw,held_out_relations())
    else:
        cases['granola']=load_cases('granola')
    restore(args.data_dir,args.output_dir,cases)


if __name__=='__main__':
    main()
