"""Offline base-versus-FT preference from correctness and calibrated granularity."""
import argparse
from importlib import import_module
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'generalization'


from generalization.benchmark_data import load_cases
from shared.io import read_rows, write
from generalization.generate import validate_records

ROOT = Path(__file__).resolve().parent
OUTCOMES = ('finetuned', 'base', 'tie', 'both_incorrect', 'unresolved')
LABELS = {'Correct', 'Incorrect', 'Abstain', 'Cannot determine'}
RULE = ('Correct > Abstain > Incorrect. If both Correct, prefer the finer answer by the '
        'calibrated metric. Both Incorrect: neither; both Abstain or equal granularity '
        'among correct answers: tie; either Cannot determine: unresolved.')


def question_key(question):
    return re.sub(r'\s+', ' ', question).strip().casefold()


def direction(parameters, base_score, ft_score, base, ft):
    p = parameters
    if not all(math.isfinite(float(v)) for v in (*p.values(), base_score, ft_score)):
        raise ValueError('Finite metric parameters and GranuScores required')
    if not 0 <= p['alpha'] <= 1 or p['band'] < 0 or min(p['granuscore_scale'], p['length_scale']) <= 0:
        raise ValueError('Invalid calibrated metric parameters')
    score = (p['alpha']*(ft_score-base_score)/p['granuscore_scale']
             +(1-p['alpha'])*(len(base.split())-len(ft.split()))/p['length_scale'])
    label = ('base_more_specific' if score > p['band'] else
             'finetuned_more_specific' if score < -p['band'] else 'equal')
    return label, score


def winner(base, ft, finer):
    if base not in LABELS or ft not in LABELS:
        raise ValueError('Invalid correctness judgment')
    if 'Cannot determine' in (base, ft):
        return 'unresolved'
    if base == ft == 'Correct':
        return {'base_more_specific': 'base', 'finetuned_more_specific': 'finetuned', 'equal': 'tie'}.get(finer, 'unresolved')
    if base == ft:
        return 'tie' if base == 'Abstain' else 'both_incorrect'
    rank = {'Correct': 2, 'Abstain': 1, 'Incorrect': 0}
    return 'finetuned' if rank[ft] > rank[base] else 'base'


def summarize(records):
    counts = Counter(r['preference'] for r in records)
    decisive = counts['base']+counts['finetuned']
    return {'pairs': len(records), 'counts': {k: counts[k] for k in OUTCOMES},
            'percent_all_pairs': {k: 100*counts[k]/len(records) if records else None for k in OUTCOMES},
            'decisive_pairs': decisive,
            'ft_share_decisive_percent': 100*counts['finetuned']/decisive if decisive else None,
            'both_correct_granularity': dict(Counter(r.get('granularity_direction', r.get('proxy_direction')) for r in records
                if r.get('base_correctness', r.get('judge_base_correct')) ==
                   r.get('finetuned_correctness', r.get('judge_finetuned_correct')) == 'Correct'))}


def breakdown(records):
    models = sorted({r['model'] for r in records})
    relations = sorted({r['relation'] for r in records})
    return {'overall': summarize(records),
            'by_model': {m: summarize([r for r in records if r['model'] == m]) for m in models},
            'by_relation': {rel: summarize([r for r in records if r['relation'] == rel]) for rel in relations},
            'by_model_relation': {m: {rel: summarize([r for r in records if r['model'] == m and r['relation'] == rel])
                for rel in relations} for m in models}}


def compare(models, generations_dir, judged_dir, metric, study):
    if metric['scope'] != 'first_sentence':
        raise ValueError('This analysis requires the first-sentence calibrated metric')
    calibration = {question_key(r['question']) for r in read_rows(study/'data/pairs.jsonl')}
    cases = load_cases('granola')
    expected = {str(c['id']): c for c in cases}
    result = []
    for model in models:
        scored = read_rows(generations_dir/'granola'/f'{model}.jsonl')
        validate_records(scored, cases, complete=True)
        lookup = {(str(r['id']), r['variant']): r for r in scored}
        judged = read_rows(judged_dir/f'{model}.jsonl')
        seen = set()
        for row in judged:
            uid = str(row['source_id'])
            if uid in seen or uid not in expected or row['model'] != model or row['dataset'] != 'granola' or row['answer_scope'] != 'first-sentence':
                raise ValueError('Duplicate or misidentified correctness judgment')
            seen.add(uid)
            case = expected[uid]
            if any(row.get(k) != case[k] for k in ('question', 'relation', 'reference_answers')) or row['qid'] != 'granola:'+case['qid']:
                raise ValueError('Correctness judgment does not match benchmark gold/identity')
            a, b = lookup[(uid, 'base')], lookup[(uid, 'finetuned')]
            if a['first_sentence'] != row['base_response'] or b['first_sentence'] != row['finetuned_response']:
                raise ValueError('Correctness judgment belongs to different answer text')
            label, shift = direction(metric['parameters'], a['first_sentence_granuscore'],
                b['first_sentence_granuscore'], a['first_sentence'], b['first_sentence'])
            result.append({**row, 'proxy_direction': label, 'hybrid_shift': shift,
                'preference': winner(row['judge_base_correct'], row['judge_finetuned_correct'], label),
                'calibration_question': question_key(row['question']) in calibration})
        if seen != expected.keys():
            raise ValueError('Correctness judgments must cover every generation pair')
    return result


def compare_human(models, study):
    human_study = import_module('generalization.human-study.human_study')
    pairs, granularity = human_study.adjudicated_study(study)
    cases = load_cases('granola')
    tasks, keys = human_study.correctness_tasks(pairs, cases)
    annotations = human_study.read_correctness(tasks, keys, study/'annotations/correctness.csv')
    gold = {(r['pair_id'], r['variant']): r for r in annotations}
    references = {r['id']: r for r in cases}
    final, reference_only = [], []
    for pair in pairs:
        if pair['dataset'] != 'granola' or pair['model'] not in models:
            continue
        case = references[pair['source_id']]
        record = {'pair_id': pair['pair_id'], 'model': pair['model'], 'dataset': 'granola',
                  'source_id': pair['source_id'], 'question': pair['question'],
                  'relation': case['relation'], 'qid': 'granola:'+case['qid'],
                  'base_response': pair['base']['text'], 'finetuned_response': pair['finetuned']['text'],
                  'granularity_direction': granularity[pair['pair_id']]}
        for field, destination in (('correctness', final), ('reference_correctness', reference_only)):
            base, ft = (gold[(pair['pair_id'], v)][field] for v in ('base', 'finetuned'))
            destination.append({**record, 'base_correctness': base, 'finetuned_correctness': ft,
                                'preference': winner(base, ft, record['granularity_direction'])})
    return final, reference_only


def compare_judge(records, predictions):
    """Change only the granularity measure; preserve cases and correctness labels."""
    index = {}
    for row in predictions:
        key = row['model'], row['dataset'], str(row['source_id'])
        if key in index:
            raise ValueError('Duplicate granularity judgment')
        index[key] = row
    result = []
    for row in records:
        key = row['model'], row['dataset'], str(row['source_id'])
        if key not in index:
            raise ValueError('Missing granularity judgment for preference pair')
        judged = index[key]
        if any(judged[k] != row[k] for k in ('question', 'relation', 'base_response', 'finetuned_response')):
            raise ValueError('Granularity judgment belongs to a different pair')
        label = judged['direction']
        if label not in {'base_more_specific', 'finetuned_more_specific', 'equal', 'abstention_present', 'not_comparable'}:
            raise ValueError('Invalid granularity judgment')
        result.append({**row, 'granularity_direction': label,
            'preference': winner(row['judge_base_correct'], row['judge_finetuned_correct'], label)})
    return result


def text_summary(report):
    rule = report['rule'].replace('calibrated metric', 'selected granularity measure') if 'judge' in report else report['rule']
    lines = ['Granola preference: base versus its fine-tuned model (first sentence).', rule,
             'FT share uses only FT/base wins; all other outcomes remain listed.', '']
    if 'judge' in report:
        lines += ['GranuScore + length and LLM judge use identical pairs and identical correctness judgments.',
                  'Granularity only breaks ties when both answers are Correct.',
                  'If both are Correct but judge granularity is abstention/not-comparable, preference is unresolved.', '']
    header = 'Group              Pairs   FT wins Base wins   Tie Both wrong Unresolved  FT/decisive %'
    if 'judge' in report:
        header = 'Group              Measure                 Pairs   FT wins Base wins   Tie Both wrong Unresolved  FT/decisive %'
    def table(title, groups, alternate=None):
        lines.extend([title, header])
        for name, group in groups:
            measures = [('combined', group), ('judge', alternate[name])] if alternate is not None else [('', group)]
            for method, value in measures:
                c = value['counts']
                share = value['ft_share_decisive_percent']
                shown = f'{share:.2f}' if share is not None else 'n/a'
                display = {'combined': 'GranuScore + length', 'judge': 'LLM judge'}.get(method, method)
                label = f'{name:<19}{display:<24}' if alternate is not None else f'{name:<19}'
                lines.append(f'{label}{value["pairs"]:>5} {c["finetuned"]:>9} {c["base"]:>9} '
                             f'{c["tie"]:>5} {c["both_incorrect"]:>10} {c["unresolved"]:>10} {shown:>14}')
        lines.append('')
    judge = report.get('judge', {})
    table('By model', report['by_model'].items(), judge.get('by_model'))
    table('By relation (pooled models)', report['by_relation'].items(), judge.get('by_relation'))
    table('By model × relation', ((f'{m}/{rel}', value)
        for m, groups in report['by_model_relation'].items() for rel, value in groups.items()),
        {f'{m}/{rel}': value for m, groups in judge['by_model_relation'].items() for rel, value in groups.items()} if judge else None)
    lines.append(report['interpretation'])
    if 'excluding_calibration_questions' in report:
        lines.append('summary.json also reports results excluding all human-study calibration questions.')
    if 'reference_only' in report:
        lines.append('summary.json also reports preferences using reference-only human correctness judgments.')
    return '\n'.join(lines)+'\n'


def matched_resolved(records, judge_records):
    """Use the intersection of resolved pairs, identically for both measures."""
    def index(rows):
        result = {(r['model'], str(r['source_id'])): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate preference pair')
        return result
    combined, judge = index(records), index(judge_records)
    if combined.keys() != judge.keys():
        raise ValueError('Preference measures must cover identical pairs')
    kept = {k for k in combined if combined[k]['preference'] != 'unresolved'
            and judge[k]['preference'] != 'unresolved'}
    accuracy = {}
    for model in sorted({k[0] for k in kept}):
        rows = [combined[k] for k in combined if k in kept and k[0] == model]
        for row in rows:
            other = judge[(model, str(row['source_id']))]
            if any(row[field] != other[field] for field in ('judge_base_correct', 'judge_finetuned_correct')):
                raise ValueError('Preference measures must use identical correctness labels')
        counts = {variant: sum(r[field] == 'Correct' for r in rows)
                  for variant, field in [('base', 'judge_base_correct'), ('finetuned', 'judge_finetuned_correct')]}
        accuracy[model] = {'pairs': len(rows), 'correct': counts,
                           'percent': {v: 100*c/len(rows) for v, c in counts.items()}}
    return {'rule': 'Exclude a pair from both measures if either preference is unresolved.',
            'accuracy_by_model': accuracy,
            'combined': breakdown([r for k, r in combined.items() if k in kept]),
            'judge': breakdown([r for k, r in judge.items() if k in kept])}


def latex_table(report):
    matched = report['matched_resolved']
    lines = [r'\begin{table}[t]', r'\small', r'\centering', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{@{}llrrr@{}}', r'\toprule',
             r'Model & Metric & FT & Base & Neither \\', r'\midrule']
    sample_counts = []
    for model, label in [('lfm', 'LFM'), ('ministral', 'Ministral'), ('qwen', 'Qwen'), ('gemma', 'Gemma')]:
        if model not in matched['combined']['by_model']:
            continue
        n = matched['combined']['by_model'][model]['pairs']
        if sample_counts:
            lines.append(r'\addlinespace[4pt]')
        sample_counts.append(f'{label}: {n:,}')
        accuracy = matched['accuracy_by_model'][model]['percent']
        shown = [f'{accuracy[v]:.2f}' for v in ('finetuned', 'base')]
        if accuracy['finetuned'] != accuracy['base']:
            best = 0 if accuracy['finetuned'] > accuracy['base'] else 1
            shown[best] = r'\textbf{'+shown[best]+'}'
        lines.append(label+' & Accuracy & '+' & '.join(shown)+r' & --- \\')
        for name, source in [('Granu + Length', matched['combined']), ('LLM judge', matched['judge'])]:
            group = source['by_model'][model]
            counts, n = group['counts'], group['pairs']
            values = [100*counts[k]/n for k in ('finetuned', 'base')]
            values.append(100*sum(counts[k] for k in ('tie', 'both_incorrect'))/n)
            formatted = [f'{v:.2f}' for v in values]
            if values[0] != values[1]:
                index = 0 if values[0] > values[1] else 1
                formatted[index] = r'\textbf{'+formatted[index]+'}'
            lines.append(' & Pref. '+name+' & '+' & '.join(formatted)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{',
        r'Accuracy and preference on held-out Granola-EQ relations (\%). Accuracy counts only correct answers. Preference ranks correct answers above abstentions above incorrect answers; when both are correct, the finer-grained answer is preferred.',
        'Neither includes ties and both-incorrect pairs. Bold marks the higher FT/Base value.',
        'All rows use the same pairs per model, excluding comparisons unresolved under either measure ($N$: '+', '.join(sample_counts)+').',
        r'}', r'\label{tab:granola-preference}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def latex_detailed_table(report):
    lines = [r'\begin{table}[t]', r'\small', r'\centering', r'\setlength{\tabcolsep}{2.5pt}',
             r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule',
             r'Measure & FT & Base & Tie & \shortstack{Both\\wrong} & Unres. \\', r'\midrule']
    for i, (model, label) in enumerate([('lfm', 'LFM'), ('ministral', 'Ministral'), ('qwen', 'Qwen'), ('gemma', 'Gemma')]):
        if model not in report['by_model']:
            continue
        if i:
            lines.append(r'\addlinespace')
        lines.append(r'\multicolumn{6}{@{}l}{\textit{'+label+r'}} \\')
        for name, source in [('Granu + Length', report), ('LLM judge', report['judge'])]:
            group = source['by_model'][model]
            percentages = group['percent_all_pairs']
            values = [percentages[k] for k in ('finetuned', 'base', 'tie', 'both_incorrect', 'unresolved')]
            formatted = [f'{v:.2f}' for v in values]
            if values[0] != values[1]:
                winner_index = 0 if values[0] > values[1] else 1
                formatted[winner_index] = r'\textbf{'+formatted[winner_index]+'}'
            lines.append(f'{name} & '+' & '.join(formatted)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{',
        'Correctness-aware preference on held-out Granola-EQ relations, in percent.',
        'FT and Base denote wins; Unres. denotes unresolved comparisons.',
        'Bold highlights the higher FT/Base win rate within each row.',
        'Both measures use the same 2,654 first-sentence answer pairs per model and identical correctness judgments.',
        'Correct answers are preferred to abstentions, and abstentions to incorrect answers;',
        'when both answers are correct, the finer-grained answer is preferred.',
        'Ties include equally fine-grained correct answers and two abstentions.',
        'Both-wrong pairs and unresolved comparisons are reported separately.',
        'Percentages use all pairs as the denominator, not only decisive comparisons.',
        r'}', r'\label{tab:granola-preference-detailed}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def latex_human_table(report, labels, records):
    """Pooled human outcomes; percentages include every pair in each panel."""
    counts = Counter(r['label'] for r in labels)
    if sum(counts.values()) != len(labels) or not labels:
        raise ValueError('Human granularity labels required')
    if len(records) != report['overall']['pairs']:
        raise ValueError('Human correctness records must match the preference cohort')
    correct = {v: sum(r[field] == 'Correct' for r in records)
               for v, field in [('base', 'base_correctness'), ('finetuned', 'finetuned_correctness')]}
    panels = [
        ('Granularity: all datasets', len(labels), counts,
         [('base_more_specific', 'FT coarser'), ('equal', 'Equal granularity'),
          ('finetuned_more_specific', 'FT finer'), ('abstention_present', 'Abstention present'),
          ('not_comparable', 'Not comparable')]),
        ('Correctness: Granola-EQ', len(records), correct,
         [('base', 'Base correct'), ('finetuned', 'FT correct')]),
        ('Preference: Granola-EQ', report['overall']['pairs'], report['overall']['counts'],
         [('finetuned', 'FT preferred'), ('base', 'Base preferred'), ('tie', 'Tie'),
          ('both_incorrect', 'Both incorrect'), ('unresolved', 'Unresolved')])]
    lines = [r'\begin{table}[t]', r'\small', r'\centering',
             r'\setlength{\tabcolsep}{6pt}', r'\begin{tabular}{@{}lrr@{}}',
             r'\toprule', r'Outcome & $n$ & \% \\']
    for title, n, values, rows in panels:
        if not title.startswith('Correctness:') and sum(values.values()) != n:
            raise ValueError('Human outcome counts do not sum to panel denominator')
        lines += [r'\midrule', r'\multicolumn{3}{@{}l}{\textit{'+title+f' ($N={n}$)'+r'}} \\',
                  r'\addlinespace[2pt]']
        lines += [f'{label} & {values[key]} & {100*values[key]/n:.1f}'+r' \\' for key, label in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\caption{',
        'Pooled human outcomes across four models. Granularity uses adjudicated labels for all 200 pairs; coarser and finer refer to FT relative to Base.',
        'Correctness uses final human judgments on the 68 Granola-EQ questions; abstentions and undetermined answers remain in the denominator.',
        'Granola-EQ preference combines human correctness labels with adjudicated granularity: correct answers rank above abstentions above incorrect answers, with finer answers preferred when both are correct.',
        'Percentages include all pairs in each panel, including non-comparable and unresolved cases.',
        r'}', r'\label{tab:human-pooled-outcomes}', r'\end{table}']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=('lfm', 'qwen', 'gemma', 'ministral'),
                        default=['lfm', 'qwen', 'gemma', 'ministral'])
    parser.add_argument('--generations-dir', type=Path, default=ROOT/'generations')
    parser.add_argument('--judged-dir', type=Path, default=ROOT/'correctness-judge/results/first-sentence')
    parser.add_argument('--metric', type=Path, default=ROOT/'human-study/results/metric.json')
    parser.add_argument('--study-dir', type=Path, default=ROOT/'human-study')
    parser.add_argument('--granularity-judgments', type=Path, default=ROOT/'granularity-judge/results/first-sentence/predictions.jsonl')
    parser.add_argument('--human', action='store_true', help='Use human correctness and adjudicated granularity, not the judge/metric')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Choose distinct models')
    if args.human:
        records, reference_only = compare_human(args.models, args.study_dir)
        report = {'rule': RULE.replace('calibrated metric', 'adjudicated human label')+
                         ' If both Correct but human granularity is not comparable, unresolved.',
                  'scope': 'first_sentence', **breakdown(records),
                  'reference_only': breakdown(reference_only),
                  'interpretation': 'Human-derived preference: human correctness plus adjudicated granularity; '
                                    'not a directly annotated preference question. Granola only.'}
    else:
        metric = json.loads(args.metric.read_text())
        records = compare(args.models, args.generations_dir, args.judged_dir, metric, args.study_dir)
        judge_records = compare_judge(records, read_rows(args.granularity_judgments))
        report = {'rule': RULE, 'scope': 'first_sentence', 'parameters': metric['parameters'],
                  **breakdown(records),
                  'matched_resolved': matched_resolved(records, judge_records),
                  'judge': {**breakdown(judge_records),
                      'excluding_calibration_questions': breakdown([r for r in judge_records if not r['calibration_question']])},
                  'excluding_calibration_questions': breakdown([r for r in records if not r['calibration_question']]),
                  'interpretation': 'Judge-and-metric-derived preference, not direct human preference. '
                                    'FT share of decisive pairs is not an unconditional win rate.'}
    output = args.output_dir or ROOT/('results/human-preference' if args.human else 'results/preference')
    write(output/'predictions.jsonl', records, rows=True)
    if args.human:
        write(output/'reference-only-predictions.jsonl', reference_only, rows=True)
        (output/'table.tex').write_text(latex_human_table(
            report, read_rows(args.study_dir/'results/final_labels.jsonl'), records), encoding='utf-8')
    else:
        write(output/'judge-predictions.jsonl', judge_records, rows=True)
        (output/'table.tex').write_text(latex_table(report), encoding='utf-8')
        (output/'table-detailed.tex').write_text(latex_detailed_table(report), encoding='utf-8')
    write(output/'summary.json', report)
    (output/'summary.txt').write_text(text_summary(report), encoding='utf-8')
    print(text_summary(report))


if __name__ == '__main__':
    main()
