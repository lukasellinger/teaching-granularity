"""Correctness-aware target-level matrices from saved Vital Persons outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {'lfm': 'LFM', 'ministral': 'Ministral', 'qwen': 'Qwen', 'gemma': 'Gemma'}
VARIANTS = {'base_plain': 'Base', 'base_idk_user_prompt': 'IDK prompt',
            'base_granularity_instruction': 'GAP', 'base_drag': 'DRAG', 'finetuned': 'FT (ours)'}
LEVELS = {'temporal': ['DAY', 'MONTH', 'YEAR', 'DECADE', 'CENTURY'],
          'location': ['CITY', 'REGION', 'COUNTRY', 'CONTINENT']}
EXTRA = ['abstain', 'wrong', 'pipeline_failure']


def matrix(rows, modality, fold_finer=True):
    """Fold only correct finer predictions; retain failures in row denominators."""
    levels = LEVELS[modality]
    counts = {t: dict.fromkeys(levels + EXTRA, 0) for t in levels}
    wrong = {t: dict.fromkeys(levels, 0) for t in levels}
    folded = dict.fromkeys(levels, 0)
    coverage = {'total': len(rows), 'included': 0, 'missing_gold': 0,
                'idk_target': 0, 'unavailable_target': 0}
    idk_outcomes = dict.fromkeys(['correct', 'wrong', 'abstain', 'pipeline_failure'], 0)
    seen = set()
    for row in rows:
        case_id = row['case_id']
        if case_id in seen:
            raise ValueError(f'Duplicate case ID: {case_id}')
        seen.add(case_id)
        if row.get('gold') is None:
            coverage['missing_gold'] += 1
            continue
        status = row['target_status']
        if status not in ('known', 'idk', 'unavailable'):
            raise ValueError(f'Unexpected target status: {status}')
        if status == 'unavailable':
            coverage['unavailable_target'] += 1
            continue
        parse_status = row['parse_status']
        if parse_status == 'idk':
            outcome = 'abstain'
        elif parse_status != 'parsed':
            outcome = 'pipeline_failure'
        else:
            if type(row.get('truth_correct')) is not bool:
                raise ValueError(f'Parsed gold-available case needs Boolean truth_correct: {case_id}')
            outcome = 'correct' if row['truth_correct'] else 'wrong'
        if status == 'idk':
            coverage['idk_target'] += 1
            idk_outcomes[outcome] += 1
            continue
        target = row['target_level']
        if target not in levels:
            raise ValueError(f'Unexpected {modality} target: {target}')
        coverage['included'] += 1
        if outcome in ('abstain', 'pipeline_failure'):
            counts[target][outcome] += 1
            continue
        predicted = row.get('prediction_level_for_target_metrics') or row['prediction']['level']
        if predicted not in levels:
            raise ValueError(f'Unexpected {modality} prediction: {predicted}')
        if outcome == 'wrong':
            counts[target]['wrong'] += 1
            wrong[target][predicted] += 1
        elif fold_finer and levels.index(predicted) <= levels.index(target):
            counts[target][target] += 1
            folded[target] += predicted != target
        else:
            counts[target][predicted] += 1
    totals = {t: sum(counts[t].values()) for t in levels}
    if sum(coverage[k] for k in ('included', 'missing_gold', 'idk_target', 'unavailable_target')) != coverage['total']:
        raise AssertionError('Coverage must partition all input rows')
    for t in levels:
        if sum(wrong[t].values()) != counts[t]['wrong']:
            raise AssertionError('Wrong matrix must decompose the Wrong column')
    normalize = lambda block, denominators: {t: {c: n/denominators[t] if denominators[t] else None
                                  for c, n in cells.items()} for t, cells in block.items()}
    wrong_totals = {t: sum(wrong[t].values()) for t in levels}
    display_columns = levels + ['abstain', 'wrong']
    display_counts = {t: {c: counts[t][c] for c in display_columns} for t in levels}
    display_totals = {t: sum(display_counts[t].values()) for t in levels}
    display = {'columns': display_columns, 'counts': display_counts,
               'row_totals': display_totals,
               'row_normalized': normalize(display_counts, display_totals),
               'excluded_pipeline_failures': {t: counts[t]['pipeline_failure'] for t in levels},
               'cell_labels': 'counts', 'shading': 'within-row percentage after excluding pipeline failures'}
    return {'display': display, 'columns': levels + EXTRA, 'counts': counts, 'row_totals': totals,
            'row_normalized': normalize(counts, totals), 'finer_correct_folded_into_diagonal': folded,
            'wrong_by_predicted_level': {'columns': levels, 'counts': wrong,
                                         'row_totals': wrong_totals,
                                         'denominator': 'Only parsed wrong answers at each target level.',
                                         'row_normalized': normalize(wrong, wrong_totals)},
            'coverage': coverage, 'idk_target_outcomes': idk_outcomes}


def collect(results_dir, models, variants, split, fold_finer=True):
    report = {
        'split': split,
        'scope': 'Known, non-IDK model-derived targets with external gold available.',
        'fold_finer': fold_finer,
        'definition': ('Correct at target or finer folds to diagonal; correct coarser retains level.'
                       if fold_finer else 'All correct answers retain their generated level; diagonal means exact target level.'),
        'wrong_definition': 'Parsed answer without gold support at its expressed level, including '
                            'compatible but unverifiable detail; not independently verified falsity.',
        'denominator': 'Main figure: included cases at each target level excluding pipeline failures. '
                       'Full pre-exclusion counts remain in the data. Wrong-answer matrix: only parsed '
                       'wrong answers at each target level; each nonempty row sums to 100%.',
        'models': {}, 'sources': {}, 'evaluation_inputs': {}}
    for model in models:
        summary_path = results_dir/model/split/'summary.json'
        if summary_path.exists():
            from dataclasses import asdict
            from shared.io import digest, file_hash
            from .data import load_cases
            cases = load_cases(model, split)
            summary = json.loads(summary_path.read_text())
            expected = digest({k: [asdict(c) for c in rows] for k, rows in cases.items()})
            if summary['inputs']['cases_sha256'] != expected:
                raise ValueError('Rescore with the current adapted gold before plotting')
            report['evaluation_inputs'][model] = {
                'summary_sha256': file_hash(summary_path),
                'cases_sha256': expected,
                'answer_status_judge': summary.get('answer_status_judge')}
        report['models'][model] = {}
        for modality in LEVELS:
            report['models'][model][modality] = {}
            identity = None
            for variant in variants:
                path = results_dir/model/split/'outcomes'/modality/f'{variant}.jsonl'
                raw = path.read_bytes()
                rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
                block = matrix(rows, modality, fold_finer=fold_finer)
                # Do not silently change the case grid, targets or gold between methods.
                current = {r['case_id']: {k: r.get(k) for k in
                           ('qid', 'target_status', 'target_level', 'gold')} for r in rows}
                if identity is not None and current != identity:
                    raise ValueError(f'Methods have different case grids, targets or gold: {model}/{modality}/{variant}')
                identity = current
                report['models'][model][modality][variant] = block
                report['sources'][f'{model}/{modality}/{variant}'] = {
                    'path': path.relative_to(results_dir).as_posix(),
                    'sha256': hashlib.sha256(raw).hexdigest()}
    return report


def collect_comparison(results_dir, models, split):
    """Match assessable Base/FT pairs before excluding failures from the display."""
    variants=['base_plain','finetuned']
    report=collect(results_dir,models,variants,split,fold_finer=False)
    report['paired_cohorts']={}
    report['scope']='Same known-target, gold-available cases assessable for both Base and FT.'
    for model in models:
        report['paired_cohorts'][model]={}
        for modality in LEVELS:
            rows={}
            for variant in variants:
                path=results_dir/model/split/'outcomes'/modality/f'{variant}.jsonl'
                rows[variant]={r['case_id']:r for r in map(json.loads,path.read_text().splitlines())}
            base=rows['base_plain']
            eligible={k for k,r in base.items() if r['gold'] is not None and r['target_status']=='known'}
            kept={k for k in eligible if all(rows[v][k]['parse_status'] in ('parsed','idk') for v in variants)}
            report['paired_cohorts'][model][modality]={
                'all_cases':len(base),'known_target_with_gold':len(eligible),'paired_cases':len(kept),
                'excluded_pipeline_failure_either_method':len(eligible-kept),
                'retained_case_ids':sorted(kept)}
            for variant in variants:
                report['models'][model][modality][variant]=matrix([rows[variant][k] for k in sorted(kept)],modality,fold_finer=False)
            assert (report['models'][model][modality]['base_plain']['display']['row_totals']==
                    report['models'][model][modality]['finetuned']['display']['row_totals'])
    return report


def draw(report, models, variant, output, wrong_only=False, comparison_modality=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import numpy as np

    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 9, 'pdf.fonttype': 42})
    # Domains are columns; models are rows. Shared column labels save space.
    fig, panels = plt.subplots(len(models), 2,
                               figsize=(7.4, 1.12*len(models)+1.4), squeeze=False)
    fig.subplots_adjust(left=.13, right=.985, top=.90, bottom=.23,
                        hspace=.22, wspace=.65)
    axes = {(i,j): panels[j,i] for j in range(len(models)) for i in (0,1)}
    specs = ([(comparison_modality,v) for v in ('base_plain','finetuned')] if comparison_modality
             else [(m,variant) for m in LEVELS])
    for i, (modality, panel_variant) in enumerate(specs):
        levels = LEVELS[modality]
        for j, model in enumerate(models):
            parent = report['models'][model][modality][panel_variant]
            block = parent['wrong_by_predicted_level'] if wrong_only else parent['display']
            columns = block['columns']
            totals = block['row_totals']
            values = np.array([[block['row_normalized'][t][c] if totals[t]
                                else np.nan for c in columns] for t in levels])
            pixels = np.ones((*values.shape, 3))
            for r, c in np.ndindex(values.shape):
                color = '#b65242' if wrong_only else ('#27825f' if c < len(levels) else
                        {'abstain': '#709bc6', 'wrong': '#b65242'}[columns[c]])
                pixels[r, c] = .92 if not np.isfinite(values[r, c]) else (
                    1-(1-np.array(to_rgb(color)))*values[r, c])
            ax = axes[i, j]
            ax.imshow(pixels, aspect='auto')
            # Keep temporal and geographic cell widths identical within each column.
            if i == 1 and comparison_modality is None:
                position = ax.get_position()
                temporal_width = len(LEVELS['temporal']) + (0 if wrong_only else len(EXTRA)-1)
                ax.set_position([position.x0, position.y0, position.width*len(columns)/temporal_width,
                                 position.height])
            for r, t in enumerate(levels):
                if not totals[t]:
                    ax.text((len(columns)-1)/2, r, 'No wrong answers' if wrong_only else 'No cases',
                            ha='center', va='center', fontsize=7.5, color='#666666')
                    continue
                for c, key in enumerate(columns):
                    if block['counts'][t][key]:
                        ax.text(c, r, str(block['counts'][t][key]), ha='center', va='center', fontsize=8,
                                color='white' if values[r,c] >= .65 else '#222222')
            labels = [x.title() for x in levels] + ([] if wrong_only else ['Abstain', 'Wrong'])
            ax.set_xticks(range(len(columns)), labels if j == len(models)-1 else [],
                          rotation=48, ha='right', fontsize=8)
            ax.set_yticks(range(len(levels)),
                          [f'{t.title()}  ({totals[t]})' for t in levels],
                          fontsize=8)
            if j == 0:
                title = VARIANTS[panel_variant] if comparison_modality else ('Temporal' if i == 0 else 'Location')
                ax.set_title(title, fontsize=11, pad=15)
            ax.text(-.055, 1.025, 'n', transform=ax.transAxes, ha='right', fontsize=7, color='#666666')
            ax.tick_params(length=0)
            if i == 0:
                ax.set_ylabel(MODELS[model], fontsize=10, fontweight='bold', labelpad=9)
            for spine in ax.spines.values():
                spine.set_visible(False)
    for j in range(len(models)-1):
        upper_bottom = min(axes[i,j].get_position().y0 for i in (0,1))
        lower_top = max(axes[i,j+1].get_position().y1 for i in (0,1))
        y = (upper_bottom+lower_top)/2
        fig.add_artist(Line2D([.015,.985], [y,y], transform=fig.transFigure,
                              color='#999999', linewidth=.4))
    handles = ([Patch(facecolor='#b65242', label='Wrong')] if wrong_only else
               [Patch(facecolor='#27825f', label='Correct'),
                Patch(facecolor='#709bc6', label='Abstain'),
                Patch(facecolor='#b65242', label='Wrong')])
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .105),
               ncol=len(handles), frameon=False, fontsize=8,
               handlelength=1.5, columnspacing=2)
    stem = f'{"wrong" if wrong_only else "correct"}-alignment-{variant}'
    if not wrong_only and not report.get('fold_finer', True):
        stem += '-unfolded'
    if comparison_modality:
        stem = f'correct-alignment-{comparison_modality}-base-ft-unfolded'
    for extension in ('pdf', 'png', 'svg'):
        fig.savefig(output/f'{stem}.{extension}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    caption = (
        'Generated granularity of parsed answers without gold support at their expressed level. '
        'Predicted levels are not folded. Cell numbers are counts; shading shows percentages conditional '
        'on wrong answers at each target level. Sample sizes count wrong answers '
        'only, not all eligible cases.' if wrong_only else
        f'Correctness-aware alignment of {VARIANTS[variant]} answers with model-derived targets. Models are arranged in rows, '
        'with temporal questions on the left and location questions on the right. Correct answers '
        'at the target level or finer are combined on the diagonal; correct coarser '
        'answers retain their generated level. Wrong answers and abstentions are separate outcomes. '
        'Pipeline failures are excluded from the displayed matrix and its denominators. ')
    if not wrong_only and not report.get('fold_finer', True):
        caption = (f'Correctness-aware alignment of {VARIANTS[variant]} answers with model-derived targets. '
                   'Models are arranged in rows, with temporal questions on the left and location questions '
                   'on the right. Correct answers retain their actual generated level: the diagonal means '
                   'correct at the target level, cells left of it mean finer and correct, and cells right '
                   'of it mean coarser and correct. Wrong answers and abstentions are separate outcomes. '
                   'Pipeline failures are excluded from the display and row denominators.')
    if comparison_modality:
        caption = (f'Correctness-aware alignment for {comparison_modality} questions: Base on the left and '
                   'FT (ours) on the right, with one row per model. Both methods use exactly the same '
                   'known-target, gold-available cases; pairs with a pipeline failure in either method are excluded. '
                   'Correct answers retain their actual generated level. The diagonal means correct at the target '
                   'level; correct finer and coarser answers remain left and right of it, respectively. '
                   'Abstentions and wrong answers are separate outcomes.')
    caption += (' Wrong denotes lack of gold support, including compatible but unverifiable detail, '
                'not independently verified falsity. Non-IDK targets with external gold only. '
                'Cell numbers are counts, with sample sizes shown beside rows; shading encodes within-row percentages. '
                'Blanks are zero and grey rows have no observations '
                'in the relevant group. '
                'Methods use identical cases within each model and modality, but targets differ across models.')
    (output/f'{stem}.tex').write_text(
        '\\begin{figure*}[t]\n\\centering\n'
        f'\\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n'
        f'\\caption{{{caption}}}\n\\label{{fig:{stem.replace("_", "-")}}}\n\\end{{figure*}}\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results-corrected')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures/alignment')
    parser.add_argument('--split', choices=('dev', 'test'), default='test')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--unfolded', action='store_true', help='Keep finer correct answers at their actual level; write separate unfolded figures')
    parser.add_argument('--compare-base-ft', action='store_true', help='Separate temporal/location unfolded Base-versus-FT figures on common assessable pairs')
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models) or len(set(args.variants)) != len(args.variants):
        parser.error('Models and variants must be distinct')
    if args.compare_base_ft:
        report=collect_comparison(args.results_dir,args.models,args.split)
        args.output_dir.mkdir(parents=True,exist_ok=True)
        for modality in LEVELS:
            draw(report,args.models,'finetuned',args.output_dir,comparison_modality=modality)
        (args.output_dir/'alignment-data-base-ft-unfolded.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Wrote separate temporal/location Base-versus-FT figures to {args.output_dir}')
        return
    report = collect(args.results_dir, args.models, args.variants, args.split, fold_finer=not args.unfolded)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        for wrong_only in ((False,) if args.unfolded else (False, True)):
            draw(report, args.models, variant, args.output_dir, wrong_only)
    data_name = 'alignment-data-unfolded.json' if args.unfolded else 'alignment-data.json'
    (args.output_dir/data_name).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {(1 if args.unfolded else 2)*len(args.variants)} matrices and exact counts to {args.output_dir}')


if __name__ == '__main__':
    main()
