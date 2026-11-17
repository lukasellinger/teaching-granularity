"""Score the same cases for every variant and summarize paired comparisons."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from shared.io import read_rows, write, digest
from .cases import Modality, temporal_case_summary, location_case_summary
from .generate import validate_records
from . import temporal, location
from .agreement import common_set_kappa_analysis


def headline(metrics, outcomes, modality):
    groups = {}
    for row in outcomes:
        if row['target_match'] is not None:
            groups.setdefault(row['target_category'], []).append(float(row['target_match']))
    positive = metrics[f'{modality}_utility_positive_informativeness']
    utility = metrics[f'{modality}_utility_uniform_wrong_penalty']
    return {
        'cases': len(outcomes), 'truth_accuracy': metrics['truth_accuracy'],
        'target_match_accuracy': metrics['target_match_accuracy'],
        'macro_target_match_accuracy': float(np.mean([np.mean(rows) for rows in groups.values()])) if groups else None,
        'target_match_by_category': {key: {'cases': len(rows), 'accuracy': float(np.mean(rows))} for key, rows in sorted(groups.items())},
        'positive_information_reward': positive,
        'error_penalty': utility-positive if utility is not None and positive is not None else None,
        'uniform_error_utility': utility,
    }


def score(cases_by_modality, variants, generation_dir, output_dir, *, pipeline=None,
          samples_by_modality=None, num_resamples=1000, confidence_level=.95, seed=42, baseline_dir=None,
          answer_status_predictions=None):
    def source_directory(variant):
        return baseline_dir if baseline_dir is not None and variant != 'finetuned' else generation_dir
    report = {'bootstrap': {'method': 'paired_qid_cluster_percentile', 'num_resamples': num_resamples,
              'confidence_level': confidence_level, 'seed': seed, 'difference': 'finetuned_minus_baseline'}, 'modalities': {}}
    # Validate all requested inputs before starting any extraction/normalization.
    inputs = {}
    status_index = None
    if answer_status_predictions is not None:
        from .answer_status.integration import load_predictions
        status_index = load_predictions(answer_status_predictions)
        report['answer_status_judge'] = {'predictions_sha256': digest(read_rows(answer_status_predictions)),
                                        'scope': 'full_answer', 'applied': True}
    for modality, cases in cases_by_modality.items():
        samples = samples_by_modality.get(modality) if samples_by_modality else None
        for variant in variants:
            path = source_directory(variant)/modality/f'{variant}.jsonl'
            rows = read_rows(path)
            validate_records(rows, cases, variant, samples, complete=True)
            inputs[modality, variant] = rows
    lines = ['Vital Persons — percentages / percentage points',
             'Modality  Variant  Truth  Target  Macro-target  Reward  Penalty  Utility']
    for modality, cases in cases_by_modality.items():
        scorer = temporal if modality == 'temporal' else location
        result = {'case_summary': (temporal_case_summary if modality == 'temporal' else location_case_summary)(cases),
                  'variants': {}, 'comparisons': {}}
        outcomes_by_variant = {}
        for variant in variants:
            generations = inputs[modality, variant]
            if modality == 'temporal':
                metrics, outcomes = temporal.score_temporal_generations(cases, generations)
            else:
                if pipeline is None:
                    raise ValueError('Location scoring requires an extraction/normalization pipeline')
                metrics, outcomes = pipeline.score(cases, generations)
            if status_index is not None:
                from .answer_status.integration import attach
                generations = attach(generations, cases, status_index, outcomes)
                if modality == 'temporal':
                    metrics, outcomes = temporal.score_temporal_generations(cases, generations)
                else:
                    metrics, outcomes = pipeline.score(cases, generations)
                decisions = {r['case_id']: r['answer_status_judge'] for r in generations}
                for row in outcomes:
                    row['answer_status_judge'] = decisions[row['case_id']]
            head = headline(metrics, outcomes, modality)
            outcomes_by_variant[variant] = outcomes
            result['variants'][variant] = {'metrics': metrics, 'headline': head}
            write(output_dir/'outcomes'/modality/f'{variant}.jsonl', outcomes, rows=True)
            fields = ('truth_accuracy', 'target_match_accuracy', 'macro_target_match_accuracy',
                      'positive_information_reward', 'error_penalty', 'uniform_error_utility')
            values = ['—' if head[key] is None else f'{100*head[key]:.2f}' for key in fields]
            lines.append(f"{modality}  {variant}  " + '  '.join(values))
        if 'finetuned' in variants and num_resamples:
            for baseline in variants:
                if baseline != 'finetuned':
                    result['comparisons'][baseline] = scorer.paired_qid_bootstrap(
                        outcomes_by_variant[baseline], outcomes_by_variant['finetuned'],
                        num_resamples=num_resamples, confidence_level=confidence_level, seed=seed)
        result['common_set_kappa'] = common_set_kappa_analysis(outcomes_by_variant, Modality(modality),
            num_resamples=num_resamples, confidence_level=confidence_level, seed=seed)
        report['modalities'][modality] = result
    if 'location' in cases_by_modality:
        report['location_informativeness'] = location._LOCATION_INFORMATION
    report['inputs'] = {'generation_sha256': {f'{m}/{v}': digest(rows) for (m, v), rows in inputs.items()},
                        'generation_runs': {f'{m}/{v}': json.loads(path.read_text())
                            for m, v in inputs if (path := source_directory(v)/m/f'{v}.run.json').is_file()},
                        'cases_sha256': digest({key: [asdict(case) for case in cases] for key, cases in cases_by_modality.items()})}
    write(output_dir/'summary.json', report)
    output_dir.joinpath('summary.txt').write_text('\n'.join(lines)+'\n')
    return report
