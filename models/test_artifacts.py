"""Offline checks linking reported artifacts to their exact checkpoints/answers."""
import hashlib
import json
from pathlib import Path

import pytest

from shared.io import digest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('lfm', 'qwen', 'gemma', 'ministral')
VARIANTS = {'base_plain', 'base_idk_user_prompt', 'base_granularity_instruction', 'base_drag', 'finetuned'}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize('model', MODELS)
def test_main_references(model):
    ref = read(ROOT / 'models' / f'{model}.json')
    settings = read(ROOT / 'generalization/generation_settings.json')['models'][model]
    assert ref['repo_id'] == settings['adapter_repo']
    assert ref['revision'] == settings['adapter_revision']
    assert ref['subfolder'] == settings['adapter_subfolder']
    assert ref['base_revision'] == settings['revision']
    assert len(ref['revision']) == 40
    assert ref['revision'] in ref['url']
    for key in ('training_config', 'training_results'):
        assert (ROOT / ref[key]).exists()
    assert ref['vital_persons_results'] == f'evaluation/vital_persons/results/{model}'


@pytest.mark.parametrize('model', MODELS)
def test_main_saved_generation_references(model, require_local_files):
    ref = read(ROOT / 'models' / f'{model}.json')
    required = [ROOT/path for path in ref['generalization_generations'].values()]
    for split in ('dev', 'test'):
        directory = ROOT/ref['vital_persons_results']/split/'generations'
        required.extend(directory/modality/f'{variant}.jsonl'
                        for modality in ('temporal', 'location') for variant in VARIANTS)
        required.extend(directory/modality/'finetuned.run.json' for modality in ('temporal', 'location'))
    require_local_files(required, reason='Saved evaluation generations are local-only')
    for path in ref['generalization_generations'].values():
        assert (ROOT / path).is_file()
    for split in ('dev', 'test'):
        directory = ROOT / ref['vital_persons_results'] / split
        for modality in ('temporal', 'location'):
            assert {p.stem for p in (directory / 'generations' / modality).glob('*.jsonl')} == VARIANTS
            signature = read(directory / 'generations' / modality / 'finetuned.run.json')['signature']
            assert signature['finetuned_artifact_sha256'] == ref['adapter_sha256']


RESULTS = sorted((ROOT / 'evaluation/vital_persons/results').glob('*/*/summary.json')) + sorted((ROOT / 'ablations').rglob('summary.json'))
RESULTS = [p.parent for p in RESULTS if (p.parent / 'outcomes').exists()]


def generation_sources():
    """Resolve rescored runs to the same source directories as their producers."""
    from ablations.source_evaluation import OUT as source_output, sources
    from ablations.greedy_table import OUT as control_output, entries
    from ablations.objective_table import BASE as short_base, OUT as short_output

    result = {directory: directory / 'generations' for directory in RESULTS
              if (directory / 'generations').is_dir()}
    for output, runs in ((source_output, sources()), (control_output, entries())):
        for name, original, _ in runs:
            result[output / 'rescored' / name] = original / 'generations'
    # Include archived short-run conditions as well as the displayed conditions.
    for config in short_base.glob('*/config.json'):
        result[short_output / 'rescored' / config.parent.name] = config.parent / 'dev/generations'
    return result


GENERATION_SOURCES = generation_sources()


def assert_utility_metrics(saved, fresh):
    for key, value in saved.items():
        if key not in fresh:
            continue
        if isinstance(value, dict) and isinstance(fresh[key], dict):
            assert_utility_metrics(value, fresh[key])
        elif 'utility' in key:
            assert value == (pytest.approx(fresh[key], abs=1e-12) if fresh[key] is not None else None), key


@pytest.mark.parametrize('directory', RESULTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_saved_answers_and_scores(directory):
    summary = read(directory / 'summary.json')
    generation_dir = GENERATION_SOURCES[directory]
    for modality, report in summary['modalities'].items():
        for variant in report['variants']:
            answers = rows(generation_dir / modality / f'{variant}.jsonl')
            outcomes = rows(directory / 'outcomes' / modality / f'{variant}.jsonl')
            assert len({r['case_id'] for r in answers}) == len(answers)
            assert {r['case_id']: r['answer'] for r in answers} == {r['case_id']: r['answer'] for r in outcomes}
            recorded = read(generation_dir / modality / f'{variant}.run.json')
            assert recorded['status'] == 'complete'
            entry = report['variants'][variant]
            if 'metrics' in entry:
                metrics = entry['metrics']
                source_key = f'{modality}/{variant}'
                assert digest(answers) == summary['inputs']['generation_sha256'][source_key]
                assert recorded == summary['inputs']['generation_runs'][source_key]
                from evaluation.vital_persons.report import headline
                assert headline(metrics, outcomes, modality) == entry['headline']
            else:
                metrics = read(directory / 'metrics' / modality / f'{variant}.json')['metrics']
            if modality == 'location' and 'location_informativeness' in summary:
                from evaluation.vital_persons.location import aggregate_location_outcomes
                assert_utility_metrics(metrics, aggregate_location_outcomes(outcomes))
            assert metrics['cases'] == len(outcomes)
            for section in report['variants'][variant].values():
                if isinstance(section, dict):
                    for key, value in section.items():
                        if key in metrics:
                            assert metrics[key] == value, (directory, modality, variant, key)


@pytest.mark.parametrize('directory', sorted((ROOT / 'ablations').rglob('config.json')), ids=lambda p: str(p.parent.relative_to(ROOT)))
def test_ablation_checkpoint_links(directory):
    directory = directory.parent
    ref = read(directory / 'model.json')
    config = read(directory / 'config.json')
    selected = read(directory / 'training.json')
    assert selected['best_checkpoint_step'] <= selected['completed_global_step']
    for path in directory.rglob('finetuned.run.json'):
        assert read(path)['signature']['finetuned_artifact_sha256'] == ref['adapter_sha256']
    assert 'adapter_path' not in ref
    assert len(ref['revision']) == 40
    assert ref['revision'] in ref['url']
    digest = hashlib.sha256()
    for name, value in sorted(ref['files'].items()):
        digest.update(name.encode())
        digest.update(value.encode())
    assert digest.hexdigest() == ref['adapter_sha256']


def test_matched_control_case_alignment(require_local_files):
    root = ROOT / 'ablations/lfm/greedy-control'
    require_local_files(
        [root/arm/'data'/f'{split}.jsonl'
         for arm in ('consensus', 'greedy') for split in ('train', 'dev', 'test')],
        reason='Rebuild the matched-control datasets with data_preparation.vital_persons.greedy_control')
    for split in ('train', 'dev', 'test'):
        def identities(arm):
            return {(r['qid'], r['relation']) for r in rows(root / arm / 'data' / f'{split}.jsonl')}
        assert identities('consensus') == identities('greedy')
    for arm in ('consensus', 'greedy'):
        assert read(root / arm / 'config.json')['idk_sft_weight'] == 0.5
