"""Every reported condition must run through shared training/evaluation paths."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from training.configuration import REPO, load_config, check_data, load_inputs
from training.__main__ import training_argument_values, early_stopping_callbacks
from shared.models import adapter_reference
from evaluation.vital_persons.generate import settings
from training.tests.test_configuration import assert_same

CONDITIONS = sorted((REPO / 'ablations/lfm').glob('*/*/config.json'))


@pytest.mark.parametrize('path', CONDITIONS, ids=lambda p: str(p.parent.relative_to(REPO / 'ablations/lfm')))
def test_exact_recorded_training_inputs_and_weights(path, require_training_inputs):
    config = load_config('lfm', path)
    reference = json.loads(path.with_name('training.json').read_text())
    require_training_inputs(config)
    actual = check_data(config)
    for key in ('train', 'dev', 'granola_dev', 'rank_balance'):
        assert_same(actual[key], reference[key])
    args = SimpleNamespace(output_dir=Path('runs/test'), wandb=False)
    values = training_argument_values(config, args)
    assert values['max_steps'] == config['max_steps']
    assert values['num_train_epochs'] == config['epochs']
    assert values['metric_for_best_model'] == 'eval_objective_loss'
    assert values['load_best_model_at_end'] is True
    if '/short/' in str(path):
        assert values['max_steps'] == 600
        assert config['early_stopping_patience'] is None
        assert early_stopping_callbacks(config) == []


def test_recognition_only_is_epoch_based_but_selects_on_vital_dev(require_training_inputs):
    config = load_config('lfm', REPO / 'ablations/lfm/sources/recognition-only/config.json')
    require_training_inputs(config)
    train, dev, auxiliary, _ = load_inputs(config)
    assert config['max_steps'] == -1 and config['epochs'] == 20 and config['eval_steps'] == 5
    assert {r.example.data_source for r in train} == {'granola'}
    assert all(not hasattr(r.example, 'data_source') for r in dev)
    assert auxiliary
    pytest.importorskip('torch')
    pytest.importorskip('transformers')
    assert len(early_stopping_callbacks(config)) == 1


@pytest.mark.parametrize('path', CONDITIONS)
def test_ablation_adapter_does_not_fall_back_to_main_model(path):
    reference = json.loads(path.with_name('model.json').read_text())
    actual = adapter_reference(settings('lfm'), path.with_name('model.json'))
    assert actual['adapter_subfolder'] == reference['subfolder'] != 'lfm'
    assert actual['adapter_revision'] == reference['revision']


def test_wrong_model_or_unpinned_adapter_is_rejected(tmp_path):
    reference = json.loads((CONDITIONS[0].with_name('model.json')).read_text())
    file = tmp_path / 'model.json'
    file.write_text(json.dumps({**reference, 'model_id': 'wrong/base'}))
    with pytest.raises(ValueError, match='base model'):
        adapter_reference(settings('lfm'), file)
    file.write_text(json.dumps({**reference, 'revision': 'main'}))
    with pytest.raises(ValueError, match='full Hub commit'):
        adapter_reference(settings('lfm'), file)


def test_vital_cli_can_dry_run_released_ablation_without_weights(tmp_path, monkeypatch, capsys):
    from evaluation.vital_persons import __main__ as cli
    ref = REPO / 'ablations/lfm/short/sft/model.json'
    monkeypatch.setattr(sys, 'argv', ['vital', '--model', 'lfm', '--split', 'dev', '--variants', 'base_plain', 'finetuned',
        '--adapter-ref', str(ref), '--output-dir', str(tmp_path / 'out'), '--dry-run'])
    cli.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan['generation']['adapter_subfolder'] == 'ablations/lfm/short/sft'
    assert plan['cases'] == {'temporal': 328, 'location': 328}
    assert not (tmp_path / 'out').exists()


def test_generalization_cli_accepts_released_ablation_without_weights(tmp_path, monkeypatch):
    from generalization import generate
    import shared.models
    calls = []
    original = shared.models.adapter_reference
    monkeypatch.setattr(shared.models, 'adapter_reference', lambda *a: calls.append(original(*a)) or calls[-1])
    monkeypatch.setattr(generate, 'load_cases', lambda _: [{'id': '1', 'qid': 'Q1', 'question': 'Question?', 'relation': 'P50'}])
    monkeypatch.setattr(generate, 'load_model', lambda *a: pytest.fail('Dry run loaded weights'))
    monkeypatch.setattr(sys, 'argv', ['generalization', '--models', 'lfm', '--adapter-ref',
        str(REPO / 'ablations/lfm/sources/recognition-only/model.json'), '--output-dir', str(tmp_path), '--dry-run'])
    generate.main()
    assert calls[0]['adapter_subfolder'] == 'ablations/lfm/sources/recognition-only'
    assert not list(tmp_path.glob('**/*.jsonl'))


def test_proxy_inputs_include_complete_vital_and_granola_test_candidates(require_local_files):
    from training.evaluate import evaluation_inputs
    config = load_config('lfm', REPO / 'ablations/lfm/sources/consistency-only/config.json')
    require_local_files(
        [REPO/'data_preparation/granola/datasets/lfm'/name for name in ('dev.jsonl', 'test.jsonl')],
        reason='Reconstruct the LFM Granola splits with data_preparation.granola.restore')
    inputs, _ = evaluation_inputs(config, 'test', 'granola')
    assert len(inputs['granola'][0]) == 41
    inputs, _ = evaluation_inputs(config, 'dev', 'both')
    assert len(inputs['vital_persons'][0]) == 1359
    assert len(inputs['granola'][0]) == 8
    assert all(r.example.contrastive_groups for r in inputs['granola'][0])


def test_scoring_reuses_external_base_generations(tmp_path):
    from evaluation.vital_persons.data import load_cases
    from evaluation.vital_persons.report import score
    from evaluation.vital_persons.generate import variant_prompt
    from shared.io import write
    cases = load_cases('lfm', 'dev')['temporal'][:2]
    base_dir, ft_dir = tmp_path / 'base', tmp_path / 'ft'
    for variant, directory in (('base_plain', base_dir), ('finetuned', ft_dir)):
        rows = []
        for case in cases:
            prompt, system = variant_prompt(variant, case.prompt)
            rows.append({'case_id': case.case_id, 'variant': variant, 'answer': 'IDK',
                         'user_prompt': prompt, 'system_prompt': system, 'generated_tokens': 1, 'finish_reason': 'eos'})
        write(directory / 'temporal' / f'{variant}.jsonl', rows, rows=True)
    report = score({'temporal': cases}, ['base_plain', 'finetuned'], ft_dir, tmp_path / 'scored',
                   baseline_dir=base_dir, num_resamples=0)
    assert report['modalities']['temporal']['variants']['base_plain']['metrics']['cases'] == 2
    assert not (ft_dir / 'temporal/base_plain.jsonl').exists()


def test_proxy_evaluation_uses_shared_trainer_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from contextlib import nullcontext
    import shared.models
    import training.runtime
    from training import evaluate
    from training.data import weight_realizations, rank_balance_profile
    from training.tests.test_data import CharacterTokenizer, example
    torch.set_num_threads(2)
    rows = weight_realizations([example('case:0', 'Q1', 'A was born in 1900.', rankable=True, prompt_template_id=0)], 'sft_contrastive')
    path = tmp_path / 'input.jsonl'
    path.write_text('{}\n')
    monkeypatch.setattr(evaluate, 'evaluation_inputs', lambda *a: ({'vital_persons': (rows, path)}, rank_balance_profile(rows, 'none')))
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=256, n_positions=512,
        n_embd=8, n_layer=1, n_head=1, bos_token_id=0, eos_token_id=0, pad_token_id=0))
    monkeypatch.setattr(shared.models, 'resolve_adapter', lambda *a: tmp_path)
    monkeypatch.setattr(shared.models, 'generation_tokenizer', lambda *a: CharacterTokenizer())
    monkeypatch.setattr(shared.models, 'load_inference_model', lambda *a: model)
    monkeypatch.setattr(training.runtime, 'base_model_scoring', lambda model: nullcontext())
    config = load_config('lfm', REPO / 'ablations/lfm/short/sft/config.json')
    args = SimpleNamespace(split='dev', dataset='vital_persons', adapter=None, settings=settings('lfm'),
                           device='cpu', output_dir=tmp_path / 'results')
    report = evaluate.evaluate(config, args)
    for variant in ('base', 'finetuned'):
        assert 'eval_sft_loss' in report['datasets']['vital_persons'][variant]
        assert 'eval_natural_contrastive_accuracy' in report['datasets']['vital_persons'][variant]
    assert (args.output_dir / 'summary.json').is_file()
