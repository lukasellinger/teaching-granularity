"""Teacher-forced base/adapter proxy evaluation on prepared dev/test examples."""
import argparse
import json
from pathlib import Path

from shared.io import file_hash, write
from .configuration import REPO, load_config, load_inputs
from .data import load_training_examples, weight_realizations


def evaluation_inputs(config, split, dataset):
    _, _, _, profile = load_inputs(config)
    result = {}
    for source in ('vital_persons', 'granola') if dataset == 'both' else (dataset,):
        directory = config['data_dir'] if source == 'vital_persons' else (config['granola_data_dir'] or f"data_preparation/granola/datasets/{config['model']}")
        path = REPO / directory / f'{split}.jsonl'
        examples = load_training_examples(path)
        if any(r.source_model != config['source_model'] or r.split.value != split for r in examples):
            raise ValueError('Evaluation source model/split mismatch')
        # Always include every candidate and unit IDK weights, as in training-time dev diagnostics.
        result[source] = (weight_realizations(examples, 'sft_contrastive'), path)
    return result, profile


def evaluate(config, args):
    from transformers import TrainingArguments
    from shared.models import generation_tokenizer, load_inference_model, resolve_adapter
    from .encoding import RealizationDataset, RealizationCollator
    from .trainer import GranularityTrainer
    from .runtime import base_model_scoring, preserve_rng_state
    inputs, profile = evaluation_inputs(config, args.split, args.dataset)
    adapter = resolve_adapter(args.settings, args.adapter)
    tokenizer = generation_tokenizer(args.settings, adapter)
    tokenizer.padding_side = 'right'
    model = load_inference_model(args.settings, args.device, adapter)
    trainer = GranularityTrainer(model=model, objective=config['objective'], contrastive_weight=config['contrastive_weight'],
        args=TrainingArguments(output_dir=str(args.output_dir / 'runtime'), use_cpu=args.device == 'cpu',
            per_device_eval_batch_size=config['eval_batch_size'], bf16=args.device.startswith('cuda'),
            remove_unused_columns=False, report_to=[], dataloader_num_workers=0),
        processing_class=tokenizer, data_collator=RealizationCollator(tokenizer))
    report = {'configuration': config, 'adapter': args.settings, 'split': args.split, 'datasets': {}}
    for source, (rows, path) in inputs.items():
        dataset = RealizationDataset(rows, tokenizer, seed=config['seed'], max_negatives_per_group=None,
            max_length=config['max_length'], system_prompt=config['system_prompt'],
            user_prompt_template=config['user_prompt_template'], rank_balance_factors=profile.level_factors)
        preflight = dataset.preflight()
        with preserve_rng_state(), base_model_scoring(model):
            base = trainer.evaluate(eval_dataset=dataset, metric_key_prefix='eval')
        finetuned = trainer.evaluate(eval_dataset=dataset, metric_key_prefix='eval')
        report['datasets'][source] = {'data_sha256': file_hash(path), 'preflight': preflight,
                                     'base': base, 'finetuned': finetuned}
    write(args.output_dir / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('lfm', 'qwen', 'gemma', 'ministral'), required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--dataset', choices=('vital_persons', 'granola', 'both'), default='vital_persons')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--adapter', type=Path, help='Local adapter directory or completed training run')
    group.add_argument('--adapter-ref', type=Path, help='Released model.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry-run', action='store_true', help='Check all evaluation data without weights or model downloads')
    args = parser.parse_args()
    config = load_config(args.model, args.config)
    from evaluation.vital_persons.generate import settings
    from shared.models import adapter_reference
    args.settings = adapter_reference(settings(args.model, adapter=args.adapter), args.adapter_ref)
    args.settings['attention_implementation'] = config['attention_implementation']
    if args.device == 'cpu':
        args.settings['dtype'] = 'float32'
    if args.adapter:
        args.adapter = Path(args.settings['local_adapter'])
    if args.dry_run:
        inputs, _ = evaluation_inputs(config, args.split, args.dataset)
        print(json.dumps({'model': args.model, 'split': args.split, 'adapter': args.settings,
                          'datasets': {key: len(rows) for key, (rows, _) in inputs.items()}}, indent=2))
        return
    if args.output_dir.exists():
        parser.error('Output already exists; use a new output directory')
    print(json.dumps(evaluate(config, args), indent=2))


if __name__ == '__main__':
    main()
