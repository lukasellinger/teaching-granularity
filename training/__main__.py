"""Train one primary-target model using a recorded configuration."""

import argparse
import json
import os
from pathlib import Path

from shared.io import write, file_hash
from .configuration import DIRECTORY, REPO, MODELS, load_config, load_inputs, check_data, identity


def resume_checkpoint(output, resume, contract):
    metadata = output / "run.json"
    if not resume:
        if output.exists():
            raise ValueError("Output already exists; choose a new directory or --resume")
        return None
    if not metadata.is_file() or json.loads(metadata.read_text())["contract"] != contract:
        raise ValueError("Resume configuration, data or training code differs")
    checkpoints = [path for path in (output / "checkpoints").glob("checkpoint-*")
                   if path.name.removeprefix("checkpoint-").isdigit() and (path / "trainer_state.json").is_file()]
    if not checkpoints:
        raise ValueError("No resumable checkpoint exists")
    return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[1])))


def training_argument_values(config, args):
    """Recorded budgets include short fixed-step and epoch-based source runs."""
    return dict(
        output_dir=str(args.output_dir / 'checkpoints'),
        num_train_epochs=config['epochs'], max_steps=config['max_steps'],
        per_device_train_batch_size=config['batch_size'], per_device_eval_batch_size=config['eval_batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'], learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_ratio'], lr_scheduler_type='linear', logging_steps=5,
        eval_strategy='steps', eval_steps=config['eval_steps'], save_strategy='steps', save_steps=config['eval_steps'],
        save_total_limit=config['save_total_limit'], load_best_model_at_end=True,
        metric_for_best_model='eval_objective_loss', greater_is_better=False,
        report_to=['wandb'] if args.wandb else [], run_name=args.output_dir.name,
        seed=config['seed'], data_seed=config['seed'], remove_unused_columns=False, bf16=True, fp16=False,
        gradient_checkpointing=config['gradient_checkpointing'], gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, optim='adamw_torch')


def early_stopping_callbacks(config):
    if config['early_stopping_patience'] is None:
        return []
    from .trainer import PrimaryMetricEarlyStoppingCallback
    return [PrimaryMetricEarlyStoppingCallback(early_stopping_patience=config['early_stopping_patience'],
                                               early_stopping_threshold=config['early_stopping_threshold'])]


def train(config, args):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    from dotenv import load_dotenv
    from .data import training_summary
    from .encoding import RealizationDataset, RealizationCollator
    from .models import lora_target_modules, validate_lora_coverage
    from shared.models import load_fast_tokenizer, causal_model_class
    from .runtime import base_model_scoring, preserve_rng_state
    from .trainer import GranularityTrainer

    if int(os.getenv("WORLD_SIZE", "1")) != 1:
        raise ValueError("The recorded runs use single-GPU training, not DDP/FSDP")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a BF16-capable CUDA GPU; --check-data needs no GPU")
    load_dotenv(REPO / ".env")
    contract = {"configuration": config, "code": {path.name: file_hash(path) for path in DIRECTORY.glob("*.py") if path.name != "tests.py"}}
    contract['code'].update({f'shared/{name}.py': file_hash(REPO/'shared'/f'{name}.py')
                             for name in ('chat', 'models', 'io', 'constants', 'text')})
    checkpoint = resume_checkpoint(args.output_dir, args.resume, contract)
    train_rows, dev_rows, granola_dev, profile = load_inputs(config)
    previous = json.loads((args.output_dir / "run.json").read_text()) if checkpoint else None
    revision = {"revision": config["revision"]} if config["revision"] else {}
    if config["model"] == "ministral":
        tokenizer = load_fast_tokenizer(config["model_id"], config["revision"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], use_fast=True, **revision)
    if not tokenizer.is_fast:
        raise ValueError("Training requires offset mappings from a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define PAD or EOS")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    common = {"seed": config["seed"], "max_length": config["max_length"],
              "system_prompt": config["system_prompt"], "user_prompt_template": config["user_prompt_template"],
              "rank_balance_factors": profile.level_factors}
    train_dataset = RealizationDataset(train_rows, tokenizer,
        max_negatives_per_group=0 if config["objective"] == "sft" else config["train_negatives_per_group"], **common)
    dev_dataset = RealizationDataset(dev_rows, tokenizer, max_negatives_per_group=None, **common)
    auxiliary = RealizationDataset(granola_dev, tokenizer, max_negatives_per_group=None, **common) if granola_dev is not None else None
    preflight = {"train": train_dataset.preflight(), "dev": dev_dataset.preflight(), "granola_dev": auxiliary.preflight() if auxiliary is not None else None}
    set_seed(config["seed"])
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", "teaching-granularity")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    training_args = TrainingArguments(**training_argument_values(config, args))
    _ = training_args.device
    loader = causal_model_class(config["model_id"])
    options = {"dtype": torch.bfloat16, **revision}
    if config["attention_implementation"]:
        options["attn_implementation"] = config["attention_implementation"]
    model = loader.from_pretrained(config["model_id"], **options)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,
        r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=lora_target_modules(config["source_model"]), bias="none"))
    if config["gradient_checkpointing"]:
        model.enable_input_require_grads()
    validate_lora_coverage(model, config["source_model"])
    resolved = getattr(model.config, "_commit_hash", None)
    if previous and previous.get("resolved_model_revision") != resolved:
        raise ValueError("Resolved base checkpoint changed since the saved run")
    trainer = GranularityTrainer(model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=dev_dataset, processing_class=tokenizer,
        data_collator=RealizationCollator(tokenizer), objective=config["objective"],
        contrastive_weight=config["contrastive_weight"], auxiliary_eval_dataset=auxiliary,
        callbacks=early_stopping_callbacks(config))
    metadata = {"contract": contract, "configuration": config, "resolved_model_revision": resolved,
                "train": training_summary(train_rows), "dev": training_summary(dev_rows),
                "granola_dev": training_summary(granola_dev) if granola_dev is not None else None, "tokenization_preflight": preflight,
                "rank_balance": profile.summary(), "checkpoint_selection": "vital_persons_dev",
                "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write(args.output_dir / "run.json", metadata)
    # Untouched-base diagnostics must not change the optimizer RNG trajectory.
    with preserve_rng_state(), base_model_scoring(model):
        base_metrics = trainer.evaluate(metric_key_prefix="base_eval")
        if auxiliary is not None:
            base_metrics.update(trainer.evaluate(eval_dataset=auxiliary, metric_key_prefix="base_eval_granola"))
    metadata["base_evaluation"] = base_metrics
    write(args.output_dir / "run.json", metadata)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(args.output_dir / "final")
    metadata.update(completed_global_step=trainer.state.global_step, best_model_checkpoint=trainer.state.best_model_checkpoint,
                    stopped_early=trainer.state.global_step < trainer.state.max_steps)
    write(args.output_dir / "run.json", metadata)
    print(f"Best-dev-loss adapter: {args.output_dir / 'final'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--config", type=Path, help="Optional explicit configuration for a separate experiment")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Verify datasets and weights, without Torch, a model or GPU")
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint with the same settings/data/code")
    parser.add_argument("--wandb", action="store_true", help="Optional W&B logging; credentials may be supplied via .env")
    args = parser.parse_args()
    config = load_config(args.model, args.config)
    if args.show_config:
        print(json.dumps(config, indent=2))
    elif args.check_data:
        print(json.dumps(check_data(config), indent=2))
    elif args.output_dir is None:
        parser.error("Provide --output-dir, --check-data or --show-config")
    else:
        train(config, args)


if __name__ == "__main__":
    main()
