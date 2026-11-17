"""Load recorded settings and check the exact prepared training inputs."""
from shared.constants import GRANOLA_TRAIN_RELATIONS, MODEL_NAMES
from shared.io import file_hash, iter_rows

import hashlib
import json
from pathlib import Path

from .data import load_training_examples, weight_realizations, rank_balance_profile, training_summary

DIRECTORY = Path(__file__).resolve().parent
REPO = DIRECTORY.parent
MODELS = ("lfm", "qwen", "gemma", "ministral")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_config(model, path=None):
    config = json.loads((Path(path) if path else DIRECTORY / "configs" / f"{model}.json").read_text())
    config.setdefault('source_model', MODEL_NAMES[model])
    if config["model"] != model:
        raise ValueError("Configuration does not match the selected model")
    if config["system_prompt"] is not None or config["user_prompt_template"] != "{question}":
        raise ValueError("This training path uses the paper's user-only questions")
    if config["objective"] not in {"sft", "contrastive", "sft_contrastive"}:
        raise ValueError("Unknown training objective")
    if config["dtype"] != "bfloat16":
        raise ValueError("The recorded training path uses bfloat16")
    for key in ("batch_size", "eval_batch_size", "gradient_accumulation_steps", "max_length", "eval_steps", "lora_rank", "lora_alpha"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if type(config['max_steps']) is not int or (config['max_steps'] != -1 and config['max_steps'] < 1) or config['epochs'] <= 0:
        raise ValueError('Use positive max_steps or -1 with a positive epoch budget')
    patience = config['early_stopping_patience']
    if patience is not None and (type(patience) is not int or patience < 1):
        raise ValueError('Early-stopping patience must be positive or null (disabled)')
    if config.get('granola_only') and not config['granola_data_dir']:
        raise ValueError('Granola-only training requires Granola data')
    if config['rank_balance'] not in {'none', 'hierarchical-sqrt'} or config['rank_balance_cap'] < 1:
        raise ValueError('Invalid rank balance mode/cap')
    if not 0 <= config["idk_sft_weight"] <= 1 or not 0 < config["granola_loss_fraction"] < 1:
        raise ValueError("Invalid IDK/source weight")
    if not 0 <= config["lora_dropout"] < 1 or not 0 <= config["warmup_ratio"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid learning-rate/LoRA setting")
    return config


def jsonl(path):
    return list(iter_rows(path))


def load_inputs(config):
    vital = REPO / config['data_dir']
    granola = REPO / config['granola_data_dir'] if config['granola_data_dir'] else None
    for filename, expected in config["expected_data_sha256"].items():
        path = granola / filename[len("granola/"):] if filename.startswith("granola/") else vital / filename
        if file_hash(path) != expected:
            raise ValueError(f"Training input checksum mismatch: {path}")
    primary = {split: load_training_examples(vital / f"{split}.jsonl") for split in ("train", "dev")}
    auxiliary = {split: load_training_examples(granola / f"{split}.jsonl") for split in ("train", "dev")} if granola else {}
    target_splits = {}
    for row in jsonl(vital / "case_targets.jsonl"):
        previous = target_splits.setdefault(row["qid"], row["split"])
        if previous != row["split"]:
            raise ValueError("Vital Persons identity appears in different splits")
    held_out_entities = set(json.loads((granola / "manifest.json").read_text())["held_out_entity_qids"]) if granola else set()
    auxiliary_splits = {}
    for split in ("train", "dev", "test") if granola else ():
        for row in jsonl(granola / f"{split}.jsonl"):
            if row["qid"] in target_splits:
                raise ValueError("Granola/Vital Persons entity overlap")
            if auxiliary_splits.setdefault(row["qid"], split) != split:
                raise ValueError("Granola entity appears in different splits")
            if split in {"train", "dev"} and (row["relation"] not in GRANOLA_TRAIN_RELATIONS or row["qid"] in held_out_entities):
                raise ValueError("Held-out Granola relation/entity entered training or dev")
    for source, splits in (("vital_persons", primary), ("granola", auxiliary)):
        for split, rows in splits.items():
            for row in rows:
                if row.source_model != config["source_model"] or row.split.value != split:
                    raise ValueError("Training source model/split mismatch")
                if source == "vital_persons" and target_splits.get(row.qid) != split:
                    raise ValueError("Vital Persons training row is outside its assigned split")
    examples = auxiliary['train'] if config.get('granola_only') else [*primary['train'], *auxiliary.get('train', [])]
    fraction = config['granola_loss_fraction'] if granola and not config.get('granola_only') else None
    train = weight_realizations(examples, config["objective"],
                                granola_loss_fraction=fraction, idk_sft_weight=config["idk_sft_weight"])
    balance_reference = weight_realizations(examples, "sft_contrastive",
                                            granola_loss_fraction=fraction, idk_sft_weight=config["idk_sft_weight"])
    # Dev uses unit IDK weight and no source mix, exactly as in the recorded runs.
    dev = weight_realizations(primary["dev"], "sft_contrastive")
    granola_dev = weight_realizations(auxiliary["dev"], "sft_contrastive") if granola else None
    profile = rank_balance_profile(balance_reference, config["rank_balance"], cap=config["rank_balance_cap"])
    return train, dev, granola_dev, profile


def check_data(config):
    train, dev, granola_dev, profile = load_inputs(config)
    return {"model": config["model"], "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
            "train": training_summary(train), "dev": training_summary(dev),
            "granola_dev": training_summary(granola_dev) if granola_dev is not None else None, "rank_balance": profile.summary()}
