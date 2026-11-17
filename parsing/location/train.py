"""Train the location extractor, filter or rewriter from its released train/dev files."""

import argparse
import hashlib
import importlib
import inspect
import json
import math
import random
from pathlib import Path

from parsing.location._inference import encode, load_base, settings
from parsing.location._validation import read_rows

DIRECTORY = Path(__file__).resolve().parent
LORA_TARGETS = (r"(?:.*\.)?(?:language_model|model)\.layers\.\d+\."
                r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
                r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj)|"
                r"mlp\.(?:gate_proj|up_proj|down_proj))")


def weight(row, configuration):
    if not math.isfinite(configuration["weight"]) or configuration["weight"] < 1:
        raise ValueError("Training weight must be finite and at least one")
    if configuration["loss"] == "label_weighted":
        return configuration["weight"] if row["expected_output"] == "Multiple" else 1.0
    if configuration["loss"] == "category_weighted":
        from parsing.location.extractor.training_features import CATEGORIES, categorize
        categories = row.get("difficulty_categories")
        if categories is None:
            categories = categorize(row)
        if not isinstance(categories, list) or set(categories) - set(CATEGORIES):
            raise ValueError("Unknown extractor training category")
        return configuration["weight"] if set(categories) & set(configuration["focus_categories"]) else 1.0
    return 1.0


def person(row):
    key = row.get("qid") or row.get("person_group")
    if not isinstance(key, str) or not key:
        raise ValueError("Every training/reference row needs qid or person_group")
    return key


def check_partition(component, splits):
    """Reproduce person membership; a later label edit must not reshuffle it."""
    rows = splits["train"] + splits["dev"]
    if component == "extractor":
        people = sorted({person(row) for row in rows})
        random.Random(42).shuffle(people)
        expected_dev = set(people[:round(len(people)*0.2)])
    elif component == "filter":
        # Fixed stratum iteration order is part of the seeded procedure.
        strata = [("Single", "birthplace"), ("Single", "deathplace"),
                  ("Multiple", "birthplace"), ("Multiple", "deathplace"),
                  ("Single", "birth"), ("Multiple", "birth"),
                  ("Multiple", "death"), ("Single", "death")]
        groups = {key: [] for key in strata}
        for row in rows:
            groups[row.get("split_label", row["expected_output"]), row["source_relation"]].append(person(row))
        rng, expected_dev = random.Random(42), set()
        for people in groups.values():
            people.sort()
            rng.shuffle(people)
            count = round(len(people)*0.2)
            if len(people) >= 2:
                count = max(1, min(len(people)-1, count))
            expected_dev.update(people[:count])
    else:
        # Rewriter membership is assigned to sampled source people before
        # expansion into replacement levels, not by splitting flattened tasks.
        return
    if expected_dev != {person(row) for row in splits["dev"]}:
        raise ValueError(f"{component}: data differs from the documented person split")


def load_data(component):
    directory = DIRECTORY / component
    configuration = json.loads((directory / "training/config.json").read_text())
    splits = {split: read_rows(directory / f"training/{split}.jsonl") for split in ("train", "dev")}
    heldout = read_rows(directory / "human_validation/annotations.jsonl")
    groups = []
    all_ids = set()
    renderer = importlib.import_module(f"parsing.location.{component}.inference")
    for split, rows in {**splits, "heldout": heldout}.items():
        if not rows:
            raise ValueError(f"Empty {split} split")
        groups.append({person(row) for row in rows})
        for row in rows:
            if not row.get("item_id") or row["item_id"] in all_ids:
                raise ValueError("Duplicate/missing example identity across splits")
            all_ids.add(row["item_id"])
            if not isinstance(row.get("expected_output"), str) or not row["expected_output"].strip():
                raise ValueError("Every example needs a non-empty assistant target")
            if component == "filter":
                if row["expected_output"] not in {"Single", "Multiple"}:
                    raise ValueError("Filter targets must be Single or Multiple")
                location, reason = renderer.input_location(row)
                if reason:
                    raise ValueError("Filter training needs a successful event extraction")
                renderer.format_prompt(row, location)
            elif component == "extractor":
                renderer.format_prompt(row["question"], row["answer"])
            else:
                renderer.format_prompt(row)
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("Train, dev and held-out people must be disjoint")
    check_partition(component, splits)
    return configuration, splits


def training_text(component, row):
    renderer = importlib.import_module(f"parsing.location.{component}.inference")
    if component == "extractor":
        return renderer.format_prompt(row["question"], row["answer"])
    if component == "filter":
        location, reason = renderer.input_location(row)
        if reason:
            raise ValueError("Filter training needs a successful event extraction")
        return renderer.format_prompt(row, location)
    return renderer.format_prompt(row)


def encode_example(tokenizer, text, target, max_length):
    prefix = encode(tokenizer, text)
    completion = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.eos_token_id is None or not prefix or not completion:
        raise ValueError("Non-empty prefix/target and EOS are required")
    ids = prefix + completion + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError(f"{len(ids)} tokens exceed max_length={max_length}; never silently truncate")
    return {"input_ids": ids, "attention_mask": [1]*len(ids),
            "labels": [-100]*len(prefix) + completion + [tokenizer.eos_token_id]}


class Collator:
    def __init__(self, pad):
        self.pad = pad

    def __call__(self, rows):
        import torch
        width = max(len(row["input_ids"]) for row in rows)
        result = {key: torch.tensor([row[key] + [pad]*(width-len(row[key])) for row in rows])
                  for key, pad in (("input_ids", self.pad), ("attention_mask", 0), ("labels", -100))}
        if any("loss_weight" in row for row in rows):
            if not all("loss_weight" in row for row in rows):
                raise ValueError("Cannot mix weighted and unweighted examples in one batch")
            result["loss_weight"] = torch.tensor([row["loss_weight"] for row in rows], dtype=torch.float32)
        return result


def weighted_loss(logits, labels, weights, mean_weight):
    import torch
    import torch.nn.functional as F
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("Invalid fixed training-weight normalization")
    if weights.shape != (labels.shape[0],) or not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid per-example weights")
    losses = []
    for prediction, target in zip(logits[:, :-1], labels[:, 1:]):
        mask = target != -100
        count = int(mask.sum())
        if not count:
            raise ValueError("Example has no assistant target tokens")
        sums = [F.cross_entropy(p.float(), t, reduction="sum")
                for p, t in zip(prediction[mask].split(32), target[mask].split(32))]
        losses.append(torch.stack(sums).sum()/count)
    return (torch.stack(losses)*weights.to(logits.device)).mean()/mean_weight


def train(args):
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

    if args.output_dir.exists():
        raise ValueError("Training output already exists; choose a new directory")
    configuration, splits = load_data(args.component)
    for key in ("batch_size", "gradient_accumulation", "epochs", "eval_steps", "patience", "learning_rate"):
        value = getattr(args, key)
        if value is not None:
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{key} must be positive")
            configuration[key] = value
    set_seed(configuration["seed"])
    model_settings = settings(DIRECTORY / args.component)
    model, tokenizer = load_base(model_settings, device=args.device, dtype=args.dtype,
                                 attention_implementation="sdpa", local_files_only=args.local_files_only)
    encoded = {split: [encode_example(tokenizer, training_text(args.component, row), row["expected_output"], configuration["max_length"])
                       for row in rows] for split, rows in splits.items()}
    weights = [weight(row, configuration) for row in splits["train"]]
    mean_weight = sum(weights)/len(weights)
    weighted = configuration["loss"] != "token_ce"
    if weighted:
        for row, value in zip(encoded["train"], weights, strict=True):
            row["loss_weight"] = value
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM, r=configuration["lora_rank"],
        lora_alpha=2*configuration["lora_rank"], lora_dropout=0.05, bias="none",
        revision=model_settings["base_revision"], target_modules=LORA_TARGETS))
    for fragment in (".linear_attn.in_proj_qkv", ".linear_attn.out_proj"):
        if not any(fragment in name for name in model.targeted_module_names):
            raise RuntimeError(f"LoRA missed required hybrid-attention module: {fragment}")
    model.config.use_cache = False

    class WeightedTrainer(Trainer):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            inputs = dict(inputs)
            weights = inputs.pop("loss_weight", None)
            if not model.training:
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            if weights is None:
                raise ValueError("Missing training weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = weighted_loss(outputs.logits, labels, weights, mean_weight)
            return (loss, outputs) if return_outputs else loss

    kwargs = dict(output_dir=str(args.output_dir / "checkpoints"), num_train_epochs=configuration["epochs"],
        per_device_train_batch_size=configuration["batch_size"], per_device_eval_batch_size=configuration["batch_size"],
        gradient_accumulation_steps=configuration["gradient_accumulation"], learning_rate=configuration["learning_rate"],
        lr_scheduler_type="cosine", weight_decay=0.01, optim="adamw_torch", max_grad_norm=1.0,
        eval_strategy="steps", save_strategy="steps", eval_steps=configuration["eval_steps"], save_steps=configuration["eval_steps"],
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        prediction_loss_only=True, logging_steps=1 if args.component == "filter" else 5, report_to="none",
        bf16=args.dtype == "bfloat16", fp16=args.dtype == "float16", use_cpu=args.device == "cpu",
        seed=configuration["seed"], data_seed=configuration["seed"], gradient_checkpointing=configuration["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False}, label_names=["labels"])
    kwargs["warmup_ratio" if "warmup_ratio" in inspect.signature(TrainingArguments).parameters else "warmup_steps"] = 0.05
    if weighted:
        kwargs["remove_unused_columns"] = False
    args.output_dir.mkdir(parents=True)
    hashes = {split: hashlib.sha256((DIRECTORY / args.component / f"training/{split}.jsonl").read_bytes()).hexdigest() for split in splits}
    (args.output_dir / "run.json").write_text(json.dumps({"component": args.component, "configuration": configuration,
        "model": model_settings, "training_data_sha256": hashes, "mean_training_weight": mean_weight,
        "heldout_used_for_selection": False}, indent=2) + "\n")
    trainer = (WeightedTrainer if weighted else Trainer)(model=model, args=TrainingArguments(**kwargs),
        processing_class=tokenizer, train_dataset=Dataset.from_list(encoded["train"]), eval_dataset=Dataset.from_list(encoded["dev"]),
        data_collator=Collator(tokenizer.pad_token_id), callbacks=[EarlyStoppingCallback(early_stopping_patience=configuration["patience"])])
    trainer.train()
    final = args.output_dir / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    (final / "prompt.txt").write_bytes((DIRECTORY / args.component / "prompt.txt").read_bytes())
    selection = {"best_checkpoint": trainer.state.best_model_checkpoint, "best_eval_loss": trainer.state.best_metric,
                 "global_step": trainer.state.global_step, "epoch": trainer.state.epoch}
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    model.eval()
    if args.component == "filter":
        from parsing.location.filter.inference import score_rows
        from parsing.location.filter.calibration import select_threshold
        scores = score_rows(model, tokenizer, splits["dev"], batch_size=configuration["batch_size"])
        dev = [{**row, **score} for row, score in zip(splits["dev"], scores, strict=True)]
        threshold, metrics = select_threshold(dev, configuration["minimum_multiple_recall"])
        from parsing.location.filter.inference import classify
        dev = [{**row, **classify(row["label_logprob"], threshold)} for row in dev]
        threshold_record = {"multiple_log_odds_threshold": threshold, "selection_split": "dev", "metrics": metrics}
        (final / "threshold.json").write_text(json.dumps(threshold_record, indent=2) + "\n")
        (args.output_dir / "dev-scores.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dev))
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"Best-dev-loss adapter: {final}; held-out predictions were not generated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("extractor", "filter", "rewriter"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-data", action="store_true", help="Check labels/split disjointness and print counts without loading a model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    for key in ("batch-size", "gradient-accumulation", "eval-steps", "patience"):
        parser.add_argument("--"+key, type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    if args.check_data:
        configuration, splits = load_data(args.component)
        weights = [weight(row, configuration) for row in splits["train"]]
        print(json.dumps({"component": args.component, "splits": {key: {"items": len(rows), "people": len({person(row) for row in rows})} for key, rows in splits.items()},
                          "weighted_examples": sum(value != 1 for value in weights), "mean_training_weight": sum(weights)/len(weights)}, indent=2))
    else:
        if args.output_dir is None:
            parser.error("--output-dir is required for training")
        train(args)


if __name__ == "__main__":
    main()
