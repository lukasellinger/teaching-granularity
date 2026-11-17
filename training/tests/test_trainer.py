import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from training.configuration import load_config, REPO
from training.data import weight_realizations, load_training_examples
from training.encoding import RealizationCollator, RealizationDataset
from training.trainer import GranularityTrainer, PrimaryMetricEarlyStoppingCallback
from training.tests.test_data import CharacterTokenizer, example


def test_sft_optimizer_step_and_vital_only_checkpoint_selection(tmp_path):
    torch.set_num_threads(2)
    config = load_config("lfm")
    vital = example("case:0", "Q1", "A was born in 1900.", rankable=True, prompt_template_id=0)
    granola = load_training_examples(REPO / config["granola_data_dir"] / "train.jsonl")[0]
    mixed = weight_realizations([vital, granola], "sft", granola_loss_fraction=.05, idk_sft_weight=.5)
    class Tokenizer(CharacterTokenizer):
        def save_pretrained(self, directory):
            return ()
    tokenizer = Tokenizer()
    def dataset(rows, negatives):
        return RealizationDataset(rows, tokenizer, seed=42, max_negatives_per_group=negatives, max_length=512)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=256, n_positions=512, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=0, eos_token_id=0, pad_token_id=0))
    original = model.transformer.wte.weight.detach().clone()
    trainer = GranularityTrainer(model=model, objective="sft",
        args=transformers.TrainingArguments(output_dir=str(tmp_path / "checkpoints"), max_steps=2,
            use_cpu=True, per_device_train_batch_size=2, per_device_eval_batch_size=1,
            gradient_accumulation_steps=1, eval_strategy="steps", eval_steps=1,
            save_strategy="steps", save_steps=1, load_best_model_at_end=True,
            metric_for_best_model="eval_objective_loss", greater_is_better=False,
            report_to="none", disable_tqdm=True, remove_unused_columns=False),
        train_dataset=dataset(mixed, 0), eval_dataset=dataset(weight_realizations([vital], "sft_contrastive"), None),
        auxiliary_eval_dataset=dataset(weight_realizations([granola], "sft_contrastive"), None),
        data_collator=RealizationCollator(tokenizer),
        callbacks=[PrimaryMetricEarlyStoppingCallback(early_stopping_patience=5)])
    trainer.train()
    assert trainer.state.global_step == 2
    assert not torch.equal(original, model.transformer.wte.weight)
    primary = [row["eval_objective_loss"] for row in trainer.state.log_history if "eval_objective_loss" in row]
    assert trainer.state.best_metric == pytest.approx(min(primary))
    assert any("eval_granola_sft_loss" in row for row in trainer.state.log_history)
    trainer.save_model(str(tmp_path / "final"))
    assert (tmp_path / "final/model.safetensors").is_file()


def test_auxiliary_metrics_do_not_advance_primary_early_stopping():
    from types import SimpleNamespace
    callback = PrimaryMetricEarlyStoppingCallback(early_stopping_patience=2)
    callback.early_stopping_patience_counter = 1
    control = transformers.TrainerControl()
    args = SimpleNamespace(metric_for_best_model="eval_objective_loss", greater_is_better=False)
    state = SimpleNamespace(best_metric=1.0)
    callback.on_evaluate(args, state, control, {"eval_granola_objective_loss": 5.0})
    assert callback.early_stopping_patience_counter == 1
    callback.on_evaluate(args, state, control, {"eval_objective_loss": 2.0})
    assert callback.early_stopping_patience_counter == 2
    assert control.should_training_stop
