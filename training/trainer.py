"""Transformers trainer for case-balanced SFT and span ranking."""

from __future__ import annotations

import inspect

import numpy as np
import torch
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback

from .data import LEVEL_NAMES, TrainingObjective
from .loss import LossComponents, loss_components, objective_loss

MODALITY_NAMES = ("temporal", "location", "granola")
_GLOBAL_STAT_COUNT = 11
_CATEGORY_STAT_COUNT = 5


def _category_statistics(
    components: LossComponents,
    category_ids: torch.Tensor,
    category_count: int,
) -> torch.Tensor:
    # One segmented reduction, including empty categories, rather than a GPU
    # kernel loop over every source/relation/level for each evaluation batch.
    weights = components.group_metric_weights
    values = torch.stack(
        (
            components.group_losses * weights,
            components.group_correct * weights,
            components.group_margins * weights,
            weights,
            torch.ones_like(weights),
        ),
        dim=1,
    )
    totals = weights.new_zeros((category_count, _CATEGORY_STAT_COUNT))
    totals.index_add_(0, category_ids, values)
    return totals.flatten()


def evaluation_statistics(
    components: LossComponents, *, per_realization: bool = False
) -> torch.Tensor:
    """Create fixed-width sufficient statistics, never vocabulary logits."""
    if per_realization:
        count = components.sft_case_weights.numel()
        owners = components.group_realization_indices
        totals = components.sft_loss.new_zeros((count, _GLOBAL_STAT_COUNT))
        totals[:, 0] = components.sft_per_realization * components.sft_case_weights
        totals[:, 1] = components.sft_case_weights
        values = torch.stack(
            (
                components.group_losses * components.group_objective_weights,
                components.group_objective_weights,
                components.group_correct * components.group_objective_weights,
                components.group_margins * components.group_objective_weights,
                components.group_losses * components.group_metric_weights,
                components.group_metric_weights,
                components.group_correct * components.group_metric_weights,
                components.group_margins * components.group_metric_weights,
                torch.ones_like(components.group_metric_weights),
            ),
            dim=1,
        )
        totals[:, 2:].index_add_(0, owners, values)
        categories = []
        for ids, names in (
            (components.group_modality_ids, MODALITY_NAMES),
            (components.group_target_level_ids, LEVEL_NAMES),
        ):
            weights = components.group_metric_weights
            values = torch.stack(
                (
                    components.group_losses * weights,
                    components.group_correct * weights,
                    components.group_margins * weights,
                    weights,
                    torch.ones_like(weights),
                ),
                dim=1,
            )
            category = weights.new_zeros((count * len(names), _CATEGORY_STAT_COUNT))
            category.index_add_(0, owners * len(names) + ids, values)
            categories.append(category.reshape(count, -1))
        return torch.cat((totals, *categories), dim=1).detach().float()
    objective_correct = (
        components.group_correct * components.group_objective_weights
    ).sum()
    objective_margin = (
        components.group_margins * components.group_objective_weights
    ).sum()
    natural_correct = (components.group_correct * components.group_metric_weights).sum()
    natural_margin = (components.group_margins * components.group_metric_weights).sum()
    values = [
        components.sft_weighted_sum,
        components.sft_weight_sum,
        components.contrastive_weighted_sum,
        components.contrastive_weight_sum,
        objective_correct,
        objective_margin,
        components.natural_contrastive_weighted_sum,
        components.natural_contrastive_weight_sum,
        natural_correct,
        natural_margin,
        torch.tensor(
            components.group_losses.numel(),
            device=components.sft_loss.device,
            dtype=components.sft_loss.dtype,
        ),
    ]
    return (
        torch.cat(
            (
                torch.stack(values),
                _category_statistics(
                    components, components.group_modality_ids, len(MODALITY_NAMES)
                ),
                _category_statistics(
                    components, components.group_target_level_ids, len(LEVEL_NAMES)
                ),
            )
        )
        .detach()
        .float()
    )


def aggregate_evaluation_statistics(
    predictions,
    *,
    objective: TrainingObjective,
    contrastive_weight: float,
) -> dict[str, float]:
    statistics = np.asarray(predictions, dtype=np.float64).reshape(
        -1,
        _GLOBAL_STAT_COUNT
        + _CATEGORY_STAT_COUNT * (len(MODALITY_NAMES) + len(LEVEL_NAMES)),
    )
    totals = statistics.sum(axis=0)
    sft_mass = totals[1] if totals[1] > 0 else 1.0
    rank_mass = totals[3] if totals[3] > 0 else 1.0
    natural_mass = totals[7] if totals[7] > 0 else 1.0
    sft_loss = float(totals[0] / sft_mass)
    contrastive_loss = float(totals[2] / rank_mass)
    contrastive_accuracy = float(totals[4] / rank_mass)
    contrastive_margin = float(totals[5] / rank_mass)
    natural_contrastive_loss = float(totals[6] / natural_mass)
    natural_contrastive_accuracy = float(totals[8] / natural_mass)
    natural_contrastive_margin = float(totals[9] / natural_mass)
    if objective == "sft":
        combined = sft_loss
    elif objective == "contrastive":
        combined = contrastive_loss
    else:
        combined = sft_loss + contrastive_weight * contrastive_loss
    metrics = {
        "objective_loss": combined,
        "sft_loss": sft_loss,
        "contrastive_loss": contrastive_loss,
        "contrastive_accuracy": contrastive_accuracy,
        "contrastive_margin": contrastive_margin,
        "contrastive_weight_mass": float(totals[3]),
        "natural_contrastive_loss": natural_contrastive_loss,
        "natural_contrastive_accuracy": natural_contrastive_accuracy,
        "natural_contrastive_margin": natural_contrastive_margin,
        "natural_contrastive_weight_mass": float(totals[7]),
        "contrastive_groups": float(totals[10]),
    }

    offset = _GLOBAL_STAT_COUNT
    for prefix, names in (
        ("modality", MODALITY_NAMES),
        ("target_level", LEVEL_NAMES),
    ):
        macro_losses = []
        macro_accuracies = []
        macro_margins = []
        for name in names:
            loss_sum, correct_sum, margin_sum, weight_sum, count = totals[
                offset : offset + _CATEGORY_STAT_COUNT
            ]
            offset += _CATEGORY_STAT_COUNT
            if count == 0:
                continue
            metric_name = name.lower()
            denominator = weight_sum if weight_sum > 0 else 1.0
            metrics[f"{prefix}_{metric_name}_loss"] = float(loss_sum / denominator)
            metrics[f"{prefix}_{metric_name}_accuracy"] = float(
                correct_sum / denominator
            )
            metrics[f"{prefix}_{metric_name}_margin"] = float(margin_sum / denominator)
            metrics[f"{prefix}_{metric_name}_weight_mass"] = float(weight_sum)
            metrics[f"{prefix}_{metric_name}_groups"] = float(count)
            macro_losses.append(float(loss_sum / denominator))
            macro_accuracies.append(float(correct_sum / denominator))
            macro_margins.append(float(margin_sum / denominator))
        if macro_losses:
            metrics[f"macro_{prefix}_loss"] = float(np.mean(macro_losses))
            metrics[f"macro_{prefix}_accuracy"] = float(np.mean(macro_accuracies))
            metrics[f"macro_{prefix}_margin"] = float(np.mean(macro_margins))
    return metrics


class EpochDatasetCallback(TrainerCallback):
    """Expose the current epoch to deterministic negative sampling."""

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        dataset = getattr(train_dataloader, "dataset", None)
        set_epoch = getattr(dataset, "set_epoch", None)
        if set_epoch is not None:
            set_epoch(int(state.epoch or 0))
        return control


class PrimaryMetricEarlyStoppingCallback(EarlyStoppingCallback):
    """Ignore auxiliary evaluations that cannot select the main checkpoint."""

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric = args.metric_for_best_model
        key = metric if metric.startswith("eval_") else f"eval_{metric}"
        if key not in metrics:
            return control
        return super().on_evaluate(
            args,
            state,
            control,
            metrics,
            **kwargs,
        )


class GranularityTrainer(Trainer):
    """Train without autoregressive generation or online extraction."""

    def __init__(
        self,
        *args,
        objective: TrainingObjective = "sft_contrastive",
        contrastive_weight: float = 1.0,
        auxiliary_eval_dataset=None,
        **kwargs,
    ) -> None:
        self.objective = objective
        self.contrastive_weight = contrastive_weight
        self.auxiliary_eval_dataset = auxiliary_eval_dataset
        if kwargs.get("compute_metrics") is None:
            kwargs["compute_metrics"] = self._compute_proxy_metrics
        callbacks = list(kwargs.pop("callbacks", []))
        callbacks.append(EpochDatasetCallback())
        kwargs["callbacks"] = callbacks
        super().__init__(*args, **kwargs)
        # The custom objective is a realization mean, not Trainer's token sum.
        self.model_accepts_loss_kwargs = False
        base = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        self._supports_logit_positions = (
            "logits_to_keep" in inspect.signature(base.forward).parameters
        )


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        if (
            eval_dataset is None
            and metric_key_prefix == "eval"
            and self.auxiliary_eval_dataset is not None
        ):
            metrics.update(
                super().evaluate(
                    eval_dataset=self.auxiliary_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix="eval_granola",
                )
            )
        return metrics

    def _compute_proxy_metrics(self, prediction) -> dict[str, float]:
        return aggregate_evaluation_statistics(
            prediction.predictions,
            objective=self.objective,
            contrastive_weight=self.contrastive_weight,
        )

    def evaluation_loop(self, *args, metric_key_prefix="eval", **kwargs):
        output = super().evaluation_loop(
            *args, metric_key_prefix=metric_key_prefix, **kwargs
        )
        # Batch means are not a globally case-weighted objective. Use the
        # exact realization statistics, including the trimmed distributed tail.
        key = f"{metric_key_prefix}_objective_loss"
        if key in output.metrics:
            output.metrics[f"{metric_key_prefix}_loss"] = output.metrics[key]
        return output


    def _components(self, model, inputs):
        positions = (
            inputs.get("logits_to_keep") if self._supports_logit_positions else None
        )
        extra = {"logits_to_keep": positions} if positions is not None else {}
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            **extra,
        )
        return outputs, loss_components(
            outputs.logits, inputs, logit_positions=positions
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        outputs, components = self._components(model, inputs)
        loss = objective_loss(
            components,
            self.objective,
            self.contrastive_weight,
        )
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            _, components = self._components(model, inputs)
            loss = objective_loss(
                components,
                self.objective,
                self.contrastive_weight,
            )
        if prediction_loss_only:
            return loss.detach(), None, None
        statistics = evaluation_statistics(components, per_realization=True)
        placeholder = torch.zeros_like(statistics)
        return loss.detach(), statistics, placeholder

