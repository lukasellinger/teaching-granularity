"""Hierarchical full-answer and span-ranking losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import TrainingObjective


@dataclass(frozen=True)
class LossComponents:
    """Differentiable losses plus sufficient statistics for proxy evaluation."""

    sft_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    sft_weighted_sum: torch.Tensor
    sft_weight_sum: torch.Tensor
    contrastive_weighted_sum: torch.Tensor
    contrastive_weight_sum: torch.Tensor
    natural_contrastive_weighted_sum: torch.Tensor
    natural_contrastive_weight_sum: torch.Tensor
    group_losses: torch.Tensor
    group_correct: torch.Tensor
    group_margins: torch.Tensor
    group_metric_weights: torch.Tensor
    group_objective_weights: torch.Tensor
    group_modality_ids: torch.Tensor
    group_target_level_ids: torch.Tensor
    sft_per_realization: torch.Tensor
    sft_case_weights: torch.Tensor
    group_realization_indices: torch.Tensor


def validate_loss_inputs(inputs: dict[str, torch.Tensor]) -> None:
    """Validate metadata on CPU in the collator, not once per GPU group."""
    positives = inputs["positive_sequence_indices"]
    batch_size = positives.numel()
    if batch_size == 0 or torch.any(inputs["sft_mask"][positives, 1:].sum(dim=1) == 0):
        raise ValueError("every realization must supervise an assistant token")
    sft_weights = inputs["sft_case_weights"]
    if sft_weights.shape != positives.shape:
        raise ValueError("every realization must have one SFT case weight")
    if torch.any(~torch.isfinite(sft_weights)) or torch.any(sft_weights <= 0):
        raise ValueError("SFT case weights must be finite and positive")
    if inputs["candidate_sequence_indices"].numel() == 0:
        return
    if torch.any(inputs["candidate_score_masks"][:, 1:].sum(dim=1) == 0):
        raise ValueError("every contrastive candidate must have a score token")
    group_ids = inputs["candidate_group_ids"]
    owners = inputs["group_realization_indices"]
    count = owners.numel()
    if not torch.equal(
        torch.unique(group_ids, sorted=True),
        torch.arange(count, device=group_ids.device),
    ):
        raise ValueError("candidate group ids must be contiguous")
    targets = torch.zeros(
        count, dtype=torch.long, device=group_ids.device
    ).scatter_add_(0, group_ids, inputs["candidate_is_target"].long())
    members = torch.zeros_like(targets).scatter_add_(
        0, group_ids, torch.ones_like(group_ids)
    )
    if torch.any(targets != 1) or torch.any(members < 2):
        raise ValueError(
            "every contrastive group needs one target and at least one negative"
        )
    balance = inputs.get("group_balance_weights")
    if balance is not None:
        if balance.shape != owners.shape:
            raise ValueError("every contrastive group must have one balance weight")
        if torch.any(~torch.isfinite(balance)) or torch.any(balance <= 0):
            raise ValueError(
                "contrastive group balance weights must be finite and positive"
            )
    rank_weights = inputs["contrastive_case_weights"]
    if rank_weights.shape != positives.shape or torch.any(
        ~torch.isfinite(rank_weights)
    ):
        raise ValueError(
            "every realization must have one finite contrastive case weight"
        )
    has_groups = (
        torch.zeros(batch_size, dtype=torch.long, device=owners.device).scatter_add_(
            0, owners, torch.ones_like(owners)
        )
        > 0
    )
    if torch.any(rank_weights[~has_groups] != 0):
        raise ValueError("SFT-only realizations must have zero contrastive weight")
    if torch.any(rank_weights[has_groups] <= 0):
        raise ValueError("rankable realizations need a positive case weight")


def loss_components(
    logits: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    *,
    logit_positions: torch.Tensor | None = None,
) -> LossComponents:
    """Reduce tokens -> groups -> realizations, then apply case weights."""
    if not inputs.get("loss_inputs_validated", False):
        validate_loss_inputs(inputs)
    shifted_logits = logits[:, :-1, :] if logit_positions is None else logits
    shifted_targets = inputs["input_ids"][:, 1:]
    sft_masks = inputs["sft_mask"][:, 1:]
    score_masks = inputs["candidate_score_masks"][:, 1:]
    if logit_positions is not None:
        shifted_targets = shifted_targets[:, logit_positions]
        sft_masks = sft_masks[:, logit_positions]
        score_masks = score_masks[:, logit_positions]
    # Keep log-softmax and case/group reductions stable for BF16/FP16 outputs.
    if shifted_logits.dtype in (torch.float16, torch.bfloat16):
        shifted_logits = shifted_logits.float()
    token_nll = F.cross_entropy(
        shifted_logits.reshape(-1, logits.shape[-1]),
        shifted_targets.reshape(-1),
        reduction="none",
    ).reshape(shifted_targets.shape)
    dtype = token_nll.dtype
    device = token_nll.device

    positive_indices = inputs["positive_sequence_indices"]
    positive_nll = token_nll[positive_indices]
    positive_mask = sft_masks[positive_indices]
    sft_token_counts = positive_mask.sum(dim=1)
    sft_per_realization = (positive_nll * positive_mask).sum(dim=1) / sft_token_counts
    sft_weights = inputs["sft_case_weights"].to(device=device, dtype=dtype)
    sft_weighted_sum = (sft_per_realization * sft_weights).sum()
    sft_weight_sum = sft_weights.sum()
    batch_size = sft_per_realization.numel()
    sft_loss = sft_weighted_sum / batch_size

    candidate_indices = inputs["candidate_sequence_indices"]
    if candidate_indices.numel() == 0:
        empty = torch.empty(0, device=device, dtype=dtype)
        zero = token_nll.sum() * 0.0
        return LossComponents(
            sft_loss=sft_loss,
            contrastive_loss=zero,
            sft_weighted_sum=sft_weighted_sum,
            sft_weight_sum=sft_weight_sum,
            contrastive_weighted_sum=zero,
            contrastive_weight_sum=zero.detach(),
            natural_contrastive_weighted_sum=zero,
            natural_contrastive_weight_sum=zero.detach(),
            group_losses=empty,
            group_correct=empty,
            group_margins=empty,
            group_metric_weights=empty,
            group_objective_weights=empty,
            group_modality_ids=inputs["group_modality_ids"],
            group_target_level_ids=inputs["group_target_level_ids"],
            sft_per_realization=sft_per_realization,
            sft_case_weights=sft_weights,
            group_realization_indices=inputs["group_realization_indices"],
        )

    candidate_nll = token_nll[candidate_indices]
    candidate_masks = score_masks
    candidate_token_counts = candidate_masks.sum(dim=1)
    candidate_scores = (
        -(candidate_nll * candidate_masks).sum(dim=1) / candidate_token_counts
    )

    group_ids = inputs["candidate_group_ids"]
    target_mask = inputs["candidate_is_target"]
    group_realization_indices = inputs["group_realization_indices"]
    group_count = group_realization_indices.numel()
    target_scores = candidate_scores.new_zeros(group_count).scatter_add_(
        0, group_ids, candidate_scores.masked_fill(~target_mask, 0)
    )
    negatives = ~target_mask
    negative_counts = candidate_scores.new_zeros(group_count).scatter_add_(
        0, group_ids, negatives.to(dtype)
    )
    pair_losses = F.softplus(candidate_scores - target_scores[group_ids]).masked_fill(
        target_mask, 0
    )
    group_losses = (
        candidate_scores.new_zeros(group_count).scatter_add_(0, group_ids, pair_losses)
        / negative_counts
    )
    best_negatives = candidate_scores.new_full(
        (group_count,), -torch.inf
    ).scatter_reduce_(
        0,
        group_ids,
        candidate_scores.detach().masked_fill(target_mask, -torch.inf),
        reduce="amax",
        include_self=True,
    )
    # Metrics need no gradient graph; ties still count as incorrect.
    group_correct = target_scores.detach().gt(best_negatives).to(dtype)
    group_margins = target_scores.detach() - best_negatives
    group_balance_weights = inputs.get("group_balance_weights")
    if group_balance_weights is None:
        group_balance_weights = torch.ones_like(group_losses)
    else:
        group_balance_weights = group_balance_weights.to(device=device, dtype=dtype)
    realization_group_counts = torch.zeros(
        batch_size, dtype=torch.long, device=device
    ).scatter_add_(
        0, group_realization_indices, torch.ones_like(group_realization_indices)
    )
    contrastive_weights = inputs["contrastive_case_weights"].to(
        device=device, dtype=dtype
    )

    group_metric_weights = contrastive_weights[group_realization_indices] / (
        realization_group_counts[group_realization_indices].to(dtype)
    )
    group_objective_weights = group_metric_weights * group_balance_weights
    contrastive_weighted_sum = (group_losses * group_objective_weights).sum()
    contrastive_weight_sum = group_objective_weights.sum()
    natural_contrastive_weighted_sum = (group_losses * group_metric_weights).sum()
    natural_contrastive_weight_sum = group_metric_weights.sum()
    contrastive_loss = contrastive_weighted_sum / batch_size
    return LossComponents(
        sft_loss=sft_loss,
        contrastive_loss=contrastive_loss,
        sft_weighted_sum=sft_weighted_sum,
        sft_weight_sum=sft_weight_sum,
        contrastive_weighted_sum=contrastive_weighted_sum,
        contrastive_weight_sum=contrastive_weight_sum,
        natural_contrastive_weighted_sum=natural_contrastive_weighted_sum,
        natural_contrastive_weight_sum=natural_contrastive_weight_sum,
        group_losses=group_losses,
        group_correct=group_correct,
        group_margins=group_margins,
        group_metric_weights=group_metric_weights,
        group_objective_weights=group_objective_weights,
        group_modality_ids=inputs["group_modality_ids"],
        group_target_level_ids=inputs["group_target_level_ids"],
        sft_per_realization=sft_per_realization,
        sft_case_weights=sft_weights,
        group_realization_indices=group_realization_indices,
    )


def objective_loss(
    components: LossComponents,
    objective: TrainingObjective,
    contrastive_weight: float,
) -> torch.Tensor:
    if contrastive_weight <= 0:
        raise ValueError("contrastive_weight must be positive")
    if objective == "sft":
        return components.sft_loss
    if objective == "contrastive":
        return components.contrastive_loss
    if objective == "sft_contrastive":
        return components.sft_loss + contrastive_weight * components.contrastive_loss
    raise ValueError(f"unsupported training objective: {objective!r}")

