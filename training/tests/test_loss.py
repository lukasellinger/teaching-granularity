from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from training.trainer import (  # noqa: E402
    MODALITY_NAMES,
    aggregate_evaluation_statistics,
    evaluation_statistics,
)
from training.data import LEVEL_NAMES  # noqa: E402
from training.loss import loss_components, objective_loss  # noqa: E402


def ranking_inputs() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 1, 4],
                [0, 1, 2],
                [0, 1, 3],
            ]
        ),
        "sft_mask": torch.tensor([[False, False, True]] * 5),
        "positive_sequence_indices": torch.tensor([0, 3]),
        "candidate_sequence_indices": torch.tensor([0, 1, 0, 2, 3, 4]),
        "candidate_score_masks": torch.tensor([[False, False, True]] * 6),
        "candidate_group_ids": torch.tensor([0, 0, 1, 1, 2, 2]),
        "candidate_is_target": torch.tensor([True, False, True, False, True, False]),
        "group_realization_indices": torch.tensor([0, 0, 1]),
        "group_modality_ids": torch.tensor([0, 1, 0]),
        "group_target_level_ids": torch.tensor([4, 10, 4]),
        "sft_case_weights": torch.tensor([0.5, 1.5]),
        "contrastive_case_weights": torch.tensor([0.5, 1.5]),
    }


def test_loss_averages_groups_inside_realization_before_case_weighting():
    inputs = ranking_inputs()
    logits = torch.zeros(5, 3, 5)
    logits[0, 1, 2] = 3.0
    logits[1, 1, 3] = -2.0
    logits[2, 1, 4] = 1.0
    logits[3, 1, 2] = -1.0
    logits[4, 1, 3] = 2.0

    components = loss_components(logits, inputs)

    expected = (
        0.5 * components.group_losses[:2].mean() + 1.5 * components.group_losses[2]
    ) / 2
    assert torch.allclose(components.contrastive_loss, expected)
    combined = objective_loss(components, "sft_contrastive", 0.25)
    assert torch.allclose(
        combined,
        components.sft_loss + 0.25 * components.contrastive_loss,
    )


def test_proxy_metrics_use_case_weights_and_report_breakdowns():
    inputs = ranking_inputs()
    logits = torch.zeros(5, 3, 5)
    logits[0, 1, 2] = 3.0
    logits[1, 1, 3] = -2.0
    logits[2, 1, 4] = 1.0
    logits[3, 1, 2] = -1.0
    logits[4, 1, 3] = 2.0
    components = loss_components(logits, inputs)
    statistics = evaluation_statistics(components).numpy()

    metrics = aggregate_evaluation_statistics(
        np.stack([statistics]),
        objective="sft_contrastive",
        contrastive_weight=0.5,
    )

    assert metrics["objective_loss"] == pytest.approx(
        metrics["sft_loss"] + 0.5 * metrics["contrastive_loss"]
    )
    assert metrics["contrastive_groups"] == 3
    assert metrics["modality_temporal_groups"] == 2
    assert metrics["modality_location_groups"] == 1
    assert metrics["target_level_year_groups"] == 2
    assert metrics["target_level_city_groups"] == 1


def test_per_realization_statistics_sum_to_the_same_case_weighted_metrics():
    inputs = ranking_inputs()
    components = loss_components(torch.randn(5, 3, 5), inputs)
    rows = evaluation_statistics(components, per_realization=True)
    assert rows.shape[0] == 2
    torch.testing.assert_close(rows.sum(dim=0), evaluation_statistics(components))
    # The last distributed batch may repeat row zero; trimming it must recover
    # exactly the original statistics, not lose a whole aggregated batch.
    gathered = torch.cat((rows, rows[:1]))[:2]
    torch.testing.assert_close(gathered.sum(dim=0), evaluation_statistics(components))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_logits_use_fp32_scoring_and_reductions(dtype):
    inputs = ranking_inputs()
    logits = torch.randn(5, 3, 5).to(dtype).requires_grad_(True)
    actual = loss_components(logits, inputs)
    expected = loss_components(logits.float(), inputs)
    assert actual.sft_loss.dtype is torch.float32
    assert actual.contrastive_loss.dtype is torch.float32
    torch.testing.assert_close(actual.sft_loss, expected.sft_loss)
    torch.testing.assert_close(actual.contrastive_loss, expected.contrastive_loss)
    grad = torch.autograd.grad(actual.sft_loss + actual.contrastive_loss, logits)[0]
    reference = torch.autograd.grad(
        expected.sft_loss + expected.contrastive_loss, logits
    )[0]
    torch.testing.assert_close(grad, reference)


def test_fractional_eval_mass_is_not_clamped_to_one():
    components = loss_components(torch.zeros(5, 3, 5), ranking_inputs())
    statistics = evaluation_statistics(components).numpy()
    baseline = aggregate_evaluation_statistics(
        statistics, objective="sft_contrastive", contrastive_weight=1
    )
    scaled = aggregate_evaluation_statistics(
        statistics * 0.01, objective="sft_contrastive", contrastive_weight=1
    )
    for name in (
        "sft_loss",
        "contrastive_loss",
        "natural_contrastive_loss",
        "objective_loss",
    ):
        assert scaled[name] == pytest.approx(baseline[name])


def test_statistic_width_is_fixed_when_there_are_no_groups():
    inputs = ranking_inputs()
    inputs.update(
        {
            "input_ids": inputs["input_ids"][:2],
            "sft_mask": inputs["sft_mask"][:2],
            "positive_sequence_indices": torch.tensor([0, 1]),
            "candidate_sequence_indices": torch.tensor([], dtype=torch.long),
            "candidate_score_masks": torch.zeros((0, 3), dtype=torch.bool),
            "candidate_group_ids": torch.tensor([], dtype=torch.long),
            "candidate_is_target": torch.tensor([], dtype=torch.bool),
            "group_realization_indices": torch.tensor([], dtype=torch.long),
            "group_modality_ids": torch.tensor([], dtype=torch.long),
            "group_target_level_ids": torch.tensor([], dtype=torch.long),
            "contrastive_case_weights": torch.tensor([0.0, 0.0]),
        }
    )
    logits = torch.zeros(2, 3, 5)

    statistics = evaluation_statistics(loss_components(logits, inputs))

    expected_width = 11 + 5 * (len(MODALITY_NAMES) + len(LEVEL_NAMES))
    assert statistics.shape == (expected_width,)


def test_equal_target_and_negative_scores_are_not_counted_as_correct():
    inputs = ranking_inputs()
    logits = torch.zeros(5, 3, 5)

    components = loss_components(logits, inputs)

    assert components.group_correct.tolist() == [0.0, 0.0, 0.0]


def test_pairwise_group_loss_has_log_two_baseline_with_multiple_negatives():
    inputs = {
        "input_ids": torch.tensor([[0, 1, 2], [0, 1, 3], [0, 1, 4]]),
        "sft_mask": torch.tensor([[False, False, True]] * 3),
        "positive_sequence_indices": torch.tensor([0]),
        "candidate_sequence_indices": torch.tensor([0, 1, 2]),
        "candidate_score_masks": torch.tensor([[False, False, True]] * 3),
        "candidate_group_ids": torch.tensor([0, 0, 0]),
        "candidate_is_target": torch.tensor([True, False, False]),
        "group_realization_indices": torch.tensor([0]),
        "group_modality_ids": torch.tensor([0]),
        "group_target_level_ids": torch.tensor([4]),
        "sft_case_weights": torch.tensor([1.0]),
        "contrastive_case_weights": torch.tensor([1.0]),
    }
    logits = torch.zeros(3, 3, 5)

    components = loss_components(logits, inputs)

    assert components.group_losses.item() == pytest.approx(math.log(2))


def test_group_balance_changes_only_the_contrastive_objective():
    inputs = ranking_inputs()
    logits = torch.zeros(5, 3, 5)
    logits[0, 1, 2] = 3.0
    logits[1, 1, 3] = -2.0
    logits[2, 1, 4] = 1.0
    logits[3, 1, 2] = -1.0
    logits[4, 1, 3] = 2.0

    natural = loss_components(logits, inputs)
    inputs["group_balance_weights"] = torch.tensor([1.0, 2.0, 0.5])
    balanced = loss_components(logits, inputs)

    expected = (
        0.25 * balanced.group_losses[0]
        + 0.5 * balanced.group_losses[1]
        + 0.75 * balanced.group_losses[2]
    ) / 2
    assert torch.allclose(balanced.contrastive_loss, expected)
    assert torch.allclose(balanced.sft_loss, natural.sft_loss)
    assert torch.allclose(
        balanced.natural_contrastive_weighted_sum,
        natural.natural_contrastive_weighted_sum,
    )
    assert not torch.allclose(
        balanced.contrastive_weighted_sum,
        balanced.natural_contrastive_weighted_sum,
    )

    metrics = aggregate_evaluation_statistics(
        np.stack([evaluation_statistics(balanced).numpy()]),
        objective="sft_contrastive",
        contrastive_weight=1.0,
    )
    assert metrics["contrastive_loss"] != pytest.approx(
        metrics["natural_contrastive_loss"]
    )
    assert "macro_modality_accuracy" in metrics
    assert "macro_target_level_accuracy" in metrics


def dense_reference_losses(logits, inputs):
    """Original dense/loop formulation, kept independent of the fast path."""
    nll = torch.nn.functional.cross_entropy(
        logits[:, :-1].transpose(1, 2), inputs["input_ids"][:, 1:], reduction="none"
    )
    positives = inputs["positive_sequence_indices"]
    mask = inputs["sft_mask"][positives, 1:]
    sft = (
        ((nll[positives] * mask).sum(1) / mask.sum(1)) * inputs["sft_case_weights"]
    ).mean()
    mask = inputs["candidate_score_masks"][:, 1:]
    scores = -(nll[inputs["candidate_sequence_indices"]] * mask).sum(1) / mask.sum(1)
    losses, margins = [], []
    for group in range(inputs["group_realization_indices"].numel()):
        members = inputs["candidate_group_ids"] == group
        target = scores[members][inputs["candidate_is_target"][members]][0]
        negatives = scores[members][~inputs["candidate_is_target"][members]]
        losses.append(torch.nn.functional.softplus(negatives - target).mean())
        margins.append(target - negatives.max())
    losses, margins = torch.stack(losses), torch.stack(margins)
    owners = inputs["group_realization_indices"]
    counts = torch.bincount(owners, minlength=positives.numel())
    weights = inputs["contrastive_case_weights"][owners] / counts[owners]
    weights = weights * inputs.get("group_balance_weights", torch.ones_like(weights))
    return sft, (losses * weights).sum() / positives.numel(), losses, margins


@pytest.mark.parametrize("objective", ["sft", "contrastive", "sft_contrastive"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vectorized_and_selective_losses_match_dense_values_and_gradients(
    objective, dtype
):
    inputs = ranking_inputs()
    inputs["input_ids"] = inputs["input_ids"].repeat(1, 3)
    inputs["sft_mask"] = torch.zeros((5, 9), dtype=torch.bool)
    inputs["sft_mask"][:, [2, 3, 7]] = True
    inputs["candidate_sequence_indices"] = torch.tensor([0, 1, 0, 2, 3, 4, 2])
    inputs["candidate_group_ids"] = torch.tensor([0, 0, 1, 1, 2, 2, 0])
    inputs["candidate_is_target"] = torch.tensor(
        [True, False, True, False, True, False, False]
    )
    inputs["candidate_score_masks"] = torch.zeros((7, 9), dtype=torch.bool)
    for row in range(7):
        inputs["candidate_score_masks"][row, 3 : 5 + row % 3] = True
    # Candidate ordering and target position are not assumed by the fast path.
    permutation = torch.tensor([1, 2, 6, 0, 4, 5, 3])
    for key in (
        "candidate_sequence_indices",
        "candidate_group_ids",
        "candidate_is_target",
        "candidate_score_masks",
    ):
        inputs[key] = inputs[key][permutation]
    inputs["group_balance_weights"] = torch.tensor([1.0, 2.0, 0.5])
    torch.manual_seed(42)
    logits = torch.randn((5, 9, 5), dtype=dtype, requires_grad=True)
    ref_sft, ref_rank, ref_groups, ref_margins = dense_reference_losses(logits, inputs)
    reference = (
        ref_sft
        if objective == "sft"
        else ref_rank
        if objective == "contrastive"
        else ref_sft + 0.25 * ref_rank
    )
    expected_gradient = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    needed = torch.zeros((5, 8), dtype=torch.bool)
    positives = inputs["positive_sequence_indices"]
    needed[positives] = inputs["sft_mask"][positives, 1:]
    for seq, mask in zip(
        inputs["candidate_sequence_indices"], inputs["candidate_score_masks"]
    ):
        needed[seq] |= mask[1:]
    positions = needed.any(0).nonzero(as_tuple=True)[0]
    for selected in (None, positions):
        projected = logits if selected is None else logits[:, selected]
        components = loss_components(projected, inputs, logit_positions=selected)
        torch.testing.assert_close(components.sft_loss, ref_sft)
        torch.testing.assert_close(components.contrastive_loss, ref_rank)
        torch.testing.assert_close(components.group_losses, ref_groups)
        torch.testing.assert_close(components.group_margins, ref_margins)
        actual = torch.autograd.grad(
            objective_loss(components, objective, 0.25), logits, retain_graph=True
        )[0]
        torch.testing.assert_close(actual, expected_gradient)


def test_selective_sft_without_contrastive_candidates():
    inputs = ranking_inputs()
    for name in (
        "candidate_sequence_indices",
        "candidate_group_ids",
        "candidate_is_target",
        "group_realization_indices",
        "group_modality_ids",
        "group_target_level_ids",
    ):
        inputs[name] = inputs[name][:0]
    inputs["candidate_score_masks"] = inputs["candidate_score_masks"][:0]
    logits = torch.randn((5, 3, 5), requires_grad=True)
    dense = loss_components(logits, inputs)
    positions = torch.tensor([1])
    sparse = loss_components(logits[:, positions], inputs, logit_positions=positions)
    torch.testing.assert_close(dense.sft_loss, sparse.sft_loss)
    assert sparse.contrastive_loss == 0


def test_invalid_group_metadata_is_still_rejected():
    inputs = ranking_inputs()
    inputs["candidate_is_target"][1] = True
    with pytest.raises(ValueError, match="one target"):
        loss_components(torch.zeros((5, 3, 5)), inputs)


def test_validated_loss_has_no_host_scalar_reads_or_dynamic_group_loops():
    from training.loss import validate_loss_inputs

    inputs = ranking_inputs()
    validate_loss_inputs(inputs)
    inputs["loss_inputs_validated"] = True
    logits = torch.randn((5, 3, 5), requires_grad=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        objective_loss(loss_components(logits, inputs), "sft_contrastive", 1).backward()
    operations = {event.key for event in profile.key_averages()}
    assert not operations & {
        "aten::item",
        "aten::_local_scalar_dense",
        "aten::nonzero",
        "aten::bincount",
    }
