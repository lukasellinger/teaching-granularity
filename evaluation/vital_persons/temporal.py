"""Temporal correctness, granularity metrics and person-cluster intervals."""
from __future__ import annotations
import math
from shared.bootstrap import paired_qid_bootstrap as _paired_qid_bootstrap
import re
from collections import Counter
from typing import Literal, Sequence
import numpy as np
from parsing.temporal import parse_temporal_response, is_idk_response, TemporalValue, TemporalLevel
from parsing.temporal.response import ParsedTemporalResponse
from .answer_status.integration import generation_abstains
from parsing.temporal.values import TEMPORAL_PRECISION
from .cases import (TemporalEvaluationCase, temporal_value_dict, TemporalTaxonomy,
                    temporal_level_for_taxonomy, temporal_levels_for_taxonomy)


_WIDTH_YEARS = {
    TemporalLevel.CENTURY: 100.0,
    TemporalLevel.EARLYMIDCENTURY: 100.0 / 3,
    TemporalLevel.DECADE: 10.0,
    TemporalLevel.EARLYMIDDECADE: 10.0 / 3,
    TemporalLevel.YEAR: 1.0,
    TemporalLevel.SEASON: 1.0 / 4,
    TemporalLevel.EARLYMIDYEAR: 1.0 / 3,
    TemporalLevel.MONTH: 1.0 / 12,
    TemporalLevel.DAY: 1.0 / 365,
}


_PRIOR_WIDTH_YEARS = 1_000.0


_ORDINAL_PRECISIONS = tuple(sorted(set(TEMPORAL_PRECISION.values())))


_PRECISION_RANK = {
    precision: rank for rank, precision in enumerate(_ORDINAL_PRECISIONS)
}


def correct_at_prediction_level(
    prediction: TemporalValue | None,
    gold: TemporalValue,
) -> bool:
    """Reject a prediction that is finer than the available gold evidence."""
    return bool(
        prediction is not None
        and prediction.precision <= gold.precision
        and prediction.consistent_with(gold)
    )


def temporal_informativeness(level: TemporalLevel) -> float:
    return math.log(_PRIOR_WIDTH_YEARS / _WIDTH_YEARS[level]) / math.log(
        _PRIOR_WIDTH_YEARS / _WIDTH_YEARS[TemporalLevel.DAY]
    )


def _score_one(
    case: TemporalEvaluationCase,
    generation: dict[str, object],
) -> dict[str, object]:
    answer = generation.get("answer")
    if not isinstance(answer, str):
        raise ValueError(f"generation for {case.case_id} has no string answer")
    parsed = (ParsedTemporalResponse('idk', None, ()) if generation_abstains(generation)
              else parse_temporal_response(answer, case.relation))
    prediction = parsed.value
    prediction_level = (
        temporal_level_for_taxonomy(
            prediction.level,
            case.temporal_taxonomy,
        )
        if prediction is not None
        else None
    )
    truth_correct = (
        correct_at_prediction_level(prediction, case.gold)
        if case.gold is not None
        else None
    )
    exact_gold = (
        bool(
            truth_correct
            and prediction is not None
            and prediction.level is case.gold.level
        )
        if case.gold is not None
        else None
    )
    target_available = case.target_status != "unavailable"
    if case.target_status == "idk":
        target_decision_correct: bool | None = parsed.status == "idk"
        target_match: bool | None = parsed.status == "idk"
    elif case.target_status == "known":
        target_decision_correct = parsed.status == "parsed"
        target_match = bool(
            prediction_level is not None
            and prediction_level is case.target_level
        )
    else:
        target_decision_correct = None
        target_match = None
    if not target_available:
        joint = None
    elif case.target_status == "idk":
        joint = bool(target_match)
    elif case.gold is None:
        joint = None
    else:
        joint = bool(target_match and truth_correct)
    error = (
        TEMPORAL_PRECISION[prediction_level]
        - TEMPORAL_PRECISION[case.target_level]
        if (
            prediction_level is not None
            and case.target_status == "known"
            and case.target_level is not None
        )
        else None
    )
    utility = (
        None
        if case.gold is None
        else (
            temporal_informativeness(prediction.level)
            * (1.0 if truth_correct else -1.0)
            if prediction is not None
            else 0.0
        )
    )
    utility_uniform_wrong_penalty = (
        None
        if case.gold is None
        else (
            0.0
            if parsed.status == "idk"
            else (
                temporal_informativeness(prediction.level)
                if truth_correct and prediction is not None
                else -1.0
            )
        )
    )
    utility_accuracy = (
        None if case.gold is None else float(bool(truth_correct))
    )
    utility_positive_informativeness = (
        None
        if case.gold is None
        else (
            temporal_informativeness(prediction.level)
            if truth_correct and prediction is not None
            else 0.0
        )
    )
    prediction_category = (
        prediction_level.value
        if prediction_level is not None
        else parsed.status.upper()
    )
    target_category = (
        case.target_level.value
        if case.target_level is not None
        else case.target_status.upper()
    )
    return {
        "case_id": case.case_id,
        "source_model": case.source_model,
        "split": case.split.value,
        "modality": "temporal",
        "temporal_taxonomy": case.temporal_taxonomy.value,
        "qid": case.qid,
        "name": case.name,
        "relation": case.relation.value,
        "prompt": case.prompt,
        "generation_prompt": generation.get("user_prompt", case.prompt),
        "system_prompt": generation.get("system_prompt"),
        "gold": (
            temporal_value_dict(case.gold) if case.gold is not None else None
        ),
        "target_status": case.target_status,
        "target_level": (
            case.target_level.value if case.target_level is not None else None
        ),
        "answer": answer,
        "answer_words": len(re.findall(r"\b\w+\b", answer)),
        "generated_tokens": generation.get("generated_tokens"),
        "finish_reason": generation.get("finish_reason"),
        "parse_status": parsed.status,
        "matched_values": [
            temporal_value_dict(value) for value in parsed.matched_values
        ],
        "prediction": (
            temporal_value_dict(prediction) if prediction is not None else None
        ),
        "prediction_level_for_target_metrics": (
            prediction_level.value if prediction_level is not None else None
        ),
        "prediction_category": prediction_category,
        "target_category": target_category,
        "truth_correct": truth_correct,
        "exact_gold": exact_gold,
        "gold_overprecise": (
            bool(
                prediction is not None
                and prediction.precision > case.gold.precision
            )
            if case.gold is not None
            else None
        ),
        "target_decision_correct": target_decision_correct,
        "target_match": target_match,
        "joint_target_truth": joint,
        "granularity_error": error,
        "utility": utility,
        "utility_uniform_wrong_penalty": utility_uniform_wrong_penalty,
        "utility_accuracy": utility_accuracy,
        "utility_positive_informativeness": utility_positive_informativeness,
    }


def score_temporal_generations(
    cases: Sequence[TemporalEvaluationCase],
    generations: Sequence[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_case: dict[str, dict[str, object]] = {}
    for generation in generations:
        case_id = generation.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("every generation needs a string case_id")
        if case_id in by_case:
            raise ValueError(f"duplicate generation for {case_id}")
        by_case[case_id] = generation
    expected = {case.case_id for case in cases}
    missing = expected - set(by_case)
    extra = set(by_case) - expected
    if missing or extra:
        raise ValueError(
            "generations do not match evaluation cases: "
            f"{len(missing)} missing and {len(extra)} extra"
        )
    outcomes = [_score_one(case, by_case[case.case_id]) for case in cases]
    return aggregate_temporal_outcomes(outcomes), outcomes


def _rate(values: Sequence[object]) -> float | None:
    return float(np.mean(values)) if values else None


def quadratic_weighted_kappa(
    targets: Sequence[TemporalLevel],
    predictions: Sequence[TemporalLevel],
    *,
    taxonomy: TemporalTaxonomy = TemporalTaxonomy.FINE,
) -> float | None:
    """Ordinal agreement beyond chance, weighted by squared rank distance."""
    if len(targets) != len(predictions):
        raise ValueError("kappa targets and predictions must have equal length")
    if not targets:
        return None
    if taxonomy is TemporalTaxonomy.FINE:
        precision_ranks = _PRECISION_RANK
        rank_count = len(_ORDINAL_PRECISIONS)
    else:
        precisions = sorted(
            {
                TEMPORAL_PRECISION[level]
                for level in temporal_levels_for_taxonomy(taxonomy)
            }
        )
        precision_ranks = {
            precision: rank for rank, precision in enumerate(precisions)
        }
        rank_count = len(precisions)
    target_ranks = np.asarray([
        precision_ranks[TEMPORAL_PRECISION[level]] for level in targets
    ])
    prediction_ranks = np.asarray([
        precision_ranks[TEMPORAL_PRECISION[level]] for level in predictions
    ])
    scale = max(rank_count - 1, 1)
    observed = float(np.mean(((target_ranks - prediction_ranks) / scale) ** 2))
    target_frequency = np.bincount(
        target_ranks,
        minlength=rank_count,
    ) / len(targets)
    prediction_frequency = np.bincount(
        prediction_ranks,
        minlength=rank_count,
    ) / len(predictions)
    ranks = np.arange(rank_count)
    weights = ((ranks[:, None] - ranks[None, :]) / scale) ** 2
    expected = float(
        np.sum(weights * target_frequency[:, None] * prediction_frequency[None, :])
    )
    if math.isclose(expected, 0.0, abs_tol=1e-15):
        return None
    return 1.0 - observed / expected


def extended_quadratic_weighted_kappa(
    targets: Sequence[TemporalLevel | Literal["IDK"]],
    predictions: Sequence[TemporalLevel | Literal["IDK"]],
    *,
    taxonomy: TemporalTaxonomy = TemporalTaxonomy.FINE,
) -> float | None:
    """Ordinal agreement with IDK below the coarsest concrete level."""
    if len(targets) != len(predictions):
        raise ValueError("kappa targets and predictions must have equal length")
    if not targets:
        return None
    if taxonomy is TemporalTaxonomy.FINE:
        precision_ranks = _PRECISION_RANK
        concrete_rank_count = len(_ORDINAL_PRECISIONS)
    else:
        precisions = sorted(
            {
                TEMPORAL_PRECISION[level]
                for level in temporal_levels_for_taxonomy(taxonomy)
            }
        )
        precision_ranks = {
            precision: rank for rank, precision in enumerate(precisions)
        }
        concrete_rank_count = len(precisions)

    def rank(category: TemporalLevel | Literal["IDK"]) -> int:
        if category == "IDK":
            return 0
        return precision_ranks[TEMPORAL_PRECISION[category]] + 1

    target_ranks = np.asarray([rank(category) for category in targets])
    prediction_ranks = np.asarray([rank(category) for category in predictions])
    rank_count = concrete_rank_count + 1
    scale = max(rank_count - 1, 1)
    observed = float(np.mean(((target_ranks - prediction_ranks) / scale) ** 2))
    target_frequency = np.bincount(
        target_ranks,
        minlength=rank_count,
    ) / len(targets)
    prediction_frequency = np.bincount(
        prediction_ranks,
        minlength=rank_count,
    ) / len(predictions)
    ranks = np.arange(rank_count)
    weights = ((ranks[:, None] - ranks[None, :]) / scale) ** 2
    expected = float(
        np.sum(weights * target_frequency[:, None] * prediction_frequency[None, :])
    )
    if math.isclose(expected, 0.0, abs_tol=1e-15):
        return None
    return 1.0 - observed / expected


def _core_metrics(outcomes: Sequence[dict[str, object]]) -> dict[str, object]:
    taxonomy_values = {
        str(row.get("temporal_taxonomy", TemporalTaxonomy.FINE.value))
        for row in outcomes
    }
    if len(taxonomy_values) > 1:
        raise ValueError("temporal outcomes mix taxonomies")
    taxonomy = TemporalTaxonomy(
        next(iter(taxonomy_values), TemporalTaxonomy.FINE.value)
    )
    parsed = [row for row in outcomes if row["parse_status"] == "parsed"]
    target_available = [
        row for row in outcomes if row["target_status"] != "unavailable"
    ]
    known_target = [
        row for row in outcomes if row["target_status"] == "known"
    ]
    idk_target = [row for row in outcomes if row["target_status"] == "idk"]
    gold_available = [row for row in outcomes if row["gold"] is not None]
    parsed_gold = [
        row for row in gold_available if row["parse_status"] == "parsed"
    ]
    joint_available = [
        row for row in target_available if row["joint_target_truth"] is not None
    ]
    parsed_known_target = [
        row
        for row in known_target
        if row["parse_status"] == "parsed"
    ]
    errors = [
        float(row["granularity_error"])
        for row in parsed_known_target
        if row["granularity_error"] is not None
    ]
    generated_token_counts = [
        int(row["generated_tokens"])
        for row in outcomes
        if isinstance(row["generated_tokens"], int)
        and not isinstance(row["generated_tokens"], bool)
    ]
    predicted_idk = [
        row for row in target_available if row["parse_status"] == "idk"
    ]
    true_positive_idk = sum(
        row["target_status"] == "idk" for row in predicted_idk
    )
    kappa_targets = [
        TemporalLevel(str(row["target_level"]))
        for row in parsed_known_target
    ]
    kappa_predictions = [
        TemporalLevel(str(row["prediction_category"]))
        for row in parsed_known_target
    ]
    extended_kappa_rows = [
        row
        for row in target_available
        if row["parse_status"] in {"parsed", "idk"}
    ]
    extended_kappa_targets = [
        "IDK"
        if row["target_status"] == "idk"
        else TemporalLevel(str(row["target_level"]))
        for row in extended_kappa_rows
    ]
    extended_kappa_predictions = [
        "IDK"
        if row["parse_status"] == "idk"
        else TemporalLevel(str(row["prediction_category"]))
        for row in extended_kappa_rows
    ]
    return {
        "cases": len(outcomes),
        "parsed_cases": len(parsed),
        "target_available_cases": len(target_available),
        "target_coverage": (
            len(target_available) / len(outcomes) if outcomes else None
        ),
        "known_target_cases": len(known_target),
        "parsed_known_target_cases": len(parsed_known_target),
        "idk_target_cases": len(idk_target),
        "predicted_idk_cases": len(predicted_idk),
        "gold_available_cases": len(gold_available),
        "parsed_gold_cases": len(parsed_gold),
        "gold_coverage": (
            len(gold_available) / len(outcomes) if outcomes else None
        ),
        "parse_rate": len(parsed) / len(outcomes) if outcomes else None,
        "unparsed_rate": _rate([
            row["parse_status"] == "unparsed" for row in outcomes
        ]),
        "non_evaluable_response_rate": _rate([
            row["parse_status"] in {"unparsed", "multiple"}
            for row in outcomes
        ]),
        "empty_answer_rate": _rate([
            not str(row["answer"]).strip() for row in outcomes
        ]),
        "explicit_idk_rate": _rate([
            row["parse_status"] == "idk" for row in outcomes
        ]),
        "multiple_value_rate": _rate([
            row["parse_status"] == "multiple" for row in outcomes
        ]),
        "max_token_rate": _rate([
            row["finish_reason"] == "max_tokens" for row in outcomes
        ]),
        "mean_generated_tokens": _rate(generated_token_counts),
        "mean_answer_words": _rate([
            int(row["answer_words"]) for row in outcomes
        ]),
        "truth_accuracy": _rate([
            row["truth_correct"] for row in gold_available
        ]),
        "truth_accuracy_when_parsed": _rate([
            row["truth_correct"] for row in parsed_gold
        ]),
        "exact_gold_accuracy": _rate([
            row["exact_gold"] for row in gold_available
        ]),
        "gold_overprecision_rate": _rate([
            row["gold_overprecise"] for row in gold_available
        ]),
        "target_decision_accuracy": _rate([
            row["target_decision_correct"] for row in target_available
        ]),
        "target_match_accuracy": _rate([
            row["target_match"] for row in target_available
        ]),
        "known_target_level_accuracy": _rate([
            row["target_match"] for row in known_target
        ]),
        "known_target_level_accuracy_when_parsed": _rate([
            row["target_match"] for row in parsed_known_target
        ]),
        "quadratic_weighted_kappa_when_parsed": quadratic_weighted_kappa(
            kappa_targets,
            kappa_predictions,
            taxonomy=taxonomy,
        ),
        "quadratic_weighted_kappa_cases": len(parsed_known_target),
        "quadratic_weighted_kappa_coverage": (
            len(parsed_known_target) / len(known_target)
            if known_target
            else None
        ),
        "extended_quadratic_weighted_kappa": (
            extended_quadratic_weighted_kappa(
                extended_kappa_targets,
                extended_kappa_predictions,
                taxonomy=taxonomy,
            )
        ),
        "extended_quadratic_weighted_kappa_cases": len(extended_kappa_rows),
        "extended_quadratic_weighted_kappa_coverage": (
            len(extended_kappa_rows) / len(target_available)
            if target_available
            else None
        ),
        "target_overprecision_rate": _rate([
            row["granularity_error"] is not None
            and float(row["granularity_error"]) > 0
            for row in known_target
        ]),
        "target_underprecision_rate": _rate([
            row["granularity_error"] is not None
            and float(row["granularity_error"]) < 0
            for row in known_target
        ]),
        "target_overprecision_rate_when_parsed": _rate([
            error > 0 for error in errors
        ]),
        "target_underprecision_rate_when_parsed": _rate([
            error < 0 for error in errors
        ]),
        "joint_evaluable_cases": len(joint_available),
        "joint_target_truth_accuracy": _rate([
            row["joint_target_truth"] for row in joint_available
        ]),
        "idk_precision": (
            true_positive_idk / len(predicted_idk) if predicted_idk else None
        ),
        "idk_recall": (
            sum(row["parse_status"] == "idk" for row in idk_target)
            / len(idk_target)
            if idk_target
            else None
        ),
        "mean_granularity_error_when_parsed": _rate(errors),
        "mean_absolute_granularity_error_when_parsed": _rate([
            abs(error) for error in errors
        ]),
        "temporal_utility": _rate([
            row["utility"] for row in gold_available
        ]),
        "temporal_utility_uniform_wrong_penalty": _rate([
            row["utility_uniform_wrong_penalty"] for row in gold_available
        ]),
        "temporal_utility_accuracy": _rate([
            row["utility_accuracy"] for row in gold_available
        ]),
        "temporal_utility_positive_informativeness": _rate([
            row["utility_positive_informativeness"] for row in gold_available
        ]),
    }


def _distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _crosstab(outcomes: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [
        row for row in outcomes if row["target_status"] != "unavailable"
    ]
    target_categories = sorted({str(row["target_category"]) for row in eligible})
    prediction_categories = sorted(
        {str(row["prediction_category"]) for row in eligible}
    )
    counts = {
        target: {
            prediction: sum(
                row["target_category"] == target
                and row["prediction_category"] == prediction
                for row in eligible
            )
            for prediction in prediction_categories
        }
        for target in target_categories
    }
    normalized = {}
    for target, row in counts.items():
        total = sum(row.values())
        normalized[target] = {
            prediction: count / total if total else None
            for prediction, count in row.items()
        }
    return {"counts": counts, "row_normalized": normalized}


def aggregate_temporal_outcomes(
    outcomes: Sequence[dict[str, object]],
) -> dict[str, object]:
    metrics = _core_metrics(outcomes)
    metrics["target_distribution"] = _distribution([
        str(row["target_category"]) for row in outcomes
    ])
    metrics["prediction_distribution"] = _distribution([
        str(row["prediction_category"]) for row in outcomes
    ])
    metrics["finish_reason_distribution"] = _distribution([
        str(row["finish_reason"]) for row in outcomes
    ])
    metrics["target_prediction_crosstab"] = _crosstab(outcomes)
    metrics["by_relation"] = {
        relation: _core_metrics([
            row for row in outcomes if row["relation"] == relation
        ])
        for relation in ("birth", "death")
    }
    by_target = {}
    for level in TemporalLevel:
        rows = [
            row for row in outcomes if row["target_level"] == level.value
        ]
        if rows:
            by_target[level.value] = _core_metrics(rows)
    idk_rows = [row for row in outcomes if row["target_status"] == "idk"]
    if idk_rows:
        by_target["IDK"] = _core_metrics(idk_rows)
    metrics["by_target"] = by_target
    return metrics


def paired_qid_bootstrap(
    base_outcomes: Sequence[dict[str, object]],
    finetuned_outcomes: Sequence[dict[str, object]],
    *,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    metric_names = (
        "target_decision_accuracy",
        "target_match_accuracy",
        "known_target_level_accuracy",
        "quadratic_weighted_kappa_when_parsed",
        "extended_quadratic_weighted_kappa",
        "target_overprecision_rate",
        "target_underprecision_rate",
        "truth_accuracy",
        "exact_gold_accuracy",
        "joint_target_truth_accuracy",
        "temporal_utility",
        "temporal_utility_uniform_wrong_penalty",
        "temporal_utility_accuracy",
        "temporal_utility_positive_informativeness",
    )
    return _paired_qid_bootstrap(
        base_outcomes, finetuned_outcomes, metric_names=metric_names, compute_metrics=_core_metrics,
        num_resamples=num_resamples, confidence_level=confidence_level, seed=seed,
    )

