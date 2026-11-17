"""Score normalized event locations; failures remain explicit outcomes."""
from __future__ import annotations
import math
from shared.bootstrap import paired_qid_bootstrap as _paired_qid_bootstrap
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from typing import Literal, Sequence
import numpy as np
from parsing.location.normalization.values import LOCATION_PRECISION, LocationValue, LocationLevel
from .cases import LocationEvaluationCase, location_value_dict
from shared.geography import FULL, FOUR_LEVEL, FOUR_LEVELS, project_location


LocationParseStatus = Literal[
    "parsed",
    "idk",
    "unparsed",
    "multiple",
    "extraction_failed",
    "normalization_failed",
    "contradictory",
]


# Full-precision estimates regenerated from the adapted gold, not rounded constants.
import json as _json
from pathlib import Path as _Path
_LOCATION_INFORMATION = _json.loads(_Path(__file__).with_name('location_informativeness.json').read_text())
_LOCATION_REMAINING_CANDIDATES = {
    level: _LOCATION_INFORMATION['remaining_candidates'].get(level.value.lower(), 1.0)
    for level in LocationLevel
}
_LOCATION_PRIOR = float(_LOCATION_INFORMATION['prior_candidates'])


_ALIASES = {
    "u.s": "united states",
    "u.s.": "united states",
    "u.s.a": "united states",
    "u.s.a.": "united states",
    "us": "united states",
    "usa": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k": "united kingdom",
    "u.k.": "united kingdom",
}


@dataclass(frozen=True)
class ParsedLocationResponse:
    status: LocationParseStatus
    value: LocationValue | None
    explicit_levels: tuple[LocationLevel, ...]
    matched_values: tuple[LocationValue, ...]
    pipeline_failure_stage: Literal["extraction", "normalization"] | None = None
    pipeline_error_types: tuple[str, ...] = ()
    pipeline_error_messages: tuple[str, ...] = ()


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = " ".join(normalized.split()).strip(" .,")
    return _ALIASES.get(normalized, normalized)


def _normalized_location(value: LocationValue) -> LocationValue:
    return LocationValue(
        level=value.level,
        needs_disambiguation=value.needs_disambiguation,
        place=tuple(_normalize_label(item) for item in value.place) if value.place else None,
        subcity=tuple(_normalize_label(item) for item in value.subcity) if value.subcity else None,
        city=_normalize_label(value.city) if value.city else None,
        region=(
            tuple(_normalize_label(item) for item in value.region)
            if value.region
            else None
        ),
        country=_normalize_label(value.country) if value.country else None,
        continent=_normalize_label(value.continent) if value.continent else None,
    )


def _is_compatible_broader_location(
    finer: LocationValue, broader: LocationValue
) -> bool:
    """Require a genuine finer-to-broader hierarchy, including shared context."""
    if finer.precision <= broader.precision:
        return False
    finer = _normalized_location(finer)
    broader = _normalized_location(broader)
    try:
        projected = finer.to_level(broader.level)
    except ValueError:
        return False
    if broader.level in {LocationLevel.REGION, LocationLevel.SUBCITY}:
        field = "region" if broader.level is LocationLevel.REGION else "subcity"
        if set(getattr(projected, field) or ()).isdisjoint(getattr(broader, field) or ()):
            return False
    elif projected.comparison_key() != broader.comparison_key():
        return False
    for field in ("place", "subcity", "city", "region", "country", "continent"):
        fine_context = getattr(finer, field)
        broad_context = getattr(broader, field)
        if fine_context is None or broad_context is None:
            continue
        if field in {"place", "region", "subcity"}:
            if set(fine_context).isdisjoint(broad_context):
                return False
        elif fine_context != broad_context:
            return False
    return True


def has_required_disambiguation_context(
    prediction: LocationValue,
    gold: LocationValue,
    explicit_levels: Sequence[LocationLevel],
) -> bool:
    if not gold.needs_disambiguation or prediction.level not in {
        LocationLevel.PLACE,
        LocationLevel.SUBCITY,
        LocationLevel.CITY,
    }:
        return True
    return any(
        LOCATION_PRECISION[level] < prediction.precision for level in explicit_levels
    )


def correct_at_prediction_level(
    prediction: LocationValue | None,
    gold: LocationValue,
    explicit_levels: Sequence[LocationLevel] = (),
) -> bool:
    """Require hierarchy consistency at the prediction's expressed level.

    Broader hierarchy inferred during normalization may disambiguate the
    written value. Requiring that broader context to be written explicitly is
    retained separately in ``correct_at_prediction_level_strict_context``.
    """
    if prediction is None or prediction.precision > gold.precision:
        return False
    normalized_prediction = _normalized_location(prediction)
    normalized_gold = _normalized_location(gold)
    fields = {
        LocationLevel.PLACE: ("place", "subcity", "city", "region", "country", "continent"),
        LocationLevel.SUBCITY: ("subcity", "city", "region", "country", "continent"),
        LocationLevel.CITY: ("city", "region", "country", "continent"),
        LocationLevel.REGION: ("region", "country", "continent"),
        LocationLevel.COUNTRY: ("country", "continent"),
        LocationLevel.CONTINENT: ("continent",),
    }[prediction.level]

    def shared_context_matches() -> bool:
        for field in fields:
            predicted = getattr(normalized_prediction, field)
            reference = getattr(normalized_gold, field)
            if predicted is None or reference is None:
                continue
            if field in {"region", "subcity"}:
                if set(predicted).isdisjoint(reference):
                    return False
            elif predicted != reference:
                return False
        return True

    return bool(
        normalized_prediction.consistent_with(normalized_gold)
        and shared_context_matches()
    )


def correct_at_prediction_level_strict_context(
    prediction: LocationValue | None,
    gold: LocationValue,
    explicit_levels: Sequence[LocationLevel] = (),
) -> bool:
    """Also require an explicit broader component for ambiguous city/place gold."""
    return bool(
        correct_at_prediction_level(prediction, gold, explicit_levels)
        and prediction is not None
        and has_required_disambiguation_context(
            prediction,
            gold,
            explicit_levels,
        )
    )


def location_informativeness(level: LocationLevel) -> float:
    remaining = _LOCATION_REMAINING_CANDIDATES[level]
    best = _LOCATION_REMAINING_CANDIDATES[LocationLevel.CITY]
    return math.log(_LOCATION_PRIOR / remaining) / math.log(_LOCATION_PRIOR / best)


def quadratic_weighted_kappa(
    targets: Sequence[LocationLevel],
    predictions: Sequence[LocationLevel],
) -> float | None:
    if len(targets) != len(predictions):
        raise ValueError("kappa targets and predictions must have equal length")
    if not targets:
        return None
    target_ranks = np.asarray([LOCATION_PRECISION[level] - 1 for level in targets])
    prediction_ranks = np.asarray(
        [LOCATION_PRECISION[level] - 1 for level in predictions]
    )
    scale = max(len(LocationLevel) - 1, 1)
    observed = float(np.mean(((target_ranks - prediction_ranks) / scale) ** 2))
    target_frequency = np.bincount(
        target_ranks,
        minlength=len(LocationLevel),
    ) / len(targets)
    prediction_frequency = np.bincount(
        prediction_ranks,
        minlength=len(LocationLevel),
    ) / len(predictions)
    ranks = np.arange(len(LocationLevel))
    weights = ((ranks[:, None] - ranks[None, :]) / scale) ** 2
    expected = float(
        np.sum(weights * target_frequency[:, None] * prediction_frequency[None, :])
    )
    if math.isclose(expected, 0.0, abs_tol=1e-15):
        return None
    return 1.0 - observed / expected


def extended_quadratic_weighted_kappa(
    targets: Sequence[LocationLevel | Literal["IDK"]],
    predictions: Sequence[LocationLevel | Literal["IDK"]],
) -> float | None:
    """Ordinal agreement with IDK below CONTINENT."""
    if len(targets) != len(predictions):
        raise ValueError("kappa targets and predictions must have equal length")
    if not targets:
        return None

    def rank(category: LocationLevel | Literal["IDK"]) -> int:
        if category == "IDK":
            return 0
        return LOCATION_PRECISION[category]

    target_ranks = np.asarray([rank(category) for category in targets])
    prediction_ranks = np.asarray([rank(category) for category in predictions])
    rank_count = len(LocationLevel) + 1
    scale = rank_count - 1
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


def score_location_prediction(
    case: LocationEvaluationCase,
    generation: dict[str, object],
    parsed: ParsedLocationResponse,
    *, rejected_mentions: Sequence[dict] = (),
) -> dict[str, object]:
    """Score one validated event-location prediction."""
    answer = generation.get("answer")
    if not isinstance(answer, str):
        raise ValueError(f"generation for {case.case_id} has no string answer")
    if case.location_taxonomy == FOUR_LEVEL:
        if case.target_level is not None and case.target_level not in FOUR_LEVELS:
            raise ValueError("Four-level evaluation requires recomputed four-level targets")
        case = replace(case, gold=project_location(case.gold, FOUR_LEVEL))
        projected = project_location(parsed.value, FOUR_LEVEL)
        explicit = tuple(level for level in parsed.explicit_levels if level in FOUR_LEVELS)
        if projected is not None and parsed.value.level not in FOUR_LEVELS:
            explicit = tuple(dict.fromkeys((*explicit, projected.level)))
        parsed = replace(parsed, value=projected, explicit_levels=explicit,
                         matched_values=tuple(value for raw in parsed.matched_values
                                              if (value := project_location(raw, FOUR_LEVEL)) is not None))
        if parsed.status == "parsed" and projected is None:
            parsed = replace(parsed, status="normalization_failed", pipeline_failure_stage="normalization",
                             pipeline_error_types=("no_supported_taxonomy_level",),
                             pipeline_error_messages=("No resolved city, region, country or continent",))
    elif case.location_taxonomy != FULL:
        raise ValueError(f"Unknown location taxonomy: {case.location_taxonomy}")
    prediction = parsed.value
    truth_correct = (
        correct_at_prediction_level(
            prediction,
            case.gold,
            parsed.explicit_levels,
        )
        if case.gold is not None
        else None
    )
    strict_context_truth_correct = (
        correct_at_prediction_level_strict_context(
            prediction,
            case.gold,
            parsed.explicit_levels,
        )
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
    strict_context_exact_gold = (
        bool(
            strict_context_truth_correct
            and prediction is not None
            and prediction.level is case.gold.level
        )
        if case.gold is not None
        else None
    )
    if case.target_status == "idk":
        target_decision_correct: bool | None = parsed.status == "idk"
        target_match: bool | None = parsed.status == "idk"
    elif case.target_status == "known":
        target_decision_correct = parsed.status == "parsed"
        target_match = bool(
            prediction is not None and prediction.level is case.target_level
        )
    else:
        target_decision_correct = None
        target_match = None
    if case.target_status == "unavailable":
        joint = None
    elif case.target_status == "idk":
        joint = bool(target_match)
    elif case.gold is None:
        joint = None
    else:
        joint = bool(target_match and truth_correct)
    if case.target_status == "unavailable":
        strict_context_joint = None
    elif case.target_status == "idk":
        strict_context_joint = bool(target_match)
    elif case.gold is None:
        strict_context_joint = None
    else:
        strict_context_joint = bool(target_match and strict_context_truth_correct)
    error = (
        prediction.precision - LOCATION_PRECISION[case.target_level]
        if (
            prediction is not None
            and case.target_status == "known"
            and case.target_level is not None
        )
        else None
    )
    utility = (
        None
        if case.gold is None
        else (
            location_informativeness(prediction.level)
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
                location_informativeness(prediction.level)
                if truth_correct and prediction is not None
                else -1.0
            )
        )
    )
    utility_accuracy = None if case.gold is None else float(bool(truth_correct))
    utility_positive_informativeness = (
        None
        if case.gold is None
        else (
            location_informativeness(prediction.level)
            if truth_correct and prediction is not None
            else 0.0
        )
    )
    strict_context_utility = (
        None
        if case.gold is None
        else (
            location_informativeness(prediction.level)
            * (1.0 if strict_context_truth_correct else -1.0)
            if prediction is not None
            else 0.0
        )
    )
    strict_context_utility_uniform_wrong_penalty = (
        None
        if case.gold is None
        else (
            0.0
            if parsed.status == "idk"
            else (
                location_informativeness(prediction.level)
                if strict_context_truth_correct and prediction is not None
                else -1.0
            )
        )
    )
    strict_context_utility_accuracy = (
        None if case.gold is None else float(bool(strict_context_truth_correct))
    )
    strict_context_utility_positive_informativeness = (
        None
        if case.gold is None
        else (
            location_informativeness(prediction.level)
            if strict_context_truth_correct and prediction is not None
            else 0.0
        )
    )
    prediction_category = (
        prediction.level.value if prediction is not None else parsed.status.upper()
    )
    target_category = (
        case.target_level.value
        if case.target_level is not None
        else case.target_status.upper()
    )
    ambiguous_without_context = bool(
        prediction is not None
        and case.gold is not None
        and case.gold.needs_disambiguation
        and not has_required_disambiguation_context(
            prediction,
            case.gold,
            parsed.explicit_levels,
        )
    )
    return {
        "case_id": case.case_id,
        "source_model": case.source_model,
        "split": case.split.value,
        "modality": "location",
        "location_taxonomy": case.location_taxonomy,
        "qid": case.qid,
        "name": case.name,
        "relation": case.relation.value,
        "prompt": case.prompt,
        "generation_prompt": generation.get("user_prompt", case.prompt),
        "system_prompt": generation.get("system_prompt"),
        "gold": location_value_dict(case.gold) if case.gold is not None else None,
        "target_status": case.target_status,
        "target_level": (
            case.target_level.value if case.target_level is not None else None
        ),
        "answer": answer,
        "answer_words": len(re.findall(r"\b\w+\b", answer)),
        "generated_tokens": generation.get("generated_tokens"),
        "finish_reason": generation.get("finish_reason"),
        "parse_status": parsed.status,
        "pipeline_failure_stage": parsed.pipeline_failure_stage,
        "pipeline_error_types": list(parsed.pipeline_error_types),
        "pipeline_error_messages": list(parsed.pipeline_error_messages),
        "rejected_location_mentions": list(rejected_mentions),
        "matched_values": [
            location_value_dict(value) for value in parsed.matched_values
        ],
        "explicit_levels": [level.value for level in parsed.explicit_levels],
        "prediction": (
            location_value_dict(prediction) if prediction is not None else None
        ),
        "prediction_category": prediction_category,
        "target_category": target_category,
        "truth_correct": truth_correct,
        "strict_context_truth_correct": strict_context_truth_correct,
        "exact_gold": exact_gold,
        "strict_context_exact_gold": strict_context_exact_gold,
        "gold_overprecise": (
            bool(prediction is not None and prediction.precision > case.gold.precision)
            if case.gold is not None
            else None
        ),
        "ambiguous_gold_without_context": ambiguous_without_context,
        "target_decision_correct": target_decision_correct,
        "target_match": target_match,
        "joint_target_truth": joint,
        "strict_context_joint_target_truth": strict_context_joint,
        "granularity_error": error,
        "utility": utility,
        "utility_uniform_wrong_penalty": utility_uniform_wrong_penalty,
        "utility_accuracy": utility_accuracy,
        "utility_positive_informativeness": utility_positive_informativeness,
        "strict_context_utility": strict_context_utility,
        "strict_context_utility_uniform_wrong_penalty": (
            strict_context_utility_uniform_wrong_penalty
        ),
        "strict_context_utility_accuracy": strict_context_utility_accuracy,
        "strict_context_utility_positive_informativeness": (
            strict_context_utility_positive_informativeness
        ),
    }


def _rate(values: Sequence[object]) -> float | None:
    return float(np.mean(values)) if values else None


def _core_metrics(outcomes: Sequence[dict[str, object]]) -> dict[str, object]:
    parsed = [row for row in outcomes if row["parse_status"] == "parsed"]
    target_available = [
        row for row in outcomes if row["target_status"] != "unavailable"
    ]
    known_target = [row for row in outcomes if row["target_status"] == "known"]
    idk_target = [row for row in outcomes if row["target_status"] == "idk"]
    gold_available = [row for row in outcomes if row["gold"] is not None]
    parsed_gold = [row for row in gold_available if row["parse_status"] == "parsed"]
    parsed_known = [row for row in known_target if row["parse_status"] == "parsed"]
    joint_available = [
        row for row in target_available if row["joint_target_truth"] is not None
    ]
    errors = [
        float(row["granularity_error"])
        for row in parsed_known
        if row["granularity_error"] is not None
    ]
    predicted_idk = [row for row in target_available if row["parse_status"] == "idk"]
    true_positive_idk = sum(row["target_status"] == "idk" for row in predicted_idk)
    kappa_targets = [LocationLevel(str(row["target_level"])) for row in parsed_known]
    kappa_predictions = [
        LocationLevel(str(row["prediction_category"])) for row in parsed_known
    ]
    extended_kappa_rows = [
        row for row in target_available if row["parse_status"] in {"parsed", "idk"}
    ]
    extended_kappa_targets = [
        (
            "IDK"
            if row["target_status"] == "idk"
            else LocationLevel(str(row["target_level"]))
        )
        for row in extended_kappa_rows
    ]
    extended_kappa_predictions = [
        (
            "IDK"
            if row["parse_status"] == "idk"
            else LocationLevel(str(row["prediction_category"]))
        )
        for row in extended_kappa_rows
    ]
    return {
        "cases": len(outcomes),
        "parsed_cases": len(parsed),
        "target_available_cases": len(target_available),
        "target_coverage": len(target_available) / len(outcomes) if outcomes else None,
        "known_target_cases": len(known_target),
        "parsed_known_target_cases": len(parsed_known),
        "idk_target_cases": len(idk_target),
        "predicted_idk_cases": len(predicted_idk),
        "gold_available_cases": len(gold_available),
        "parsed_gold_cases": len(parsed_gold),
        "gold_coverage": len(gold_available) / len(outcomes) if outcomes else None,
        "parse_rate": len(parsed) / len(outcomes) if outcomes else None,
        "unparsed_rate": _rate([row["parse_status"] == "unparsed" for row in outcomes]),
        "multiple_value_rate": _rate(
            [row["parse_status"] == "multiple" for row in outcomes]
        ),
        "normalization_failure_rate": _rate(
            [row["parse_status"] == "normalization_failed" for row in outcomes]
        ),
        "extraction_failure_rate": _rate(
            [row["parse_status"] == "extraction_failed" for row in outcomes]
        ),
        "normalization_processing_failure_rate": _rate(
            [row["pipeline_failure_stage"] == "normalization" for row in outcomes]
        ),
        "pipeline_failure_rate": _rate(
            [row["pipeline_failure_stage"] is not None for row in outcomes]
        ),
        "contradiction_rate": _rate(
            [row["parse_status"] == "contradictory" for row in outcomes]
        ),
        "non_evaluable_response_rate": _rate(
            [
                row["parse_status"]
                in {
                    "unparsed",
                    "multiple",
                    "extraction_failed",
                    "normalization_failed",
                    "contradictory",
                }
                for row in outcomes
            ]
        ),
        "empty_answer_rate": _rate(
            [not str(row["answer"]).strip() for row in outcomes]
        ),
        "explicit_idk_rate": _rate([row["parse_status"] == "idk" for row in outcomes]),
        "max_token_rate": _rate(
            [row["finish_reason"] == "max_tokens" for row in outcomes]
        ),
        "mean_generated_tokens": _rate(
            [
                int(row["generated_tokens"])
                for row in outcomes
                if isinstance(row["generated_tokens"], int)
                and not isinstance(row["generated_tokens"], bool)
            ]
        ),
        "mean_answer_words": _rate([int(row["answer_words"]) for row in outcomes]),
        "truth_accuracy": _rate([row["truth_correct"] for row in gold_available]),
        "truth_accuracy_when_parsed": _rate(
            [row["truth_correct"] for row in parsed_gold]
        ),
        "strict_context_truth_accuracy": _rate(
            [row["strict_context_truth_correct"] for row in gold_available]
        ),
        "strict_context_truth_accuracy_when_parsed": _rate(
            [row["strict_context_truth_correct"] for row in parsed_gold]
        ),
        "exact_gold_accuracy": _rate([row["exact_gold"] for row in gold_available]),
        "strict_context_exact_gold_accuracy": _rate(
            [row["strict_context_exact_gold"] for row in gold_available]
        ),
        "gold_overprecision_rate": _rate(
            [row["gold_overprecise"] for row in gold_available]
        ),
        "ambiguous_gold_without_context_rate": _rate(
            [row["ambiguous_gold_without_context"] for row in parsed_gold]
        ),
        "target_decision_accuracy": _rate(
            [row["target_decision_correct"] for row in target_available]
        ),
        "target_match_accuracy": _rate(
            [row["target_match"] for row in target_available]
        ),
        "known_target_level_accuracy": _rate(
            [row["target_match"] for row in known_target]
        ),
        "known_target_level_accuracy_when_parsed": _rate(
            [row["target_match"] for row in parsed_known]
        ),
        "quadratic_weighted_kappa_when_parsed": quadratic_weighted_kappa(
            kappa_targets,
            kappa_predictions,
        ),
        "quadratic_weighted_kappa_cases": len(parsed_known),
        "quadratic_weighted_kappa_coverage": (
            len(parsed_known) / len(known_target) if known_target else None
        ),
        "extended_quadratic_weighted_kappa": (
            extended_quadratic_weighted_kappa(
                extended_kappa_targets,
                extended_kappa_predictions,
            )
        ),
        "extended_quadratic_weighted_kappa_cases": len(extended_kappa_rows),
        "extended_quadratic_weighted_kappa_coverage": (
            len(extended_kappa_rows) / len(target_available)
            if target_available
            else None
        ),
        "target_overprecision_rate": _rate(
            [
                row["granularity_error"] is not None
                and float(row["granularity_error"]) > 0
                for row in known_target
            ]
        ),
        "target_underprecision_rate": _rate(
            [
                row["granularity_error"] is not None
                and float(row["granularity_error"]) < 0
                for row in known_target
            ]
        ),
        "target_overprecision_rate_when_parsed": _rate([error > 0 for error in errors]),
        "target_underprecision_rate_when_parsed": _rate(
            [error < 0 for error in errors]
        ),
        "joint_evaluable_cases": len(joint_available),
        "joint_target_truth_accuracy": _rate(
            [row["joint_target_truth"] for row in joint_available]
        ),
        "strict_context_joint_target_truth_accuracy": _rate(
            [row["strict_context_joint_target_truth"] for row in joint_available]
        ),
        "idk_precision": (
            true_positive_idk / len(predicted_idk) if predicted_idk else None
        ),
        "idk_recall": (
            sum(row["parse_status"] == "idk" for row in idk_target) / len(idk_target)
            if idk_target
            else None
        ),
        "mean_granularity_error_when_parsed": _rate(errors),
        "mean_absolute_granularity_error_when_parsed": _rate(
            [abs(error) for error in errors]
        ),
        "location_utility": _rate([row["utility"] for row in gold_available]),
        "location_utility_uniform_wrong_penalty": _rate(
            [row["utility_uniform_wrong_penalty"] for row in gold_available]
        ),
        "location_utility_accuracy": _rate(
            [row["utility_accuracy"] for row in gold_available]
        ),
        "location_utility_positive_informativeness": _rate(
            [row["utility_positive_informativeness"] for row in gold_available]
        ),
        "strict_context_location_utility": _rate(
            [row["strict_context_utility"] for row in gold_available]
        ),
        "strict_context_location_utility_uniform_wrong_penalty": _rate(
            [
                row["strict_context_utility_uniform_wrong_penalty"]
                for row in gold_available
            ]
        ),
        "strict_context_location_utility_accuracy": _rate(
            [row["strict_context_utility_accuracy"] for row in gold_available]
        ),
        "strict_context_location_utility_positive_informativeness": _rate(
            [
                row["strict_context_utility_positive_informativeness"]
                for row in gold_available
            ]
        ),
    }


def _distribution(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _crosstab(outcomes: Sequence[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in outcomes if row["target_status"] != "unavailable"]
    targets = sorted({str(row["target_category"]) for row in eligible})
    predictions = sorted({str(row["prediction_category"]) for row in eligible})
    counts = {
        target: {
            prediction: sum(
                row["target_category"] == target
                and row["prediction_category"] == prediction
                for row in eligible
            )
            for prediction in predictions
        }
        for target in targets
    }
    return {
        "counts": counts,
        "row_normalized": {
            target: {
                prediction: count / sum(row.values()) if sum(row.values()) else None
                for prediction, count in row.items()
            }
            for target, row in counts.items()
        },
    }


def aggregate_location_outcomes(
    outcomes: Sequence[dict[str, object]],
) -> dict[str, object]:
    metrics = _core_metrics(outcomes)
    metrics["target_distribution"] = _distribution(
        [str(row["target_category"]) for row in outcomes]
    )
    metrics["prediction_distribution"] = _distribution(
        [str(row["prediction_category"]) for row in outcomes]
    )
    metrics["finish_reason_distribution"] = _distribution(
        [str(row["finish_reason"]) for row in outcomes]
    )
    metrics["target_prediction_crosstab"] = _crosstab(outcomes)
    metrics["by_relation"] = {
        relation: _core_metrics(
            [row for row in outcomes if row["relation"] == relation]
        )
        for relation in ("birthplace", "deathplace")
    }
    by_target = {}
    for level in LocationLevel:
        rows = [row for row in outcomes if row["target_level"] == level.value]
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
        "location_utility",
        "location_utility_uniform_wrong_penalty",
        "location_utility_accuracy",
        "location_utility_positive_informativeness",
        "strict_context_truth_accuracy",
        "strict_context_exact_gold_accuracy",
        "strict_context_joint_target_truth_accuracy",
        "strict_context_location_utility",
        "strict_context_location_utility_uniform_wrong_penalty",
        "strict_context_location_utility_accuracy",
        "strict_context_location_utility_positive_informativeness",
    )
    return _paired_qid_bootstrap(
        base_outcomes, finetuned_outcomes, metric_names=metric_names, compute_metrics=_core_metrics,
        num_resamples=num_resamples, confidence_level=confidence_level, seed=seed,
    )

