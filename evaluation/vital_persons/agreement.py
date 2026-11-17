

"""Within-model weighted-kappa comparisons on one fixed case intersection."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np

from .location import (
    extended_quadratic_weighted_kappa as extended_location_kappa,
)
from .location import (
    quadratic_weighted_kappa as location_kappa,
)
from .cases import (
    LocationLevel,
    Modality,
    TemporalLevel,
    TemporalTaxonomy,
)
from .temporal import (
    extended_quadratic_weighted_kappa as extended_temporal_kappa,
)
from .temporal import (
    quadratic_weighted_kappa as temporal_kappa,
)


KappaMode = Literal["ordinary", "extended"]
Outcome = Mapping[str, object]


def _index_outcomes(
    outcomes_by_variant: Mapping[str, Sequence[Outcome]],
) -> tuple[list[str], dict[str, dict[str, Outcome]]]:
    if not outcomes_by_variant:
        raise ValueError("common-set kappa needs at least one variant")
    indexed: dict[str, dict[str, Outcome]] = {}
    reference_ids: list[str] | None = None
    for variant, outcomes in outcomes_by_variant.items():
        by_case: dict[str, Outcome] = {}
        case_ids: list[str] = []
        for row in outcomes:
            case_id = row.get("case_id")
            if not isinstance(case_id, str):
                raise ValueError(f"{variant} outcome has no string case_id")
            if case_id in by_case:
                raise ValueError(f"duplicate {variant} outcome for {case_id}")
            by_case[case_id] = row
            case_ids.append(case_id)
        if reference_ids is None:
            reference_ids = case_ids
        elif set(case_ids) != set(reference_ids):
            raise ValueError(
                "common-set kappa variants do not contain identical case IDs"
            )
        indexed[variant] = by_case
    assert reference_ids is not None
    return reference_ids, indexed


def _validate_shared_targets(
    case_ids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Outcome]],
) -> None:
    reference_variant = next(iter(indexed))
    for case_id in case_ids:
        reference = indexed[reference_variant][case_id]
        if not isinstance(reference.get("qid"), str):
            raise ValueError(f"common-set outcome {case_id} has no string qid")
        expected = (
            reference.get("qid"),
            reference.get("relation"),
            reference.get("modality"),
            reference.get("target_status"),
            reference.get("target_level"),
            reference.get("temporal_taxonomy"),
        )
        for variant, rows in indexed.items():
            actual = (
                rows[case_id].get("qid"),
                rows[case_id].get("relation"),
                rows[case_id].get("modality"),
                rows[case_id].get("target_status"),
                rows[case_id].get("target_level"),
                rows[case_id].get("temporal_taxonomy"),
            )
            if actual != expected:
                raise ValueError(
                    "common-set kappa found inconsistent target metadata for "
                    f"{case_id}: {reference_variant}={expected}, "
                    f"{variant}={actual}"
                )


def _temporal_taxonomy(
    case_ids: Sequence[str],
    reference: Mapping[str, Outcome],
) -> TemporalTaxonomy:
    values = {
        str(reference[case_id].get(
            "temporal_taxonomy",
            TemporalTaxonomy.FINE.value,
        ))
        for case_id in case_ids
    }
    if len(values) != 1:
        raise ValueError(
            f"common-set temporal outcomes mix taxonomies: {sorted(values)}"
        )
    return TemporalTaxonomy(values.pop())


def _eligible(row: Outcome, mode: KappaMode) -> bool:
    if mode == "ordinary":
        return (
            row.get("target_status") == "known"
            and row.get("parse_status") == "parsed"
        )
    return (
        row.get("target_status") in {"known", "idk"}
        and row.get("parse_status") in {"parsed", "idk"}
    )


def _denominator_eligible(row: Outcome, mode: KappaMode) -> bool:
    if mode == "ordinary":
        return row.get("target_status") == "known"
    return row.get("target_status") in {"known", "idk"}


def _category_pairs(
    case_ids: Sequence[str],
    reference: Mapping[str, Outcome],
    variant_rows: Mapping[str, Outcome],
    modality: Modality,
    mode: KappaMode,
) -> tuple[list[object], list[object]]:
    level_type = TemporalLevel if modality is Modality.TEMPORAL else LocationLevel
    targets: list[object] = []
    predictions: list[object] = []
    for case_id in case_ids:
        target_row = reference[case_id]
        prediction_row = variant_rows[case_id]
        if mode == "extended" and target_row["target_status"] == "idk":
            target: object = "IDK"
        else:
            target = level_type(str(target_row["target_level"]))
        if mode == "extended" and prediction_row["parse_status"] == "idk":
            prediction: object = "IDK"
        else:
            prediction = level_type(str(prediction_row["prediction_category"]))
        targets.append(target)
        predictions.append(prediction)
    return targets, predictions


def _scorer(
    modality: Modality,
    mode: KappaMode,
    taxonomy: TemporalTaxonomy | None,
) -> Callable[[Sequence[object], Sequence[object]], float | None]:
    if modality is Modality.LOCATION:
        return location_kappa if mode == "ordinary" else extended_location_kappa

    temporal_function = (
        temporal_kappa if mode == "ordinary" else extended_temporal_kappa
    )

    def score(
        targets: Sequence[object],
        predictions: Sequence[object],
    ) -> float | None:
        assert taxonomy is not None
        return temporal_function(  # type: ignore[arg-type]
            targets,
            predictions,
            taxonomy=taxonomy,
        )

    return score


def _score_variant(
    case_ids: Sequence[str],
    reference: Mapping[str, Outcome],
    variant_rows: Mapping[str, Outcome],
    modality: Modality,
    mode: KappaMode,
    taxonomy: TemporalTaxonomy | None,
) -> float | None:
    targets, predictions = _category_pairs(
        case_ids,
        reference,
        variant_rows,
        modality,
        mode,
    )
    return _scorer(modality, mode, taxonomy)(targets, predictions)


def _bootstrap_differences(
    common_ids: Sequence[str],
    reference: Mapping[str, Outcome],
    indexed: Mapping[str, Mapping[str, Outcome]],
    modality: Modality,
    mode: KappaMode,
    taxonomy: TemporalTaxonomy | None,
    *,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    if "finetuned" not in indexed:
        return {}
    baselines = [variant for variant in indexed if variant != "finetuned"]
    if not baselines:
        return {}
    by_qid: dict[str, list[str]] = defaultdict(list)
    for case_id in common_ids:
        qid = reference[case_id].get("qid")
        if not isinstance(qid, str):
            raise ValueError(f"common-set outcome {case_id} has no string qid")
        by_qid[qid].append(case_id)
    qids = sorted(by_qid)
    point_scores = {
        variant: _score_variant(
            common_ids,
            reference,
            rows,
            modality,
            mode,
            taxonomy,
        )
        for variant, rows in indexed.items()
    }
    samples: dict[str, list[float]] = {baseline: [] for baseline in baselines}
    if num_resamples and qids:
        rng = np.random.default_rng(seed)
        for _ in range(num_resamples):
            sampled_qids = rng.choice(qids, size=len(qids), replace=True)
            sampled_ids = [
                case_id
                for qid in sampled_qids
                for case_id in by_qid[str(qid)]
            ]
            sampled_scores = {
                variant: _score_variant(
                    sampled_ids,
                    reference,
                    rows,
                    modality,
                    mode,
                    taxonomy,
                )
                for variant, rows in indexed.items()
            }
            finetuned = sampled_scores["finetuned"]
            if finetuned is None:
                continue
            for baseline in baselines:
                baseline_score = sampled_scores[baseline]
                if baseline_score is not None:
                    samples[baseline].append(finetuned - baseline_score)

    alpha = (1.0 - confidence_level) / 2.0
    comparisons: dict[str, object] = {}
    for baseline in baselines:
        finetuned_score = point_scores["finetuned"]
        baseline_score = point_scores[baseline]
        differences = samples[baseline]
        estimate = (
            finetuned_score - baseline_score
            if finetuned_score is not None and baseline_score is not None
            else None
        )
        lower = (
            float(np.quantile(differences, alpha)) if differences else None
        )
        upper = (
            float(np.quantile(differences, 1.0 - alpha))
            if differences
            else None
        )
        comparisons[baseline] = {
            "method": "paired_qid_cluster_percentile",
            "difference_definition": "finetuned_minus_baseline",
            "baseline": baseline_score,
            "finetuned": finetuned_score,
            "difference": {
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "excludes_zero": (
                    (lower > 0.0 or upper < 0.0)
                    if lower is not None and upper is not None
                    else None
                ),
            },
            "requested_resamples": num_resamples,
            "valid_resamples": len(differences),
            "num_qid_clusters": len(qids),
            "confidence_level": confidence_level,
            "seed": seed,
        }
    return comparisons


def _mode_analysis(
    mode: KappaMode,
    case_ids: Sequence[str],
    reference: Mapping[str, Outcome],
    indexed: Mapping[str, Mapping[str, Outcome]],
    modality: Modality,
    taxonomy: TemporalTaxonomy | None,
    *,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    denominator_ids = [
        case_id
        for case_id in case_ids
        if _denominator_eligible(reference[case_id], mode)
    ]
    common_ids = [
        case_id
        for case_id in denominator_ids
        if all(_eligible(rows[case_id], mode) for rows in indexed.values())
    ]
    qids = {str(reference[case_id]["qid"]) for case_id in common_ids}
    target_distribution = Counter(
        "IDK"
        if mode == "extended"
        and reference[case_id]["target_status"] == "idk"
        else str(reference[case_id]["target_level"])
        for case_id in common_ids
    )
    scores = {
        variant: _score_variant(
            common_ids,
            reference,
            rows,
            modality,
            mode,
            taxonomy,
        )
        for variant, rows in indexed.items()
    }
    variant_eligible_cases = {
        variant: sum(
            _eligible(rows[case_id], mode) for case_id in denominator_ids
        )
        for variant, rows in indexed.items()
    }
    metric_name = (
        "quadratic_weighted_kappa"
        if mode == "ordinary"
        else "extended_quadratic_weighted_kappa"
    )
    return {
        "eligibility": (
            "known_target_and_parsed_by_every_variant"
            if mode == "ordinary"
            else "target_available_and_parsed_or_idk_by_every_variant"
        ),
        "denominator_cases": len(denominator_ids),
        "common_cases": len(common_ids),
        "common_qids": len(qids),
        "common_case_ids": common_ids,
        "common_case_sha256": hashlib.sha256(
            "\n".join(sorted(common_ids)).encode("utf-8")
        ).hexdigest(),
        "coverage": (
            len(common_ids) / len(denominator_ids)
            if denominator_ids
            else None
        ),
        "target_distribution": dict(sorted(target_distribution.items())),
        "variant_eligible_cases": variant_eligible_cases,
        "by_variant": {
            variant: {
                metric_name: score,
                "cases": len(common_ids),
            }
            for variant, score in scores.items()
        },
        "finetuned_minus_baselines": _bootstrap_differences(
            common_ids,
            reference,
            indexed,
            modality,
            mode,
            taxonomy,
            num_resamples=num_resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
    }


def common_set_kappa_analysis(
    outcomes_by_variant: Mapping[str, Sequence[Outcome]],
    modality: Modality,
    *,
    num_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Score every within-model variant on exactly the same eligible cases."""
    if num_resamples < 0:
        raise ValueError("num_resamples must be non-negative")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    case_ids, indexed = _index_outcomes(outcomes_by_variant)
    _validate_shared_targets(case_ids, indexed)
    reference = indexed[next(iter(indexed))]
    taxonomy = (
        _temporal_taxonomy(case_ids, reference)
        if modality is Modality.TEMPORAL
        else None
    )
    return {
        "scope": "within_model_all_evaluated_variants_intersection",
        "variants": list(indexed),
        "ordinary": _mode_analysis(
            "ordinary",
            case_ids,
            reference,
            indexed,
            modality,
            taxonomy,
            num_resamples=num_resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
        "extended": _mode_analysis(
            "extended",
            case_ids,
            reference,
            indexed,
            modality,
            taxonomy,
            num_resamples=num_resamples,
            confidence_level=confidence_level,
            seed=seed,
        ),
    }
