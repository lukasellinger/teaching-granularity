from copy import deepcopy

import pytest

from evaluation.vital_persons.agreement import common_set_kappa_analysis
from evaluation.vital_persons.cases import Modality


def _temporal_row(
    index: int,
    target_status: str,
    target_level: str | None,
    parse_status: str,
    prediction: str | None,
) -> dict[str, object]:
    return {
        "case_id": f"case-{index}",
        "qid": f"Q{index}",
        "target_status": target_status,
        "target_level": target_level,
        "parse_status": parse_status,
        "prediction_category": prediction,
        "temporal_taxonomy": "five-level",
    }


def test_common_set_kappa_uses_one_intersection_for_every_variant() -> None:
    finetuned = [
        _temporal_row(1, "known", "YEAR", "parsed", "YEAR"),
        _temporal_row(2, "known", "MONTH", "parsed", "MONTH"),
        _temporal_row(3, "known", "DAY", "parsed", "DAY"),
        _temporal_row(4, "known", "YEAR", "parsed", "MONTH"),
        _temporal_row(5, "idk", None, "idk", "IDK"),
    ]
    base = deepcopy(finetuned)
    base[1]["prediction_category"] = "YEAR"
    base[2]["parse_status"] = "unparsed"
    base[2]["prediction_category"] = None
    base[3]["prediction_category"] = "MONTH"
    prompt = deepcopy(finetuned)
    prompt[0]["prediction_category"] = "MONTH"
    prompt[3]["parse_status"] = "multiple"
    prompt[3]["prediction_category"] = None
    prompt[4]["parse_status"] = "parsed"
    prompt[4]["prediction_category"] = "CENTURY"

    result = common_set_kappa_analysis(
        {
            "finetuned": finetuned,
            "base_plain": base,
            "base_prompt": prompt,
        },
        Modality.TEMPORAL,
        num_resamples=100,
        confidence_level=0.95,
        seed=42,
    )

    ordinary = result["ordinary"]
    assert ordinary["denominator_cases"] == 4
    assert ordinary["common_cases"] == 2
    assert ordinary["coverage"] == 0.5
    assert ordinary["variant_eligible_cases"] == {
        "finetuned": 4,
        "base_plain": 3,
        "base_prompt": 3,
    }
    assert ordinary["target_distribution"] == {"MONTH": 1, "YEAR": 1}
    assert ordinary["common_case_ids"] == ["case-1", "case-2"]
    assert len(ordinary["common_case_sha256"]) == 64
    assert {
        values["cases"] for values in ordinary["by_variant"].values()
    } == {2}
    assert ordinary["by_variant"]["finetuned"][
        "quadratic_weighted_kappa"
    ] == 1.0
    comparison = ordinary["finetuned_minus_baselines"]["base_plain"]
    assert comparison["requested_resamples"] == 100
    assert 0 < comparison["valid_resamples"] <= 100

    extended = result["extended"]
    assert extended["denominator_cases"] == 5
    assert extended["common_cases"] == 3
    assert extended["coverage"] == 0.6
    assert extended["target_distribution"] == {
        "IDK": 1,
        "MONTH": 1,
        "YEAR": 1,
    }
    assert {values["cases"] for values in extended["by_variant"].values()} == {
        3
    }
    assert "extended_quadratic_weighted_kappa" in extended["by_variant"][
        "finetuned"
    ]


def test_common_set_kappa_supports_location_and_no_bootstrap() -> None:
    rows = [
        {
            "case_id": "place-1",
            "qid": "Q1",
            "target_status": "known",
            "target_level": "COUNTRY",
            "parse_status": "parsed",
            "prediction_category": "COUNTRY",
        },
        {
            "case_id": "place-2",
            "qid": "Q2",
            "target_status": "known",
            "target_level": "CITY",
            "parse_status": "parsed",
            "prediction_category": "CITY",
        },
    ]
    result = common_set_kappa_analysis(
        {"finetuned": rows, "base_plain": deepcopy(rows)},
        Modality.LOCATION,
        num_resamples=0,
        confidence_level=0.95,
        seed=42,
    )

    assert result["ordinary"]["by_variant"]["finetuned"] == {
        "quadratic_weighted_kappa": 1.0,
        "cases": 2,
    }
    comparison = result["ordinary"]["finetuned_minus_baselines"][
        "base_plain"
    ]
    assert comparison["difference"]["estimate"] == 0.0
    assert comparison["difference"]["ci_lower"] is None
    assert comparison["valid_resamples"] == 0


def test_common_set_kappa_rejects_different_targets() -> None:
    finetuned = [_temporal_row(1, "known", "YEAR", "parsed", "YEAR")]
    base = deepcopy(finetuned)
    base[0]["target_level"] = "MONTH"

    with pytest.raises(ValueError, match="inconsistent target metadata"):
        common_set_kappa_analysis(
            {"finetuned": finetuned, "base_plain": base},
            Modality.TEMPORAL,
            num_resamples=0,
            confidence_level=0.95,
            seed=42,
        )
