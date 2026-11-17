from evaluation.vital_persons.cases import TemporalEvaluationCase
from evaluation.vital_persons.cases import Relation, Split, TemporalLevel, TemporalTaxonomy
from evaluation.vital_persons.temporal import (
    correct_at_prediction_level,
    extended_quadratic_weighted_kappa,
    is_idk_response,
    paired_qid_bootstrap,
    parse_temporal_response,
    quadratic_weighted_kappa,
    score_temporal_generations,
)
from evaluation.vital_persons.cases import TemporalValue


def case(
    qid,
    relation,
    gold,
    *,
    target_status="known",
    target_level=TemporalLevel.YEAR,
    temporal_taxonomy=TemporalTaxonomy.FINE,
):
    return TemporalEvaluationCase(
        case_id=f"MODEL:{qid}:{relation.value}",
        source_model="MODEL",
        split=Split.TEST,
        qid=qid,
        name=qid,
        relation=relation,
        prompt="question",
        gold=gold,
        target_status=target_status,
        target_level=target_level if target_status == "known" else None,
        temporal_taxonomy=temporal_taxonomy,
    )


def generation(case_id, answer, finish_reason="eos"):
    return {
        "case_id": case_id,
        "answer": answer,
        "generated_tokens": 5,
        "finish_reason": finish_reason,
    }


def test_temporal_parser_selects_requested_event_not_first_date():
    response = "She was born in 1900 and died in 1950."

    parsed = parse_temporal_response(response, Relation.DEATH)

    assert parsed.status == "parsed"
    assert parsed.value.year == 1950


def test_temporal_parser_rejects_distinct_multiple_requested_values():
    response = "Sources say she was born in 1900 or born in 1901."

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "multiple"
    assert parsed.value is None
    assert {value.year for value in parsed.matched_values} == {1900, 1901}


def test_temporal_parser_ignores_dates_of_later_narrative_events():
    responses = [
        (
            "King Juan Carlos I of Spain passed away on **November 19, "
            "2014**, at the age of 87.\n\nHe died at the Royal Palace of "
            "Zarzuela in Madrid, surrounded by his family. His death was "
            "announced by the Spanish royal family later that day, marking "
            "the end of a reign that lasted from 1975 until his abdication "
            "in 2014."
        ),
        (
            "Muhammad Ali Jinnah, the founder of Pakistan, passed away on "
            "**14 December 1948**. He was 89 years old at the time of his "
            "death. His death marked the end of a pivotal era in South Asian "
            "history, as he had been instrumental in the creation of "
            "Pakistan in 1947."
        ),
    ]

    parsed = [
        parse_temporal_response(response, Relation.DEATH)
        for response in responses
    ]

    assert [result.status for result in parsed] == ["parsed", "parsed"]
    assert [result.value.level for result in parsed] == [
        TemporalLevel.DAY,
        TemporalLevel.DAY,
    ]
    assert [result.value.year for result in parsed] == [2014, 1948]


def test_temporal_parser_scopes_later_biographical_dates():
    responses = [
        (
            "Wassily Kandinsky died on December 13, 1944. He was living in "
            "exile, having fled Germany in 1933."
        ),
        (
            "Auguste Comte died on September 5, 1857. His death occurred "
            "shortly after the death of his wife, who had died in 1856."
        ),
        (
            "Slobodan Milošević died on March 11, 2006. His trial was set "
            "to begin on March 16, 2006."
        ),
    ]

    parsed = [
        parse_temporal_response(response, Relation.DEATH)
        for response in responses
    ]

    assert [result.status for result in parsed] == ["parsed"] * 3
    assert [result.value.year for result in parsed] == [1944, 1857, 2006]


def test_later_explicit_requested_event_alternative_remains_multiple():
    response = (
        "Epicurus was born in 341 BC. His birth year is traditionally placed "
        "in 341 BC, though some estimates suggest it could be 342 BC."
    )

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "multiple"
    assert {value.year for value in parsed.matched_values} == {-341, -342}


def test_referential_later_year_alternatives_remain_multiple():
    response = (
        "Historians estimate his birth year as 575 AD, based on the age he "
        "would have been when he ascended to the throne in 610 AD. Some "
        "sources suggest a slightly later date, such as 576 or 577."
    )

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "multiple"
    assert {value.year for value in parsed.matched_values} == {575, 576, 577}


def test_explicit_between_range_remains_multiple():
    response = (
        "Historians place his birth in the early 5th century, likely between "
        "400 and 410 AD."
    )

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "multiple"
    assert {
        value.year for value in parsed.matched_values if value.year is not None
    } == {400, 410}


def test_hyphenated_occupation_period_is_not_a_birth_date():
    response = (
        "The exact birth date of Tulsidas, the renowned 16th-century poet, "
        "is unknown. Historians place his birth around 1511 CE."
    )

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "parsed"
    assert parsed.value.year == 1511


def test_concise_standalone_temporal_answer_is_parsed():
    responses = [
        "around 1401",
        "Approximately 573 CE in Mecca",
        "Correct aggregated answer: July 26, 1946\n"
        "Explanation: This is the aggregate.",
    ]

    parsed = [
        parse_temporal_response(response, Relation.BIRTH)
        for response in responses
    ]

    assert [result.status for result in parsed] == ["parsed"] * 3
    assert [result.value.year for result in parsed] == [1401, 573, 1946]


def test_standalone_temporal_alternatives_remain_multiple():
    responses = [
        "Athens, Greece, around 470 or 469 BCE",
        "December 1923 - January 1924",
        "May or June 1966",
        "1905-1906",
    ]

    parsed = [
        parse_temporal_response(response, Relation.BIRTH)
        for response in responses
    ]

    assert [result.status for result in parsed] == ["multiple"] * 4


def test_short_unrelated_event_is_not_a_standalone_temporal_answer():
    responses = [
        "He ascended to the throne in 610 AD.",
        "He fought a battle in 1900.",
    ]
    parsed = [
        parse_temporal_response(response, Relation.BIRTH)
        for response in responses
    ]

    assert [result.status for result in parsed] == ["unparsed", "unparsed"]


def test_explicit_gregorian_equivalent_is_one_date():
    responses = [
        (
            "Sergei Prokofiev was born on April 23, 1891. His birth date is "
            "given in the Julian calendar; in the modern Gregorian calendar, "
            "this corresponds to May 6, 1891."
        ),
        (
            "Elizabeth died on December 25, 1762 according to the Julian "
            "calendar, which corresponds to January 5, 1763 in the modern "
            "Gregorian calendar."
        ),
        (
            "Husayn died on 10 Muharram, 61 AH. In the Gregorian calendar, "
            "this date corresponds to October 10, 680 AD."
        ),
    ]

    parsed = [
        parse_temporal_response(response, Relation.BIRTH if index == 0 else Relation.DEATH)
        for index, response in enumerate(responses)
    ]

    assert [result.status for result in parsed] == ["parsed"] * 3
    assert [result.value.year for result in parsed] == [1891, 1763, 680]
    assert [result.value.month for result in parsed] == [5, 1, 10]


def test_common_era_date_ignores_parenthetical_ah_equivalent():
    parsed = parse_temporal_response(
        "Al-Ghazali died on February 16, 1111 (505 AH).",
        Relation.DEATH,
    )

    assert parsed.status == "parsed"
    assert parsed.value.year == 1111


def test_repeated_identical_value_is_one_prediction():
    response = "She was born in 1900; records confirm she was born in 1900."

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "parsed"
    assert parsed.value.year == 1900


def test_compatible_finer_date_becomes_the_prediction():
    response = "She was born in the 1900s (specifically in 1905)."

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "parsed"
    assert parsed.value.level is TemporalLevel.YEAR
    assert parsed.value.year == 1905
    assert {value.level for value in parsed.matched_values} == {
        TemporalLevel.DECADE,
        TemporalLevel.YEAR,
    }


def test_incompatible_coarse_and_fine_dates_remain_multiple():
    response = (
        "She was born in 1900, although another source says she was born "
        "on January 1, 1901."
    )

    parsed = parse_temporal_response(response, Relation.BIRTH)

    assert parsed.status == "multiple"
    assert parsed.value is None
    assert {value.year for value in parsed.matched_values} == {1900, 1901}


def test_common_hyphenated_temporal_forms_are_parsed():
    century = parse_temporal_response(
        "She was born in the 19th-century.",
        Relation.BIRTH,
    )
    modified_decade = parse_temporal_response(
        "She was born in the mid-1900s.",
        Relation.BIRTH,
    )

    assert century.status == "parsed"
    assert century.value.level is TemporalLevel.CENTURY
    assert modified_decade.status == "parsed"
    assert modified_decade.value.level is TemporalLevel.EARLYMIDDECADE


def test_idk_detection_is_exact_but_allows_markdown_wrapping():
    assert is_idk_response("**I don't know.**")
    assert is_idk_response("I do not know!")
    assert not is_idk_response("I don't know the exact day, but it was in 1900.")


def test_prediction_finer_than_gold_is_not_counted_as_correct():
    gold = TemporalValue(level=TemporalLevel.YEAR, year=1900)
    fine = TemporalValue(
        level=TemporalLevel.DAY,
        year=1900,
        month=1,
        day=1,
    )
    coarse = TemporalValue(level=TemporalLevel.DECADE, decade=1900)

    assert not correct_at_prediction_level(fine, gold)
    assert correct_at_prediction_level(coarse, gold)


def test_quadratic_weighted_kappa_is_ordinal_and_handles_degenerate_data():
    targets = [
        TemporalLevel.CENTURY,
        TemporalLevel.DECADE,
        TemporalLevel.YEAR,
        TemporalLevel.DAY,
    ]

    assert quadratic_weighted_kappa(targets, targets) == 1.0
    assert quadratic_weighted_kappa(targets, list(reversed(targets))) < 0.0
    assert quadratic_weighted_kappa(
        [TemporalLevel.YEAR],
        [TemporalLevel.YEAR],
    ) is None


def test_extended_kappa_places_idk_below_century():
    categories = [
        "IDK",
        TemporalLevel.CENTURY,
        TemporalLevel.YEAR,
        TemporalLevel.DAY,
    ]

    assert extended_quadratic_weighted_kappa(categories, categories) == 1.0
    assert extended_quadratic_weighted_kappa(
        categories,
        list(reversed(categories)),
    ) < 0.0


def test_metrics_keep_unparsed_and_unavailable_cases_in_correct_denominators():
    cases = [
        case(
            "Q1",
            Relation.BIRTH,
            TemporalValue(level=TemporalLevel.DAY, year=1900, month=2, day=3),
        ),
        case(
            "Q1",
            Relation.DEATH,
            TemporalValue(level=TemporalLevel.YEAR, year=1950),
            target_status="idk",
        ),
        case(
            "Q2",
            Relation.BIRTH,
            TemporalValue(level=TemporalLevel.YEAR, year=1800),
            target_status="unavailable",
        ),
    ]
    generations = [
        generation(cases[0].case_id, "Q1 was born in 1900."),
        generation(cases[1].case_id, "I don't know."),
        generation(cases[2].case_id, "No date is available."),
    ]

    metrics, outcomes = score_temporal_generations(cases, generations)

    assert metrics["cases"] == 3
    assert metrics["target_available_cases"] == 2
    assert metrics["target_coverage"] == 2 / 3
    assert metrics["parse_rate"] == 1 / 3
    assert metrics["truth_accuracy"] == 1 / 3
    assert metrics["target_match_accuracy"] == 1.0
    assert metrics["known_target_level_accuracy"] == 1.0
    assert metrics["quadratic_weighted_kappa_coverage"] == 1.0
    assert metrics["extended_quadratic_weighted_kappa"] == 1.0
    assert metrics["extended_quadratic_weighted_kappa_cases"] == 2
    assert metrics["extended_quadratic_weighted_kappa_coverage"] == 1.0
    assert metrics["target_overprecision_rate"] == 0.0
    assert metrics["target_underprecision_rate"] == 0.0
    assert metrics["joint_evaluable_cases"] == 2
    assert metrics["joint_target_truth_accuracy"] == 1.0
    assert metrics["prediction_distribution"] == {
        "IDK": 1,
        "UNPARSED": 1,
        "YEAR": 1,
    }
    assert outcomes[2]["target_match"] is None


def test_known_target_without_gold_is_not_put_in_joint_truth_denominator():
    item = case("Q1", Relation.DEATH, None)

    metrics, _ = score_temporal_generations(
        [item],
        [generation(item.case_id, "Q1 died in 1950.")],
    )

    assert metrics["target_match_accuracy"] == 1.0
    assert metrics["gold_available_cases"] == 0
    assert metrics["joint_evaluable_cases"] == 0
    assert metrics["joint_target_truth_accuracy"] is None


def test_truth_can_be_correct_while_joint_target_truth_is_incorrect():
    item = case(
        "Q1",
        Relation.BIRTH,
        TemporalValue(level=TemporalLevel.DAY, year=1900, month=2, day=3),
        target_level=TemporalLevel.YEAR,
    )

    metrics, outcomes = score_temporal_generations(
        [item],
        [generation(item.case_id, "Q1 was born in February 1900.")],
    )

    assert outcomes[0]["truth_correct"] is True
    assert outcomes[0]["target_match"] is False
    assert metrics["truth_accuracy"] == 1.0
    assert metrics["joint_target_truth_accuracy"] == 0.0


def test_temporal_utility_ablation_changes_only_declared_payoffs():
    gold = TemporalValue(
        level=TemporalLevel.DAY,
        year=1900,
        month=2,
        day=3,
    )
    cases = [case(f"Q{index}", Relation.BIRTH, gold) for index in range(1, 5)]
    generations = [
        generation(cases[0].case_id, "Q1 was born in 1900."),
        generation(cases[1].case_id, "Q2 was born in 1800."),
        generation(cases[2].case_id, "I don't know."),
        generation(cases[3].case_id, "No date is available."),
    ]

    metrics, outcomes = score_temporal_generations(cases, generations)

    information_reward = outcomes[0]["utility"]
    assert 0.0 < information_reward < 1.0
    assert outcomes[1]["utility"] == -information_reward
    assert [row["utility"] for row in outcomes[2:]] == [0.0, 0.0]
    assert [row["utility_uniform_wrong_penalty"] for row in outcomes] == [
        information_reward,
        -1.0,
        0.0,
        -1.0,
    ]
    assert [row["utility_accuracy"] for row in outcomes] == [1.0, 0.0, 0.0, 0.0]
    assert [row["utility_positive_informativeness"] for row in outcomes] == [
        information_reward,
        0.0,
        0.0,
        0.0,
    ]
    assert metrics["temporal_utility"] == 0.0
    assert metrics["temporal_utility_uniform_wrong_penalty"] == (
        information_reward - 2.0
    ) / 4.0
    assert metrics["temporal_utility_accuracy"] == 0.25
    assert metrics["temporal_utility_positive_informativeness"] == (
        information_reward / 4.0
    )
    assert metrics["temporal_utility_accuracy"] == metrics["truth_accuracy"]


def test_five_level_metrics_collapse_intermediate_predictions():
    item = case(
        "Q1",
        Relation.BIRTH,
        TemporalValue(level=TemporalLevel.DAY, year=1900, month=3, day=1),
        target_level=TemporalLevel.YEAR,
        temporal_taxonomy=TemporalTaxonomy.FIVE_LEVEL,
    )

    metrics, outcomes = score_temporal_generations(
        [item],
        [generation(item.case_id, "Q1 was born in spring 1900.")],
    )

    assert outcomes[0]["prediction"]["level"] == "SEASON"
    assert outcomes[0]["prediction_category"] == "YEAR"
    assert outcomes[0]["prediction_level_for_target_metrics"] == "YEAR"
    assert outcomes[0]["target_match"] is True
    assert metrics["known_target_level_accuracy"] == 1.0
    assert metrics["target_overprecision_rate"] == 0.0


def test_max_token_completions_are_reported_not_silently_dropped():
    item = case(
        "Q1",
        Relation.BIRTH,
        TemporalValue(level=TemporalLevel.YEAR, year=1900),
    )
    metrics, _ = score_temporal_generations(
        [item],
        [generation(item.case_id, "Born in 1900", "max_tokens")],
    )

    assert metrics["max_token_rate"] == 1.0


def test_paired_qid_bootstrap_is_reproducible():
    cases = [
        case(
            f"Q{index}",
            Relation.BIRTH,
            TemporalValue(level=TemporalLevel.YEAR, year=1900 + index),
        )
        for index in range(1, 5)
    ]
    base_generations = [
        generation(item.case_id, "I don't know.") for item in cases
    ]
    ft_generations = [
        generation(item.case_id, f"Born in {1900 + index}.")
        for index, item in enumerate(cases, start=1)
    ]
    _, base = score_temporal_generations(cases, base_generations)
    _, finetuned = score_temporal_generations(cases, ft_generations)

    first = paired_qid_bootstrap(
        base,
        finetuned,
        num_resamples=100,
        confidence_level=0.95,
        seed=7,
    )
    second = paired_qid_bootstrap(
        base,
        finetuned,
        num_resamples=100,
        confidence_level=0.95,
        seed=7,
    )

    assert first == second
    truth = first["metrics"]["truth_accuracy"]
    assert truth["difference"] == {
        "estimate": 1.0,
        "ci_lower": 1.0,
        "ci_upper": 1.0,
        "excludes_zero": True,
    }
    assert first["difference_definition"] == "finetuned_minus_baseline"
    assert first["num_resamples"] == 100
    assert first["seed"] == 7
