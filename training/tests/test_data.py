from __future__ import annotations
import math
import pytest
torch = pytest.importorskip("torch")
from training.schemas import AbstentionLevel, Candidate, ContrastiveGroup, IDK_ANSWER, LocationLevel, Modality, Relation, Split, TemporalLevel, TrainingExample
from training.data import WeightedRealization, load_training_examples, rank_balance_profile, semantic_case_key, training_summary, weight_realizations
from training.encoding import RealizationCollator, RealizationDataset, encode_realization, rotating_negative_indices, selected_negative_indices


def example(
    case_id: str,
    qid: str,
    positive: str,
    *,
    rankable: bool,
    prompt_template_id: int,
) -> TrainingExample:
    groups = []
    if rankable:
        start = positive.index("1900")
        alternative = positive.replace("1900", "the 1900s")
        alternative_start = alternative.index("the 1900s")
        groups = [
            ContrastiveGroup(
                modality=Modality.TEMPORAL,
                target_level=TemporalLevel.YEAR,
                target_index=0,
                candidates=[
                    Candidate(
                        answer=positive,
                        score_start=start,
                        score_end=start + 4,
                        level=TemporalLevel.YEAR,
                    ),
                    Candidate(
                        answer=alternative,
                        score_start=alternative_start,
                        score_end=alternative_start + len("the 1900s"),
                        level=TemporalLevel.DECADE,
                    ),
                ],
            )
        ]
    return TrainingExample(
        case_id=case_id,
        qid=qid,
        split=Split.TRAIN,
        source_model="model",
        relation=Relation.BIRTH,
        prompt_template_id=prompt_template_id,
        prompt=f"When was {qid} born?",
        source_answer=positive,
        positive_answer=positive,
        is_idk=False,
        primary_target_level=TemporalLevel.YEAR,
        contrastive_groups=groups,
    )

def location_example(
    case_id: str,
    qid: str,
    target_level: LocationLevel,
) -> TrainingExample:
    positive = f"{qid} was born in France."
    target_start = positive.index("France")
    alternative = positive.replace("France", "Europe")
    alternative_start = alternative.index("Europe")
    return TrainingExample(
        case_id=case_id,
        qid=qid,
        split=Split.TRAIN,
        source_model="model",
        relation=Relation.BIRTHPLACE,
        prompt_template_id=0,
        prompt=f"Where was {qid} born?",
        source_answer=positive,
        positive_answer=positive,
        is_idk=False,
        primary_target_level=target_level,
        contrastive_groups=[
            ContrastiveGroup(
                modality=Modality.LOCATION,
                target_level=target_level,
                target_index=0,
                candidates=[
                    Candidate(
                        answer=positive,
                        score_start=target_start,
                        score_end=target_start + len("France"),
                        level=target_level,
                    ),
                    Candidate(
                        answer=alternative,
                        score_start=alternative_start,
                        score_end=alternative_start + len("Europe"),
                        level=(
                            LocationLevel.CONTINENT
                            if target_level is not LocationLevel.CONTINENT
                            else LocationLevel.COUNTRY
                        ),
                    ),
                ],
            )
        ],
    )

def idk_example(case_id: str, qid: str) -> TrainingExample:
    return TrainingExample(
        case_id=case_id,
        qid=qid,
        split=Split.TRAIN,
        source_model="model",
        relation=Relation.BIRTH,
        prompt_template_id=0,
        prompt=f"When was {qid} born?",
        source_answer=IDK_ANSWER,
        positive_answer=IDK_ANSWER,
        is_idk=True,
    )

def temporal_example_with_level(
    case_id: str,
    qid: str,
    target_level: TemporalLevel,
) -> TrainingExample:
    item = example(
        case_id,
        qid,
        f"{qid} was born in 1900.",
        rankable=True,
        prompt_template_id=0,
    )
    group = item.contrastive_groups[0]
    candidates = list(group.candidates)
    candidates[0] = candidates[0].model_copy(update={"level": target_level})
    return item.model_copy(
        update={
            "primary_target_level": target_level,
            "contrastive_groups": [
                group.model_copy(
                    update={
                        "target_level": target_level,
                        "candidates": candidates,
                    }
                )
            ],
        }
    )

def test_all_realizations_are_retained_and_each_case_has_equal_sft_weight():
    rows = [
        example("a:0", "A", "A was born in 1900.", rankable=True, prompt_template_id=0),
        example(
            "a:1",
            "A",
            "In 1900, A was born.",
            rankable=False,
            prompt_template_id=7,
        ),
        example("b:0", "B", "B was born in 1900.", rankable=True, prompt_template_id=0),
    ]

    weighted = weight_realizations(rows, "sft_contrastive")

    assert len(weighted) == 3
    assert [row.sft_case_weight for row in weighted] == [0.75, 0.75, 1.5]
    assert [row.contrastive_case_weight for row in weighted] == [1.5, 0.0, 1.5]
    assert semantic_case_key(rows[0]) == semantic_case_key(rows[1])
    assert math.isclose(sum(row.sft_case_weight for row in weighted), 3.0)
    assert math.isclose(sum(row.contrastive_case_weight for row in weighted), 3.0)

def test_contrastive_only_drops_sft_only_rows_before_computing_weights():
    rows = [
        example("a:0", "A", "A was born in 1900.", rankable=True, prompt_template_id=0),
        example(
            "a:1",
            "A",
            "In 1900, A was born.",
            rankable=False,
            prompt_template_id=1,
        ),
        example("b:0", "B", "B was born in 1900.", rankable=True, prompt_template_id=0),
    ]

    weighted = weight_realizations(rows, "contrastive")

    assert [row.example.case_id for row in weighted] == ["a:0", "b:0"]
    assert [row.contrastive_case_weight for row in weighted] == [1.0, 1.0]

def test_idk_sft_weight_changes_relative_case_mass_and_keeps_mean_one():
    known = example(
        "known:0", "KNOWN", "KNOWN was born in 1900.",
        rankable=True, prompt_template_id=0,
    )
    idk = idk_example("idk:0", "IDK")

    full = weight_realizations([known, idk], "sft_contrastive")
    half = weight_realizations(
        [known, idk], "sft_contrastive", idk_sft_weight=0.5
    )
    none = weight_realizations(
        [known, idk], "sft_contrastive", idk_sft_weight=0.0
    )

    assert [row.sft_case_weight for row in full] == [1.0, 1.0]
    assert [row.sft_case_weight for row in half] == pytest.approx([4 / 3, 2 / 3])
    assert sum(row.sft_case_weight for row in half) == pytest.approx(2.0)
    assert [row.contrastive_case_weight for row in half] == [2.0, 0.0]
    assert [row.example.case_id for row in none] == ["known:0"]
    assert none[0].sft_case_weight == 1.0
    assert none[0].contrastive_case_weight == 1.0

@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_idk_sft_weight_must_be_between_zero_and_one(weight: float):
    with pytest.raises(ValueError, match="between zero and one"):
        weight_realizations(
            [idk_example("idk:0", "IDK")],
            "sft_contrastive",
            idk_sft_weight=weight,
        )

def test_hierarchical_rank_balance_corrects_modality_and_level_mass():
    examples = [
        *[
            temporal_example_with_level(
                f"year:{index}", f"Y{index}", TemporalLevel.YEAR
            )
            for index in range(4)
        ],
        temporal_example_with_level("season:0", "S0", TemporalLevel.SEASON),
        *[
            location_example(f"country:{index}", f"C{index}", LocationLevel.COUNTRY)
            for index in range(2)
        ],
        location_example("city:0", "L0", LocationLevel.CITY),
    ]
    rows = weight_realizations(examples, "sft_contrastive")

    profile = rank_balance_profile(rows, "hierarchical-sqrt", cap=4.0)

    assert profile.level_factors["SEASON"] > profile.level_factors["YEAR"]
    assert profile.level_factors["CITY"] > profile.level_factors["COUNTRY"]
    assert (
        profile.raw_modality_factors["location"]
        > profile.raw_modality_factors["temporal"]
    )
    before_ratio = (
        profile.modality_base_mass["temporal"] / profile.modality_base_mass["location"]
    )
    after_ratio = (
        profile.modality_balanced_mass["temporal"]
        / profile.modality_balanced_mass["location"]
    )
    assert abs(math.log(after_ratio)) < abs(math.log(before_ratio))
    assert max(profile.level_factors.values()) <= 4.0
    assert sum(profile.modality_balanced_mass.values()) == pytest.approx(
        sum(profile.modality_base_mass.values())
    )

def test_rank_balance_mass_counts_semantic_cases_not_realizations():
    examples = [
        example("a:0", "A", "A was born in 1900.", rankable=True, prompt_template_id=0),
        example("a:1", "A", "In 1900 A was born.", rankable=True, prompt_template_id=1),
        example(
            "a:2", "A", "The birth year was 1900.", rankable=True, prompt_template_id=2
        ),
        temporal_example_with_level("b:0", "B", TemporalLevel.SEASON),
    ]
    rows = weight_realizations(examples, "sft_contrastive")

    profile = rank_balance_profile(rows, "none")

    assert profile.level_case_counts == {"SEASON": 1, "YEAR": 1}
    assert profile.level_base_mass["SEASON"] == pytest.approx(
        profile.level_base_mass["YEAR"]
    )

def test_negative_rotation_covers_seeded_permutation_before_repeating():
    windows = [
        rotating_negative_indices(
            list(range(8)),
            limit=2,
            seed=42,
            epoch=epoch,
            case_id="case",
            modality=Modality.TEMPORAL,
        )
        for epoch in range(5)
    ]

    assert len(set(index for window in windows[:4] for index in window)) == 8
    assert windows[4] == windows[0]

def ordinal_temporal_group(
    target: TemporalLevel,
    negative_levels: list[TemporalLevel | AbstentionLevel],
) -> ContrastiveGroup:
    candidates = [
        Candidate(
            answer=f"target {target.value}",
            score_start=7,
            score_end=7 + len(target.value),
            level=target,
        )
    ]
    for level in negative_levels:
        answer = IDK_ANSWER if level is AbstentionLevel.IDK else f"negative {level.value}"
        start = 0 if level is AbstentionLevel.IDK else len("negative ")
        candidates.append(
            Candidate(
                answer=answer,
                score_start=start,
                score_end=len(answer),
                level=level,
            )
        )
    return ContrastiveGroup(
        modality=Modality.TEMPORAL,
        target_level=target,
        target_index=0,
        candidates=candidates,
    )

def test_ordinal_selection_keeps_both_boundaries_and_idk():
    group = ordinal_temporal_group(
        TemporalLevel.YEAR,
        [
            TemporalLevel.DAY,
            TemporalLevel.MONTH,
            TemporalLevel.DECADE,
            TemporalLevel.CENTURY,
            AbstentionLevel.IDK,
        ],
    )

    selected = selected_negative_indices(
        group,
        limit=3,
        seed=42,
        epoch=0,
        case_id="case",
    )

    assert {group.candidates[index].level for index in selected} == {
        TemporalLevel.MONTH,
        TemporalLevel.DECADE,
        AbstentionLevel.IDK,
    }

def test_ordinal_selection_fills_a_missing_boundary_and_never_exceeds_three():
    group = ordinal_temporal_group(
        TemporalLevel.DAY,
        [
            TemporalLevel.MONTH,
            TemporalLevel.YEAR,
            TemporalLevel.DECADE,
            TemporalLevel.CENTURY,
            AbstentionLevel.IDK,
        ],
    )

    selected = selected_negative_indices(
        group,
        limit=4,
        seed=42,
        epoch=0,
        case_id="case",
    )
    levels = {group.candidates[index].level for index in selected}

    assert len(selected) == 3
    assert TemporalLevel.MONTH in levels
    assert AbstentionLevel.IDK in levels

def test_ordinal_selection_retains_groups_with_fewer_than_three_negatives():
    group = ordinal_temporal_group(
        TemporalLevel.YEAR,
        [TemporalLevel.DECADE, AbstentionLevel.IDK],
    )

    selected = selected_negative_indices(
        group,
        limit=3,
        seed=42,
        epoch=0,
        case_id="case",
    )

    assert selected == [1, 2]

def test_ordinal_selection_uses_its_own_budget_not_the_generic_cap():
    group = ordinal_temporal_group(
        TemporalLevel.YEAR,
        [
            TemporalLevel.MONTH,
            TemporalLevel.DECADE,
            AbstentionLevel.IDK,
        ],
    )

    selected = selected_negative_indices(
        group,
        limit=2,
        seed=42,
        epoch=0,
        case_id="case",
    )

    assert {group.candidates[index].level for index in selected} == {
        TemporalLevel.MONTH,
        TemporalLevel.DECADE,
        AbstentionLevel.IDK,
    }

class CharacterTokenizer:
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
    ):
        assert not tokenize
        assert not enable_thinking
        text = "".join(
            f"<{message['role']}>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return text

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_attention_mask,
        return_offsets_mapping,
        truncation,
    ):
        assert not add_special_tokens
        assert return_attention_mask
        assert not truncation
        result = {
            "input_ids": [ord(character) for character in text],
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def pad(
        self,
        features,
        *,
        padding,
        pad_to_multiple_of,
        return_tensors,
    ):
        assert padding
        assert return_tensors == "pt"
        width = max(len(ids) for ids in features["input_ids"])
        if pad_to_multiple_of:
            width = math.ceil(width / pad_to_multiple_of) * pad_to_multiple_of
        ids = []
        masks = []
        for input_ids, attention_mask in zip(
            features["input_ids"], features["attention_mask"]
        ):
            amount = width - len(input_ids)
            ids.append(input_ids + [self.pad_token_id] * amount)
            masks.append(attention_mask + [0] * amount)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }

def test_encoding_supervises_full_answer_and_limits_training_negatives():
    positive = "A was born in 1900."
    item = example("a:0", "A", positive, rankable=True, prompt_template_id=0)
    candidate = Candidate(
        answer=positive.replace("1900", "the 20th century"),
        score_start=positive.index("1900"),
        score_end=positive.index("1900") + len("the 20th century"),
        level=TemporalLevel.CENTURY,
    )
    expanded_group = item.contrastive_groups[0].model_copy(
        update={"candidates": [*item.contrastive_groups[0].candidates, candidate]}
    )
    item = item.model_copy(update={"contrastive_groups": [expanded_group]})
    row = WeightedRealization(item, 1.0, 1.0)

    encoded = encode_realization(
        row,
        CharacterTokenizer(),
        seed=42,
        epoch=0,
        max_negatives_per_group=1,
        max_length=200,
        system_prompt=None,
        rank_balance_factors={"YEAR": 2.5},
    )

    assert len(encoded["candidate_sequence_indices"]) == 2
    assert encoded["group_balance_weights"] == [2.5]
    positive_sequence = encoded["sequences"][encoded["positive_sequence_index"]]
    assert sum(positive_sequence["sft_mask"]) >= len(positive)
    assert sum(encoded["candidate_is_target"]) == 1

def test_encoding_wraps_every_training_question_in_the_user_prompt_template():
    item = example(
        "a:0", "A", "A was born in 1900.", rankable=False, prompt_template_id=0
    )
    encoded = encode_realization(
        WeightedRealization(item, 1.0, 0.0),
        CharacterTokenizer(),
        seed=42,
        epoch=0,
        max_negatives_per_group=0,
        max_length=300,
        system_prompt=None,
        user_prompt_template="Instruction. Question: {question} Answer:",
    )

    sequence = encoded["sequences"][encoded["positive_sequence_index"]]
    rendered = "".join(chr(token) for token in sequence["input_ids"])
    assert "<user>Instruction. Question: When was A born? Answer:" in rendered
    assert rendered.endswith("<assistant>A was born in 1900.")

def test_collator_deduplicates_identical_sequences_across_realizations():
    row = WeightedRealization(
        example("a:0", "A", "A was born in 1900.", rankable=True, prompt_template_id=0),
        0.5,
        1.5,
    )
    feature = encode_realization(
        row,
        CharacterTokenizer(),
        seed=42,
        epoch=0,
        max_negatives_per_group=None,
        max_length=200,
        system_prompt=None,
    )

    batch = RealizationCollator(CharacterTokenizer())([feature, feature])

    assert batch["input_ids"].shape[0] == len(feature["sequences"])
    assert batch["positive_sequence_indices"].tolist() == [0, 0]
    assert batch["group_realization_indices"].tolist() == [0, 1]
    assert batch["sft_case_weights"].tolist() == [0.5, 0.5]
    assert batch["contrastive_case_weights"].tolist() == [1.5, 1.5]
    assert batch["group_balance_weights"].tolist() == [1.0, 1.0]
