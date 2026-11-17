"""Original case/source weighting and primary-only prepared-data loading."""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence
from pydantic import ValidationError
from .schemas import TrainingExample, GranolaTrainingExample, GranolaLevel, GranolaRelation, granola_level_key

TrainingObjective = Literal["sft", "contrastive", "sft_contrastive"]
RankBalanceMode = Literal["none", "hierarchical-sqrt"]
PreparedExample = TrainingExample | GranolaTrainingExample
CaseKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class WeightedRealization:
    """One retained answer and its normalized case-level loss weights."""

    example: PreparedExample
    sft_case_weight: float
    contrastive_case_weight: float

@dataclass(frozen=True)
class RankBalanceProfile:
    """Train-derived target-level corrections for the ranking objective."""

    mode: RankBalanceMode
    cap: float
    raw_modality_factors: dict[str, float]
    raw_within_modality_level_factors: dict[str, float]
    level_factors: dict[str, float]
    level_case_counts: dict[str, int]
    level_base_mass: dict[str, float]
    modality_base_mass: dict[str, float]
    modality_balanced_mass: dict[str, float]

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "cap": self.cap,
            "raw_modality_factors": self.raw_modality_factors,
            "raw_within_modality_level_factors": (
                self.raw_within_modality_level_factors
            ),
            "level_factors": self.level_factors,
            "level_case_counts": self.level_case_counts,
            "level_base_mass": self.level_base_mass,
            "modality_base_mass": self.modality_base_mass,
            "modality_balanced_mass": self.modality_balanced_mass,
        }

def example_source(example: PreparedExample) -> str:
    return getattr(example, "data_source", "vital_persons")

def group_level_key(example: PreparedExample, level: str) -> str:
    return (
        granola_level_key(example.relation.value, level)
        if example_source(example) == "granola"
        else level
    )

def semantic_case_key(example: PreparedExample) -> CaseKey:
    """Return the unit that must have equal total training influence."""
    return (
        example_source(example),
        example.source_model,
        example.qid,
        example.relation.value,
    )

def load_training_examples(path: Path) -> list[PreparedExample]:
    """Load a prepared JSONL split and fail with a precise row location."""
    examples: list[PreparedExample] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"cannot read training split {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            try:
                payload = json.loads(line)
                schema = (
                    GranolaTrainingExample
                    if payload.get("data_source") == "granola"
                    else TrainingExample
                )
                examples.append(schema.model_validate(payload))
            except (ValidationError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError(
                    f"invalid training example at {path}:{line_number}: {error}"
                ) from error
    if not examples:
        raise ValueError(f"training split is empty: {path}")
    return examples

def weight_realizations(
    examples: Sequence[PreparedExample],
    objective: TrainingObjective,
    *,
    granola_loss_fraction: float | None = None,
    idk_sft_weight: float = 1.0,
) -> list[WeightedRealization]:
    """Keep all eligible rows while making every semantic case count once.

    The fixed ``N / (C * R_i)`` normalization gives weights with dataset mean
    one. SFT counts every retained realization. Contrastive weighting counts
    only realizations with at least one rankable group, so an SFT-only row does
    not dilute the ranking loss of another realization from the same case.
    ``idk_sft_weight`` changes IDK case mass relative to known case mass and
    then renormalizes the SFT weights to keep their dataset mean at one. A zero
    weight removes IDK rows entirely so they do not consume optimizer batches.
    """
    if objective not in {"sft", "contrastive", "sft_contrastive"}:
        raise ValueError(f"unsupported training objective: {objective!r}")
    if not 0 <= idk_sft_weight <= 1:
        raise ValueError("IDK SFT weight must be between zero and one")
    selected = [
        example
        for example in examples
        if (objective != "contrastive" or example.contrastive_groups)
        and not (idk_sft_weight == 0 and example.is_idk)
    ]
    if not selected:
        raise ValueError(f"no examples are eligible for objective {objective!r}")

    total = len(selected)
    sft_counts = Counter(semantic_case_key(example) for example in selected)
    sft_case_count = len(sft_counts)
    rankable = [example for example in selected if example.contrastive_groups]
    rank_counts = Counter(semantic_case_key(example) for example in rankable)
    rank_case_count = len(rank_counts)

    weighted = []
    for example in selected:
        key = semantic_case_key(example)
        sft_weight = total / (sft_case_count * sft_counts[key])
        rank_count = rank_counts.get(key, 0)
        rank_weight = (
            total / (rank_case_count * rank_count)
            if example.contrastive_groups and rank_count and rank_case_count
            else 0.0
        )
        weighted.append(
            WeightedRealization(
                example=example,
                sft_case_weight=sft_weight,
                contrastive_case_weight=rank_weight,
            )
        )

    if idk_sft_weight != 1:
        adjusted_sft_total = sum(
            row.sft_case_weight
            * (idk_sft_weight if row.example.is_idk else 1.0)
            for row in weighted
        )
        if adjusted_sft_total <= 0:
            raise ValueError("IDK weighting removed all SFT loss mass")
        sft_scale = total / adjusted_sft_total
        weighted = [
            WeightedRealization(
                example=row.example,
                sft_case_weight=row.sft_case_weight
                * (idk_sft_weight if row.example.is_idk else 1.0)
                * sft_scale,
                contrastive_case_weight=row.contrastive_case_weight,
            )
            for row in weighted
        ]

    _validate_case_weights(weighted)
    if granola_loss_fraction is not None:
        if not 0 < granola_loss_fraction < 1:
            raise ValueError("Granola loss fraction must be between zero and one")
        sources = {example_source(row.example) for row in weighted}
        if sources != {"vital_persons", "granola"}:
            raise ValueError(
                "source mixing requires both vital_persons and granola rows"
            )
        targets = {
            "vital_persons": 1 - granola_loss_fraction,
            "granola": granola_loss_fraction,
        }
        sft_mass = Counter()
        rank_mass = Counter()
        for row in weighted:
            source = example_source(row.example)
            sft_mass[source] += row.sft_case_weight
            rank_mass[source] += row.contrastive_case_weight
        if objective != "sft" and any(rank_mass[source] <= 0 for source in sources):
            raise ValueError(
                "each source must have rankable rows for contrastive source mixing"
            )
        available_rank_share = sum(
            targets[source] for source in sources if rank_mass[source] > 0
        )
        weighted = [
            WeightedRealization(
                example=row.example,
                sft_case_weight=row.sft_case_weight
                * total
                * targets[example_source(row.example)]
                / sft_mass[example_source(row.example)],
                contrastive_case_weight=(
                    row.contrastive_case_weight
                    * total
                    * targets[example_source(row.example)]
                    / (rank_mass[example_source(row.example)] * available_rank_share)
                    if rank_mass[example_source(row.example)] > 0
                    else 0.0
                ),
            )
            for row in weighted
        ]
        _validate_case_weights(weighted)
    return weighted

def _validate_case_weights(rows: Sequence[WeightedRealization]) -> None:
    total = len(rows)
    sft_total = sum(row.sft_case_weight for row in rows)
    if abs(sft_total - total) > 1e-8 * total:
        raise AssertionError("SFT case weights do not have mean one")
    rank_rows = [row for row in rows if row.contrastive_case_weight]
    if rank_rows:
        rank_total = sum(row.contrastive_case_weight for row in rows)
        if abs(rank_total - total) > 1e-8 * total:
            raise AssertionError("contrastive case weights do not have mean one")

def _capped_unit_mean_factors(
    raw_factors: Mapping[str, float],
    masses: Mapping[str, float],
    cap: float,
) -> dict[str, float]:
    """Scale positive factors to weighted mean one without exceeding ``cap``."""
    if cap < 1:
        raise ValueError("rank-balance cap must be at least one")
    total_mass = sum(masses.values())
    if total_mass <= 0:
        return {level: 1.0 for level in raw_factors}

    def weighted_mean(scale: float) -> float:
        return (
            sum(
                masses[level] * min(cap, scale * factor)
                for level, factor in raw_factors.items()
            )
            / total_mass
        )

    lower = 0.0
    upper = 1.0
    while weighted_mean(upper) < 1.0:
        upper *= 2
    for _ in range(80):
        middle = (lower + upper) / 2
        if weighted_mean(middle) < 1.0:
            lower = middle
        else:
            upper = middle
    scale = upper
    return {level: min(cap, scale * factor) for level, factor in raw_factors.items()}

def rank_balance_profile(
    rows: Sequence[WeightedRealization],
    mode: RankBalanceMode,
    *,
    cap: float = 4.0,
) -> RankBalanceProfile:
    """Derive ranking weights from train-case mass, never realization counts.

    Existing inverse-realization case weights define the base mass of a group.
    ``hierarchical-sqrt`` tempers imbalance twice: between modalities and then
    between levels inside each modality. The combined correction is capped and
    normalized to retain mean-one contrastive mass.
    """
    if mode not in {"none", "hierarchical-sqrt"}:
        raise ValueError(f"unsupported rank-balance mode: {mode!r}")
    if cap < 1:
        raise ValueError("rank-balance cap must be at least one")

    level_mass: Counter[str] = Counter()
    modality_mass: Counter[str] = Counter()
    level_modalities: dict[str, str] = {}
    level_sources: dict[str, str] = {}
    level_cases: defaultdict[str, set[CaseKey]] = defaultdict(set)
    for row in rows:
        groups = row.example.contrastive_groups
        if not groups or row.contrastive_case_weight <= 0:
            continue
        base_group_mass = row.contrastive_case_weight / len(groups)
        case_key = semantic_case_key(row.example)
        for group in groups:
            level = group_level_key(row.example, group.target_level.value)
            source = example_source(row.example)
            modality = (
                f"granola:{row.example.relation.value}"
                if source == "granola"
                else group.modality.value
            )
            level_sources[level] = source
            previous = level_modalities.setdefault(level, modality)
            if previous != modality:
                raise ValueError(
                    f"target level {level!r} occurs in multiple modalities"
                )
            level_mass[level] += base_group_mass
            modality_mass[modality] += base_group_mass
            level_cases[level].add(case_key)

    if not level_mass:
        return RankBalanceProfile(
            mode=mode,
            cap=cap,
            raw_modality_factors={},
            raw_within_modality_level_factors={},
            level_factors={},
            level_case_counts={},
            level_base_mass={},
            modality_base_mass={},
            modality_balanced_mass={},
        )

    if mode == "none":
        modality_factors = {modality: 1.0 for modality in modality_mass}
        within_modality_factors = {level: 1.0 for level in level_mass}
        factors = {level: 1.0 for level in level_mass}
    else:
        levels_by_modality: defaultdict[str, list[str]] = defaultdict(list)
        for level, modality in level_modalities.items():
            levels_by_modality[modality].append(level)
        modality_factors = {}
        within_modality_factors = {}
        raw_factors = {}
        factors = {}
        # Correct within each source, so balancing cannot undo the requested
        # auxiliary loss share. Granola's relation taxonomies stay separate.
        for source in sorted(set(level_sources.values())):
            source_levels = [
                level for level in level_mass if level_sources[level] == source
            ]
            source_modalities = {level_modalities[level] for level in source_levels}
            average_modality_mass = sum(
                modality_mass[modality] for modality in source_modalities
            ) / len(source_modalities)
            for modality in source_modalities:
                levels = levels_by_modality[modality]
                average_level_mass = modality_mass[modality] / len(levels)
                modality_factor = math.sqrt(
                    average_modality_mass / modality_mass[modality]
                )
                modality_factors[modality] = modality_factor
                for level in levels:
                    level_factor = math.sqrt(average_level_mass / level_mass[level])
                    within_modality_factors[level] = level_factor
                    raw_factors[level] = modality_factor * level_factor
            factors.update(
                _capped_unit_mean_factors(
                    {level: raw_factors[level] for level in source_levels},
                    {level: level_mass[level] for level in source_levels},
                    cap,
                )
            )

    balanced_modality_mass: Counter[str] = Counter()
    for level, mass in level_mass.items():
        balanced_modality_mass[level_modalities[level]] += mass * factors[level]
    return RankBalanceProfile(
        mode=mode,
        cap=cap,
        raw_modality_factors=dict(sorted(modality_factors.items())),
        raw_within_modality_level_factors=dict(sorted(within_modality_factors.items())),
        level_factors=dict(sorted(factors.items())),
        level_case_counts={
            level: len(level_cases[level]) for level in sorted(level_mass)
        },
        level_base_mass={level: level_mass[level] for level in sorted(level_mass)},
        modality_base_mass=dict(sorted(modality_mass.items())),
        modality_balanced_mass=dict(sorted(balanced_modality_mass.items())),
    )

LEVEL_NAMES = (
    "DAY",
    "MONTH",
    "SEASON",
    "EARLYMIDYEAR",
    "YEAR",
    "EARLYMIDDECADE",
    "DECADE",
    "EARLYMIDCENTURY",
    "CENTURY",
    "PLACE",
    "CITY",
    "REGION",
    "COUNTRY",
    "CONTINENT",
) + tuple(
    granola_level_key(relation.value, level.value)
    for relation in GranolaRelation
    for level in GranolaLevel
)

_LEVEL_IDS = {level: index for index, level in enumerate(LEVEL_NAMES)}

def _level_id(level: str) -> int:
    try:
        return _LEVEL_IDS[level]
    except KeyError as error:
        raise ValueError(f"unknown granularity level: {level!r}") from error

def training_summary(rows: Sequence[WeightedRealization]) -> dict[str, object]:
    """Return auditable counts for run metadata and console output."""
    cases = Counter(semantic_case_key(row.example) for row in rows)
    idk_rows = [row for row in rows if row.example.is_idk]
    idk_cases = {semantic_case_key(row.example) for row in idk_rows}
    rankable_rows = [row for row in rows if row.example.contrastive_groups]
    rankable_cases = {semantic_case_key(row.example) for row in rankable_rows}
    source_sft_mass = Counter()
    source_rank_mass = Counter()
    for row in rows:
        source_sft_mass[example_source(row.example)] += row.sft_case_weight
        source_rank_mass[example_source(row.example)] += row.contrastive_case_weight
    return {
        "realizations": len(rows),
        "semantic_cases": len(cases),
        "idk_realizations": len(idk_rows),
        "idk_semantic_cases": len(idk_cases),
        "idk_sft_weight_mass": sum(row.sft_case_weight for row in idk_rows),
        "known_sft_weight_mass": sum(
            row.sft_case_weight for row in rows if not row.example.is_idk
        ),
        "rankable_realizations": len(rankable_rows),
        "rankable_cases": len(rankable_cases),
        "mean_realizations_per_case": len(rows) / len(cases),
        "maximum_realizations_per_case": max(cases.values()),
        "case_definition": ["data_source", "source_model", "qid", "relation"],
        "weighting": "N/(C*R_i)",
        "realizations_by_source": dict(
            Counter(example_source(row.example) for row in rows)
        ),
        "sft_weight_mass_by_source": dict(source_sft_mass),
        "contrastive_weight_mass_by_source": dict(source_rank_mass),
    }
