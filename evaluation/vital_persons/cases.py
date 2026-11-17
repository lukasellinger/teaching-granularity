"""Canonical person/event cases and Wikidata gold values."""
from __future__ import annotations
from shared.constants import QUESTIONS
from shared.constants import MODEL_NAMES as SOURCE_MODELS
import re
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal
from parsing.temporal import TemporalLevel, TemporalValue
from parsing.temporal.values import TEMPORAL_PRECISION, FIVE_LEVEL_TEMPORAL_MAP, FIVE_LEVEL_TEMPORAL_LEVELS
from parsing.location.normalization.values import LocationValue, LocationLevel
from shared.geography import project_location

class Split(StrEnum):
    TRAIN = 'train'
    DEV = 'dev'
    TEST = 'test'

class Modality(StrEnum):
    TEMPORAL = 'temporal'
    LOCATION = 'location'

class TemporalTaxonomy(StrEnum):
    FINE = 'fine'
    FIVE_LEVEL = 'five-level'

class Relation(StrEnum):
    BIRTH = 'birth'
    DEATH = 'death'
    BIRTHPLACE = 'birthplace'
    DEATHPLACE = 'deathplace'

TEMPORAL_RELATIONS = (Relation.BIRTH, Relation.DEATH)
LOCATION_RELATIONS = (Relation.BIRTHPLACE, Relation.DEATHPLACE)
REPO = Path(__file__).resolve().parents[2]
VITAL = REPO / 'data_preparation/vital_persons'


def temporal_levels_for_taxonomy(
    taxonomy: TemporalTaxonomy,
) -> tuple[TemporalLevel, ...]:
    """Return the only temporal classes scored by a taxonomy."""
    if taxonomy is TemporalTaxonomy.FIVE_LEVEL:
        return FIVE_LEVEL_TEMPORAL_LEVELS
    return tuple(TemporalLevel)


def temporal_level_for_taxonomy(
    level: TemporalLevel,
    taxonomy: TemporalTaxonomy,
) -> TemporalLevel:
    """Collapse an extracted level into the configured target taxonomy."""
    if taxonomy is TemporalTaxonomy.FIVE_LEVEL:
        return FIVE_LEVEL_TEMPORAL_MAP[level]
    return level


TargetStatus = Literal["known", "idk", "unavailable"]


_GOLD_PRECISION_TO_LEVEL = {
    7: TemporalLevel.CENTURY,
    8: TemporalLevel.DECADE,
    9: TemporalLevel.YEAR,
    10: TemporalLevel.MONTH,
    11: TemporalLevel.DAY,
}


_TIMESTAMP = re.compile(
    r"^(?P<year>[+-]?\d+)-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T\d{2}:\d{2}:\d{2}Z$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class TemporalEvaluationCase:
    """A semantic evaluation case, independent of answer realizations."""

    case_id: str
    source_model: str
    split: Split
    qid: str
    name: str
    relation: Relation
    prompt: str
    gold: TemporalValue | None
    target_status: TargetStatus
    target_level: TemporalLevel | None
    temporal_taxonomy: TemporalTaxonomy = TemporalTaxonomy.FINE


@dataclass(frozen=True)
class LocationEvaluationCase:
    """A semantic location case with model-derived and external targets."""

    case_id: str
    source_model: str
    split: Split
    qid: str
    name: str
    relation: Relation
    prompt: str
    gold: LocationValue | None
    target_status: TargetStatus
    target_level: LocationLevel | None
    location_taxonomy: str = "full"


def temporal_value_dict(value: TemporalValue) -> dict[str, object]:
    """Return the minimal stable JSON representation of a temporal value."""
    return {
        "level": value.level.value,
        "year": value.year,
        "month": value.month,
        "day": value.day,
        "decade": value.decade,
        "century": value.century,
        "decade_modifier": (
            value.decade_modifier.value if value.decade_modifier else None
        ),
        "year_modifier": (
            value.year_modifier.value if value.year_modifier else None
        ),
        "century_modifier": (
            value.century_modifier.value if value.century_modifier else None
        ),
        "season": value.season.value if value.season else None,
    }


def location_value_dict(value: LocationValue) -> dict[str, object]:
    """Return the stable JSON representation of a normalized location."""
    return {
        "level": value.level.value,
        "needs_disambiguation": value.needs_disambiguation,
        "place": list(value.place) if value.place is not None else None,
        "subcity": list(value.subcity) if value.subcity is not None else None,
        "city": value.city,
        "region": list(value.region) if value.region is not None else None,
        "country": value.country,
        "continent": value.continent,
    }


def parse_gold_temporal(timestamp: object, precision: object) -> TemporalValue:
    """Parse a Wikidata timestamp and respect its declared precision."""
    if not isinstance(timestamp, str):
        raise ValueError(f"gold timestamp must be a string, got {timestamp!r}")
    match = _TIMESTAMP.fullmatch(timestamp)
    if match is None:
        raise ValueError(f"unsupported gold timestamp: {timestamp!r}")
    try:
        numeric_precision = float(precision)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid gold precision: {precision!r}") from error
    if not numeric_precision.is_integer():
        raise ValueError(f"gold precision must be integral: {precision!r}")
    level = _GOLD_PRECISION_TO_LEVEL.get(int(numeric_precision))
    if level is None:
        raise ValueError(f"unsupported gold precision: {precision!r}")
    # Wikidata may encode unknown month/day as 00 at coarser precision.
    # Only validate the components actually supported by the gold evidence.
    full = TemporalValue(
        level=TemporalLevel.DAY if level is TemporalLevel.DAY else TemporalLevel.MONTH if level is TemporalLevel.MONTH else TemporalLevel.YEAR,
        year=int(match["year"]),
        month=int(match["month"]) if level in {TemporalLevel.DAY, TemporalLevel.MONTH} else None,
        day=int(match["day"]) if level is TemporalLevel.DAY else None,
    )
    return full.to_level(level)


def parse_gold_location(row: dict[str, object], prefix: str) -> LocationValue | None:
    """Build the most-specific supported location hierarchy from metadata."""
    scalar_fields = {}
    for field in ("city", "country", "continent"):
        value = row.get(f"{prefix}_{field}")
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"invalid gold {prefix}_{field}: {value!r}")
        scalar_fields[field] = value.strip() if isinstance(value, str) else None

    raw_region = row.get(f"{prefix}_region")
    if raw_region is None:
        region = None
    elif isinstance(raw_region, list) and all(
        isinstance(value, str) and value.strip() for value in raw_region
    ):
        region = tuple(dict.fromkeys(value.strip() for value in raw_region)) or None
    else:
        raise ValueError(f"invalid gold {prefix}_region: {raw_region!r}")

    raw_disambiguation = row.get(f"{prefix}_needs_disambiguation", False)
    if not isinstance(raw_disambiguation, bool):
        raise ValueError(
            f"invalid gold {prefix}_needs_disambiguation: "
            f"{raw_disambiguation!r}"
        )

    raw_subcity = row.get(f"{prefix}_subcity")
    if raw_subcity is None:
        subcity = None
    elif isinstance(raw_subcity, list) and all(
        isinstance(value, str) and value.strip() for value in raw_subcity
    ):
        subcity = tuple(dict.fromkeys(value.strip() for value in raw_subcity)) or None
    else:
        raise ValueError(f"invalid gold {prefix}_subcity: {raw_subcity!r}")
    raw_place = row.get(f"{prefix}_place")
    if isinstance(raw_place, str) and raw_place.strip():
        place = (raw_place.strip(),)
    elif raw_place is None:
        place = None
    elif isinstance(raw_place, list) and all(isinstance(v, str) and v.strip() for v in raw_place):
        place = tuple(dict.fromkeys(v.strip() for v in raw_place)) or None
    else:
        raise ValueError(f"invalid gold {prefix}_place: {raw_place!r}")
    values = {**scalar_fields, "place": place, "subcity": subcity, "region": region}
    for level, field in (
        (LocationLevel.PLACE, "place"),
        (LocationLevel.SUBCITY, "subcity"),
        (LocationLevel.CITY, "city"),
        (LocationLevel.REGION, "region"),
        (LocationLevel.COUNTRY, "country"),
        (LocationLevel.CONTINENT, "continent"),
    ):
        if values[field]:
            return LocationValue(
                level=level,
                needs_disambiguation=raw_disambiguation,
                **values,
            )
    return None


def temporal_case_summary(cases: list[TemporalEvaluationCase]) -> dict[str, object]:
    summary = _case_summary(cases, TEMPORAL_RELATIONS)
    taxonomies = {case.temporal_taxonomy for case in cases}
    if len(taxonomies) != 1:
        raise ValueError("temporal evaluation cases mix taxonomies")
    summary["temporal_taxonomy"] = next(iter(taxonomies)).value
    return summary


def location_case_summary(cases: list[LocationEvaluationCase]) -> dict[str, object]:
    summary = _case_summary(cases, LOCATION_RELATIONS)
    taxonomies = {case.location_taxonomy for case in cases}
    if len(taxonomies) != 1:
        raise ValueError("Location evaluation cases mix taxonomies")
    summary["location_taxonomy"] = next(iter(taxonomies))
    return summary


def _case_summary(cases, relations) -> dict[str, object]:
    known = sum(case.target_status == "known" for case in cases)
    idk = sum(case.target_status == "idk" for case in cases)
    unavailable = sum(case.target_status == "unavailable" for case in cases)
    gold = sum(case.gold is not None for case in cases)
    return {
        "cases": len(cases),
        "qids": len({case.qid for case in cases}),
        "by_relation": {
            relation.value: sum(case.relation is relation for case in cases)
            for relation in relations
        },
        "targets": {
            "known": known,
            "idk": idk,
            "unavailable": unavailable,
            "coverage": (known + idk) / len(cases),
        },
        "gold": {
            "available": gold,
            "unavailable": len(cases) - gold,
            "coverage": gold / len(cases),
        },
    }

