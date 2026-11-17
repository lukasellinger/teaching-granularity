"""Location values, projection and within-answer consensus."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
import re

class LocationLevel(StrEnum):
    PLACE = "PLACE"
    SUBCITY = "SUBCITY"
    CITY = "CITY"
    REGION = "REGION"
    COUNTRY = "COUNTRY"
    CONTINENT = "CONTINENT"

LOCATION_PRECISION = {
    LocationLevel.CONTINENT: 1,
    LocationLevel.COUNTRY: 2,
    LocationLevel.REGION: 3,
    LocationLevel.CITY: 4,
    LocationLevel.SUBCITY: 5,
    LocationLevel.PLACE: 6,
}

LOCATION_FIELDS = {
    LocationLevel.PLACE: "place",
    LocationLevel.SUBCITY: "subcity",
    LocationLevel.CITY: "city",
    LocationLevel.REGION: "region",
    LocationLevel.COUNTRY: "country",
    LocationLevel.CONTINENT: "continent",
}

LOCATION_CONTEXT_FIELDS = {
    LocationLevel.PLACE: ("place", "subcity", "city", "region", "country", "continent"),
    LocationLevel.SUBCITY: ("subcity", "city", "region", "country", "continent"),
    LocationLevel.CITY: ("city", "region", "country", "continent"),
    LocationLevel.REGION: ("region", "country", "continent"),
    LocationLevel.COUNTRY: ("country", "continent"),
    LocationLevel.CONTINENT: ("continent",),
}

@dataclass(frozen=True)
class LocationValue:
    level: LocationLevel
    needs_disambiguation: bool = False
    place: tuple[str, ...] | None = None
    subcity: tuple[str, ...] | None = None
    city: str | None = None
    region: tuple[str, ...] | None = None
    country: str | None = None
    continent: str | None = None

    def __post_init__(self) -> None:
        # Accept legacy scalar places without interpreting commas as separators.
        if isinstance(self.place, str):
            object.__setattr__(self, "place", (self.place,))
        elif isinstance(self.place, list):
            object.__setattr__(self, "place", tuple(self.place))
        field = LOCATION_FIELDS[self.level]
        if not getattr(self, field):
            raise ValueError(f"{field} is required for {self.level.value}")

    @classmethod
    def from_dict(cls, data: dict) -> LocationValue:
        subcity = data.get("subcity")
        if isinstance(subcity, str):
            subcity = (subcity,)
        elif isinstance(subcity, list):
            subcity = tuple(subcity)
        region = data.get("region")
        if isinstance(region, str):
            region = (region,)
        elif isinstance(region, list):
            region = tuple(region)
        return cls(
            level=LocationLevel(data["level"]),
            needs_disambiguation=bool(data.get("needs_disambiguation", False)),
            place=data.get("place"),
            subcity=subcity,
            city=data.get("city"),
            region=region,
            country=data.get("country"),
            continent=data.get("continent"),
        )

    @property
    def precision(self) -> int:
        return LOCATION_PRECISION[self.level]

    def to_level(self, level: LocationLevel) -> LocationValue:
        if LOCATION_PRECISION[level] > self.precision:
            raise ValueError(
                f"cannot project {self.level.value} to finer {level.value}"
            )
        fields = LOCATION_CONTEXT_FIELDS[level]
        values = {field: getattr(self, field) for field in fields}
        required = LOCATION_FIELDS[level]
        if not values.get(required):
            raise ValueError(f"missing {required} for {level.value}")
        return LocationValue(
            level=level,
            needs_disambiguation=self.needs_disambiguation,
            **values,
        )

    def comparison_key(self) -> str | tuple[str, ...]:
        value = getattr(self, LOCATION_FIELDS[self.level])
        if isinstance(value, tuple):
            return value
        assert isinstance(value, str)
        return value

    def consistent_with(self, other: LocationValue) -> bool:
        lower, higher = sorted((self, other), key=lambda value: value.precision)
        try:
            projected = higher.to_level(lower.level)
        except ValueError:
            return False
        if lower.level in {LocationLevel.PLACE, LocationLevel.REGION, LocationLevel.SUBCITY}:
            field = LOCATION_FIELDS[lower.level]
            return not set(getattr(lower, field) or ()).isdisjoint(getattr(projected, field) or ())
        return projected.comparison_key() == lower.comparison_key()

    def to_text(self) -> str:
        value = getattr(self, LOCATION_FIELDS[self.level])
        if isinstance(value, tuple):
            value = value[-1]
        if not isinstance(value, str):
            raise ValueError(f"cannot render {self.level.value}")
        parts = [value]

        # A venue without its containing location is rarely a usable answer
        # (for example, many hospitals share a name).  PLACE remains the
        # semantic target level; the city/subcity is disambiguating context,
        # not a second candidate.
        if self.level is LocationLevel.PLACE:
            broader = (
                self.subcity[-1] if self.subcity
                else self.city
                or (self.region[-1] if self.region else None)
                or self.country
                or self.continent
            )
            if broader:
                parts.append(broader)

        # The normalizer explicitly records when a bare label needs context.
        # Country is stable and avoids choosing arbitrarily among several
        # administrative-region labels.
        if self.needs_disambiguation and self.level in {
            LocationLevel.PLACE,
            LocationLevel.SUBCITY,
            LocationLevel.CITY,
            LocationLevel.REGION,
        }:
            context = self.city if self.level is LocationLevel.SUBCITY else self.country or self.continent
            if context and not re.search(
                rf"(?<!\w){re.escape(context)}(?!\w)", value, re.IGNORECASE
            ):
                parts.append(context)

        rendered = []
        for part in parts:
            if part.casefold() not in {existing.casefold() for existing in rendered}:
                rendered.append(part)
        return ", ".join(rendered)

def location_consensus(
    values: list[LocationValue | None],
    min_share: float,
) -> LocationValue | None:
    required = consensus_count(len(values), min_share)
    levels = sorted(LOCATION_PRECISION, key=LOCATION_PRECISION.get, reverse=True)
    for level in levels:
        groups: dict[str, list[LocationValue]] = defaultdict(list)
        for value in values:
            if value is None:
                continue
            try:
                projected = value.to_level(level)
            except ValueError:
                continue
            keys = (
                getattr(projected, LOCATION_FIELDS[level]) or ()
                if level in {LocationLevel.PLACE, LocationLevel.REGION, LocationLevel.SUBCITY}
                else (str(projected.comparison_key()),)
            )
            for key in dict.fromkeys(keys):
                groups[key].append(projected)

        eligible = [
            (len(group), key, group)
            for key, group in groups.items()
            if len(group) >= required
        ]
        if eligible:
            _, key, group = sorted(eligible, key=lambda item: (-item[0], item[1]))[0]
            return merge_location_group(level, key, group)
    return None

def consensus_count(total: int, min_share: float) -> int:
    if total <= 0:
        raise ValueError("consensus requires at least one sample")
    if not 0 < min_share <= 1:
        raise ValueError("min_share must be in (0, 1]")
    return ceil(total * min_share)

def common_scalar(values) -> str | None:
    unique = {value for value in values if value is not None}
    return next(iter(unique)) if len(unique) == 1 else None

def merge_location_group(
    level: LocationLevel,
    key: str,
    group: list[LocationValue],
) -> LocationValue:
    """Build a consensus value without inheriting context from one sample."""
    required_field = LOCATION_FIELDS[level]
    context: dict[str, str | tuple[str, ...] | None] = {}
    for field in LOCATION_CONTEXT_FIELDS[level]:
        if field == required_field:
            if field in {"place", "region", "subcity"}:
                aliases = [set(getattr(value, field) or ()) for value in group]
                shared = set.intersection(*aliases) if aliases else set()
                if key not in shared:
                    raise ValueError(
                        f"consensus key {key!r} is not shared by its supporting group"
                    )
                context[field] = tuple(sorted(shared))
            else:
                context[field] = key
        elif field in {"place", "region", "subcity"}:
            aliases = [set(getattr(value, field) or ()) for value in group]
            shared = set.intersection(*aliases) if aliases else set()
            context[field] = tuple(sorted(shared)) or None
        else:
            context[field] = common_scalar(
                getattr(value, field) for value in group
            )
    return LocationValue(
        level=level,
        needs_disambiguation=any(
            value.needs_disambiguation for value in group
        ),
        **context,
    )
