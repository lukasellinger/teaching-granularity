"""Temporal values, precision levels, projection and formatting."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum

class TemporalLevel(StrEnum):
    DAY = "DAY"
    MONTH = "MONTH"
    SEASON = "SEASON"
    EARLYMIDYEAR = "EARLYMIDYEAR"
    YEAR = "YEAR"
    EARLYMIDDECADE = "EARLYMIDDECADE"
    DECADE = "DECADE"
    EARLYMIDCENTURY = "EARLYMIDCENTURY"
    CENTURY = "CENTURY"

TEMPORAL_PRECISION = {
    TemporalLevel.CENTURY: 7.0,
    TemporalLevel.EARLYMIDCENTURY: 7.5,
    TemporalLevel.DECADE: 8.0,
    TemporalLevel.EARLYMIDDECADE: 8.5,
    TemporalLevel.YEAR: 9.0,
    TemporalLevel.SEASON: 9.5,
    TemporalLevel.EARLYMIDYEAR: 9.5,
    TemporalLevel.MONTH: 10.0,
    TemporalLevel.DAY: 11.0,
}


FIVE_LEVEL_TEMPORAL_LEVELS = (
    TemporalLevel.DAY,
    TemporalLevel.MONTH,
    TemporalLevel.YEAR,
    TemporalLevel.DECADE,
    TemporalLevel.CENTURY,
)


FIVE_LEVEL_TEMPORAL_MAP = {
    TemporalLevel.DAY: TemporalLevel.DAY,
    TemporalLevel.MONTH: TemporalLevel.MONTH,
    TemporalLevel.SEASON: TemporalLevel.YEAR,
    TemporalLevel.EARLYMIDYEAR: TemporalLevel.YEAR,
    TemporalLevel.YEAR: TemporalLevel.YEAR,
    TemporalLevel.EARLYMIDDECADE: TemporalLevel.DECADE,
    TemporalLevel.DECADE: TemporalLevel.DECADE,
    TemporalLevel.EARLYMIDCENTURY: TemporalLevel.CENTURY,
    TemporalLevel.CENTURY: TemporalLevel.CENTURY,
}


class TemporalModifier(StrEnum):
    EARLY = "early"
    MID = "mid"
    LATE = "late"


class Season(StrEnum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"


TEMPORAL_FIELDS = {
    TemporalLevel.CENTURY: ("century",),
    TemporalLevel.EARLYMIDCENTURY: ("century", "century_modifier"),
    TemporalLevel.DECADE: ("decade",),
    TemporalLevel.EARLYMIDDECADE: ("decade", "decade_modifier"),
    TemporalLevel.YEAR: ("year",),
    TemporalLevel.SEASON: ("year", "season"),
    TemporalLevel.EARLYMIDYEAR: ("year", "year_modifier"),
    TemporalLevel.MONTH: ("year", "month"),
    TemporalLevel.DAY: ("year", "month", "day"),
}


@dataclass(frozen=True)
class TemporalValue:
    level: TemporalLevel
    year: int | None = None
    month: int | None = None
    day: int | None = None
    decade: int | None = None
    century: int | None = None
    decade_modifier: TemporalModifier | None = None
    year_modifier: TemporalModifier | None = None
    century_modifier: TemporalModifier | None = None
    season: Season | None = None

    def __post_init__(self) -> None:
        if self.year is not None:
            if self.decade is None:
                object.__setattr__(self, "decade", self.year_to_decade(self.year))
            if self.century is None:
                object.__setattr__(self, "century", self.year_to_century(self.year))
            if self.decade_modifier is None:
                object.__setattr__(
                    self,
                    "decade_modifier",
                    self.year_to_decade_modifier(self.year),
                )
            if self.century_modifier is None:
                object.__setattr__(
                    self,
                    "century_modifier",
                    self.year_to_century_modifier(self.year),
                )
        elif self.decade is not None:
            if self.century is None:
                object.__setattr__(
                    self,
                    "century",
                    self.decade_to_century(self.decade),
                )
            if self.century_modifier is None:
                object.__setattr__(
                    self,
                    "century_modifier",
                    self.decade_to_century_modifier(self.decade),
                )

        if self.month is not None:
            if not 1 <= self.month <= 12:
                raise ValueError("month must be between 1 and 12")
            if self.season is None:
                object.__setattr__(self, "season", self.month_to_season(self.month))
            if self.year_modifier is None:
                object.__setattr__(
                    self,
                    "year_modifier",
                    self.month_to_year_modifier(self.month),
                )
        if self.day is not None and not 1 <= self.day <= 31:
            raise ValueError("day must be between 1 and 31")

        for field in TEMPORAL_FIELDS[self.level]:
            if getattr(self, field) is None:
                raise ValueError(f"{field} is required for {self.level.value}")

    @property
    def precision(self) -> float:
        return TEMPORAL_PRECISION[self.level]

    def to_level(self, level: TemporalLevel) -> TemporalValue:
        if TEMPORAL_PRECISION[level] > self.precision:
            raise ValueError(
                f"cannot project {self.level.value} to finer {level.value}"
            )
        fields = TEMPORAL_FIELDS[level]
        values = {field: getattr(self, field) for field in fields}
        missing = [field for field, value in values.items() if value is None]
        if missing:
            raise ValueError(f"missing fields for {level.value}: {missing}")
        return TemporalValue(level=level, **values)

    def comparison_key(self) -> tuple:
        return tuple(getattr(self, field) for field in TEMPORAL_FIELDS[self.level])

    def consistent_with(self, other: TemporalValue) -> bool:
        lower, higher = sorted((self, other), key=lambda value: value.precision)
        try:
            projected = higher.to_level(lower.level)
        except ValueError:
            return False
        return projected.comparison_key() == lower.comparison_key()

    def to_text(self) -> str:
        if self.level is TemporalLevel.DAY:
            return f"{month_name(self.month)} {self.day}, {format_year(self.year)}"
        if self.level is TemporalLevel.MONTH:
            return f"{month_name(self.month)} {format_year(self.year)}"
        if self.level is TemporalLevel.SEASON:
            return f"{self.season.value} {format_year(self.year)}"
        if self.level is TemporalLevel.EARLYMIDYEAR:
            return f"{self.year_modifier.value} {format_year(self.year)}"
        if self.level is TemporalLevel.YEAR:
            return format_year(self.year)
        if self.level is TemporalLevel.EARLYMIDDECADE:
            return f"{self.decade_modifier.value} {format_decade(self.decade)}"
        if self.level is TemporalLevel.DECADE:
            return format_decade(self.decade)
        if self.level is TemporalLevel.EARLYMIDCENTURY:
            return f"{self.century_modifier.value} {format_century(self.century)}"
        if self.level is TemporalLevel.CENTURY:
            return format_century(self.century)
        raise ValueError(f"unsupported temporal level: {self.level}")

    @staticmethod
    def year_to_decade(year: int) -> int:
        return (year // 10) * 10 if year >= 0 else -((abs(year) // 10) * 10)

    @staticmethod
    def year_to_century(year: int) -> int:
        if year > 0:
            return ((year - 1) // 100) + 1
        return -(((abs(year) - 1) // 100) + 1)

    @staticmethod
    def year_to_decade_modifier(year: int) -> TemporalModifier:
        offset = abs(year) % 10
        if offset <= 3:
            return TemporalModifier.EARLY
        if offset <= 6:
            return TemporalModifier.MID
        return TemporalModifier.LATE

    @staticmethod
    def year_to_century_modifier(year: int) -> TemporalModifier:
        offset = (abs(year) - 1) % 100
        if offset <= 33:
            return TemporalModifier.EARLY
        if offset <= 66:
            return TemporalModifier.MID
        return TemporalModifier.LATE

    @staticmethod
    def decade_to_century(decade: int) -> int | None:
        if abs(decade) % 100 == 0:
            return None
        return TemporalValue.year_to_century(decade)

    @staticmethod
    def decade_to_century_modifier(
        decade: int,
    ) -> TemporalModifier | None:
        offset = (abs(decade) // 10) % 10
        if offset == 0:
            return None
        if offset <= 2:
            return TemporalModifier.EARLY
        if offset == 3:
            return None
        if offset <= 5:
            return TemporalModifier.MID
        if offset == 6:
            return None
        return TemporalModifier.LATE

    @staticmethod
    def month_to_season(month: int) -> Season:
        if month in {12, 1, 2}:
            return Season.WINTER
        if month in {3, 4, 5}:
            return Season.SPRING
        if month in {6, 7, 8}:
            return Season.SUMMER
        return Season.AUTUMN

    @staticmethod
    def month_to_year_modifier(month: int) -> TemporalModifier:
        if month <= 4:
            return TemporalModifier.EARLY
        if month <= 8:
            return TemporalModifier.MID
        return TemporalModifier.LATE


def month_name(month: int | None) -> str:
    if month is None:
        raise ValueError("month is required")
    return (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )[month - 1]


def format_year(year: int | None) -> str:
    if year is None:
        raise ValueError("year is required")
    return f"{abs(year)} BCE" if year < 0 else str(year)


def format_decade(decade: int | None) -> str:
    if decade is None:
        raise ValueError("decade is required")
    suffix = " BCE" if decade < 0 else ""
    return f"{abs(decade)}s{suffix}"


def format_century(century: int | None) -> str:
    if century is None:
        raise ValueError("century is required")
    number = abs(century)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    era = " BCE" if century < 0 else ""
    return f"{number}{suffix} century{era}"


def to_five_level(value: TemporalValue) -> TemporalValue:
    """Project intermediate season/modifier levels onto the five-level hierarchy."""
    return value.to_level(FIVE_LEVEL_TEMPORAL_MAP[value.level])
