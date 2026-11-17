"""Deterministic temporal parsing with original-answer coordinates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .values import TemporalLevel, Season, TemporalModifier, TemporalValue


MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}
MONTH_PATTERN = "|".join(
    sorted((re.escape(month) for month in MONTHS), key=len, reverse=True)
)
ERA_PATTERN = (
    r"(?:b\.?\s*c\.?\s*e?\.?|c\.?\s*e\.?|a\.?\s*d\.?)"
)
NON_GREGORIAN_ERA_PATTERN = re.compile(
    r"\s*a\.?\s*h\.?\b",
    flags=re.IGNORECASE,
)
MODIFIER_PATTERN = r"early|mid|late"
SEASON_PATTERN = r"spring|summer|autumn|winter"


@dataclass(frozen=True)
class TemporalMatch:
    start: int
    end: int
    value: TemporalValue


@dataclass(frozen=True)
class TemporalParseIssue:
    start: int
    end: int
    status: Literal["invalid", "incomplete", "ambiguous"]
    reason: str


@dataclass(frozen=True)
class TemporalParseResult:
    matches: tuple[TemporalMatch, ...]
    issues: tuple[TemporalParseIssue, ...]

    @property
    def has_temporal_content(self) -> bool:
        return bool(self.matches or self.issues)


def parse_temporal_text(text: str, *, include_brackets: bool = False) -> TemporalParseResult:
    """Return valid values AND diagnostics for recognized date-shaped text.

    Diagnostics distinguish invalid
    or incomplete dates from absence of temporal content, without guessing a
    correction or treating a leftover year inside an invalid date as valid.
    """
    parseable = text.translate(str.maketrans({c: " " for c in "()[]{}"})) if include_brackets else text
    searchable, mapping = strip_brackets_with_mapping(parseable)
    matches = parse_temporal_matches(parseable)
    issues: list[TemporalParseIssue] = []
    textual_shapes = (
        rf"\b(?P<month>{MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b"
        rf"(?:\s*,?\s+(?P<year>\d+)\s*(?P<era>{ERA_PATTERN})?(?!\w))?",
        rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>{MONTH_PATTERN})\b"
        rf"(?:\s+(?P<year>\d+)\s*(?P<era>{ERA_PATTERN})?(?!\w))?",
    )

    def add_issue(match, status, reason):
        start, end = mapping[match.start()][0], mapping[match.end() - 1][1]
        if not any(start < item.end and end > item.start for item in issues):
            issues.append(TemporalParseIssue(start, end, status, reason))

    for pattern in textual_shapes:
        for shape in re.finditer(pattern, searchable, re.IGNORECASE):
            start, end = mapping[shape.start()][0], mapping[shape.end() - 1][1]
            # A shorter shape may be part of a valid longer form supported by
            # the parser, e.g. March 24 in 1962 or March 24 BCE (month/year).
            if any(item.start <= start and item.end >= end
                   and item.value.level in {TemporalLevel.DAY, TemporalLevel.MONTH}
                   for item in matches):
                continue
            year = shape["year"]
            if not 1 <= int(shape["day"]) <= 31:
                add_issue(shape, "invalid", "day_out_of_range")
            elif year is None:
                add_issue(shape, "incomplete", "missing_year")
            elif len(year) > 4:
                add_issue(shape, "invalid", "year_exceeds_supported_width")
    # Slash-date ambiguity/invalidity was already suppressed by the match-only
    # parser. Expose that outcome rather than confusing it with no date at all.
    slash = re.compile(r"\b(?P<first>\d{1,2})/(?P<second>\d{1,2})/(?P<year>\d+)\b")
    for shape in slash.finditer(searchable):
        start, end = mapping[shape.start()][0], mapping[shape.end() - 1][1]
        if any(item.start <= start and item.end >= end for item in matches):
            continue
        first, second = int(shape["first"]), int(shape["second"])
        invalid = (not 1 <= first <= 31 or not 1 <= second <= 31
                   or min(first, second) > 12 or len(shape["year"]) > 4)
        add_issue(shape, "invalid" if invalid else "ambiguous",
                  "invalid_numeric_date" if invalid else "ambiguous_numeric_date")
    retained = tuple(item for item in matches if not any(
        item.start < issue.end and item.end > issue.start for issue in issues))
    return TemporalParseResult(retained, tuple(sorted(issues, key=lambda item: item.start)))


def strip_brackets_with_mapping(
    text: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Remove (), [] and {} content while retaining source coordinates."""
    retained: list[tuple[str, int, int]] = []
    depths = {"(": 0, "[": 0, "{": 0}
    opening_for = {")": "(", "]": "[", "}": "{"}

    for index, char in enumerate(text):
        if char in depths:
            depths[char] += 1
            continue
        if char in opening_for:
            opening = opening_for[char]
            depths[opening] = max(0, depths[opening] - 1)
            continue
        if not any(depths.values()):
            retained.append((char, index, index + 1))

    collapsed: list[tuple[str, int, int]] = []
    for char, start, end in retained:
        if char in " \t" and collapsed and collapsed[-1][0] in " \t":
            previous, previous_start, _ = collapsed[-1]
            collapsed[-1] = (previous, previous_start, end)
        else:
            collapsed.append((char, start, end))

    normalized: list[tuple[str, int, int]] = []
    for item in collapsed:
        if item[0] in ",.;:!?":
            while normalized and normalized[-1][0].isspace():
                normalized.pop()
        normalized.append(item)

    while normalized and normalized[0][0].isspace():
        normalized.pop(0)
    while normalized and normalized[-1][0].isspace():
        normalized.pop()

    return (
        "".join(char for char, _, _ in normalized),
        [(start, end) for _, start, end in normalized],
    )


def remap_stripped_span(
    source_text: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    stripped, mapping = strip_brackets_with_mapping(source_text)
    if not 0 <= start < end <= len(stripped):
        raise ValueError(
            f"invalid stripped span ({start}, {end}) for length {len(stripped)}"
        )
    return mapping[start][0], mapping[end - 1][1]


def parse_temporal_matches(text: str) -> list[TemporalMatch]:
    """Return non-overlapping matches in original-answer coordinates."""
    stripped, mapping = strip_brackets_with_mapping(text)
    searchable = stripped
    parsed: list[tuple[int, int, TemporalValue]] = []
    occupied: list[tuple[int, int]] = []

    def add_span(start: int, end: int, value: TemporalValue) -> None:
        if any(start < other_end and end > other_start for other_start, other_end in occupied):
            return
        occupied.append((start, end))
        parsed.append((start, end, value))

    def add(match: re.Match, value: TemporalValue) -> None:
        add_span(*match.span(), value)

    timestamp = re.fullmatch(
        r"([+-]?\d+)-(\d{2})-(\d{2})t\d{2}:\d{2}:\d{2}z",
        searchable,
        flags=re.IGNORECASE,
    )
    if timestamp:
        year, month, day = map(int, timestamp.groups())
        return [
            TemporalMatch(
                start=0,
                end=len(text),
                value=TemporalValue(
                    level=TemporalLevel.DAY,
                    year=year,
                    month=month,
                    day=day,
                ),
            )
        ]

    textual_dates = (
        re.compile(
            rf"\b(?P<month>{MONTH_PATTERN})\.?\s+"
            rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?"
            rf"(?:\s*,?\s+|\s+in\s+)"
            rf"(?P<year>\d{{1,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+"
            rf"(?:of\s+)?(?P<month>{MONTH_PATTERN})\.?\s+"
            rf"(?:in\s+)?"
            rf"(?P<year>\d{{1,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
            flags=re.IGNORECASE,
        ),
    )
    for pattern in textual_dates:
        for match in pattern.finditer(searchable):
            try:
                value = TemporalValue(
                    level=TemporalLevel.DAY,
                    year=apply_era(int(match["year"]), match["era"]),
                    month=MONTHS[match["month"].lower().rstrip(".")],
                    day=int(match["day"]),
                )
            except ValueError:
                continue
            add(match, value)

    numeric_patterns = (
        re.compile(
            r"\b(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})\b"
        ),
        re.compile(
            r"\b(?P<day>\d{1,2})\.(?P<month>\d{1,2})\."
            r"(?P<year>\d{4})\b"
        ),
    )
    for pattern in numeric_patterns:
        for match in pattern.finditer(searchable):
            try:
                value = TemporalValue(
                    level=TemporalLevel.DAY,
                    year=int(match["year"]),
                    month=int(match["month"]),
                    day=int(match["day"]),
                )
            except ValueError:
                continue
            add(match, value)

    slash_date = re.compile(
        r"\b(?P<first>\d{1,2})[/-](?P<second>\d{1,2})[/-]"
        r"(?P<year>\d{1,4})\b"
    )
    for match in slash_date.finditer(searchable):
        first = int(match["first"])
        second = int(match["second"])
        if first > 12 >= second:
            day, month = first, second
        elif second > 12 >= first:
            day, month = second, first
        else:
            occupied.append(match.span())
            continue
        try:
            value = TemporalValue(
                level=TemporalLevel.DAY,
                year=int(match["year"]),
                month=month,
                day=day,
            )
        except ValueError:
            occupied.append(match.span())
            continue
        add(match, value)

    coordinated_months = re.compile(
        rf"\b(?P<first>{MONTH_PATTERN})\.?\s+or\s+"
        rf"(?P<second>{MONTH_PATTERN})\.?\s+"
        rf"(?P<year>\d{{3,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in coordinated_months.finditer(searchable):
        year = apply_era(int(match["year"]), match["era"])
        add_span(
            *match.span("first"),
            TemporalValue(
                level=TemporalLevel.MONTH,
                year=year,
                month=MONTHS[match["first"].lower().rstrip(".")],
            ),
        )
        add_span(
            match.start("second"),
            match.end(),
            TemporalValue(
                level=TemporalLevel.MONTH,
                year=year,
                month=MONTHS[match["second"].lower().rstrip(".")],
            ),
        )

    month_year = re.compile(
        rf"\b(?P<month>{MONTH_PATTERN})\.?"
        rf"(?:\s+(?:in|of(?:\s+the\s+year)?))?\s+"
        rf"(?P<year>\d{{1,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in month_year.finditer(searchable):
        raw_year = match["year"]
        if len(raw_year) < 3 and match["era"] is None:
            continue
        if has_non_gregorian_era_suffix(searchable, match.end("year")):
            continue
        add(
            match,
            TemporalValue(
                level=TemporalLevel.MONTH,
                year=apply_era(int(raw_year), match["era"]),
                month=MONTHS[match["month"].lower().rstrip(".")],
            ),
        )

    season_year = re.compile(
        rf"\b(?P<season>{SEASON_PATTERN})"
        rf"(?:\s+(?:in|of)(?:\s+the\s+year)?)?\s+"
        rf"(?P<year>\d{{1,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in season_year.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.SEASON,
                year=apply_era(int(match["year"]), match["era"]),
                season=Season(match["season"].lower()),
            ),
        )

    modified_century = re.compile(
        rf"\b(?P<modifier>{MODIFIER_PATTERN})"
        rf"(?:-|(?:\s+in)?\s+(?:the\s+)?)"
        rf"(?P<century>\d{{1,2}})(?:st|nd|rd|th)\s+century\s*"
        rf"(?P<era>{ERA_PATTERN})?(?!\w)",
        flags=re.IGNORECASE,
    )
    for match in modified_century.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.EARLYMIDCENTURY,
                century=apply_era(int(match["century"]), match["era"]),
                century_modifier=TemporalModifier(match["modifier"].lower()),
            ),
        )

    century = re.compile(
        rf"\b(?P<century>\d{{1,2}})(?:st|nd|rd|th)\s+century\s*"
        rf"(?P<era>{ERA_PATTERN})?(?!\w)",
        flags=re.IGNORECASE,
    )
    for match in century.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.CENTURY,
                century=apply_era(int(match["century"]), match["era"]),
            ),
        )

    modified_decade = re.compile(
        rf"\b(?P<modifier>{MODIFIER_PATTERN})(?:\s+in)?\s+(?:the\s+)?"
        rf"(?P<decade>\d{{3,4}})s\s*(?P<era>{ERA_PATTERN})?(?!\w)",
        flags=re.IGNORECASE,
    )
    for match in modified_decade.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.EARLYMIDDECADE,
                decade=apply_era(int(match["decade"]), match["era"]),
                decade_modifier=TemporalModifier(match["modifier"].lower()),
            ),
        )

    decade = re.compile(
        rf"\b(?P<decade>\d{{3,4}})s\s*(?P<era>{ERA_PATTERN})?(?!\w)",
        flags=re.IGNORECASE,
    )
    for match in decade.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.DECADE,
                decade=apply_era(int(match["decade"]), match["era"]),
            ),
        )

    modified_year = re.compile(
        rf"\b(?P<modifier>{MODIFIER_PATTERN})"
        rf"(?:\s+in(?:\s+the\s+year)?)?\s+"
        rf"(?P<year>\d{{3,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in modified_year.finditer(searchable):
        add(
            match,
            TemporalValue(
                level=TemporalLevel.EARLYMIDYEAR,
                year=apply_era(int(match["year"]), match["era"]),
                year_modifier=TemporalModifier(match["modifier"].lower()),
            ),
        )

    year_range = re.compile(
        rf"\b(?P<first>\d{{3,4}})\s*[-–—]\s*"
        rf"(?P<second>\d{{3,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in year_range.finditer(searchable):
        era = match["era"]
        add_span(
            *match.span("first"),
            TemporalValue(
                level=TemporalLevel.YEAR,
                year=apply_era(int(match["first"]), era),
            ),
        )
        add_span(
            match.start("second"),
            match.end(),
            TemporalValue(
                level=TemporalLevel.YEAR,
                year=apply_era(int(match["second"]), era),
            ),
        )

    bare_year = re.compile(
        rf"\b(?P<year>\d{{1,4}})\s*(?P<era>{ERA_PATTERN})?(?![\w-])",
        flags=re.IGNORECASE,
    )
    for match in bare_year.finditer(searchable):
        raw_year = match["year"]
        era = match["era"]
        if looks_like_list_item(searchable, match.start(), raw_year):
            continue
        if looks_like_duration(searchable, match.start(), match.end("year")):
            continue
        if has_non_gregorian_era_suffix(searchable, match.end("year")):
            continue
        inherited_era = trailing_range_era(
            searchable,
            match.end("year"),
        )
        if len(raw_year) < 3 and era is None:
            continue
        add(
            match,
            TemporalValue(
                level=TemporalLevel.YEAR,
                year=apply_era(int(raw_year), era or inherited_era),
            ),
        )

    result = []
    for start, end, value in sorted(parsed, key=lambda item: item[0]):
        original_start = mapping[start][0]
        original_end = mapping[end - 1][1]
        result.append(TemporalMatch(original_start, original_end, value))
    return result


def apply_era(year: int, era: str | None) -> int:
    if era and era.lower().replace(".", "").replace(" ", "").startswith("bc"):
        return -year
    return year


def has_non_gregorian_era_suffix(text: str, year_end: int) -> bool:
    """Return whether a numeric year is explicitly in the Islamic AH era."""
    return NON_GREGORIAN_ERA_PATTERN.match(text[year_end:]) is not None


def trailing_range_era(text: str, year_end: int) -> str | None:
    """Return an era suffix shared by both endpoints of a range.

    In ``600 to 576 BCE``, the trailing era applies to the complete range.
    Both endpoints must therefore be negative rather than silently dropping
    the first or misreading it as the positive year 600.
    """
    match = re.match(
        rf"\s*(?:-|–|—|to|and|or)\s*\d{{1,4}}\s*"
        rf"(?P<era>{ERA_PATTERN})(?!\w)",
        text[year_end:],
        flags=re.IGNORECASE,
    )
    return match["era"] if match is not None else None


def looks_like_list_item(text: str, start: int, raw_year: str) -> bool:
    if int(raw_year) >= 1000:
        return False
    line_start = text.rfind("\n", 0, start) + 1
    return bool(re.fullmatch(r"\s*", text[line_start:start])) and text[
        start + len(raw_year):
    ].startswith(".")


def looks_like_duration(text: str, start: int, end: int) -> bool:
    """Reject quantities such as ``400 years after`` as calendar years."""
    suffix = text[end:]
    if re.match(
        r"\s+years?\s+(?:after|before|old)\b",
        suffix,
        flags=re.IGNORECASE,
    ):
        return True
    prefix = text[max(0, start - 24):start]
    return re.search(
        r"\b(?:age[sd]?|aged)\s+(?:approximately\s+|about\s+)?$",
        prefix,
        flags=re.IGNORECASE,
    ) is not None
