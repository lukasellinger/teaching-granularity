"""Deterministic temporal surface edits with score offsets."""
from __future__ import annotations
from dataclasses import dataclass
import re
from parsing.temporal import TemporalMatch, TemporalValue, TemporalLevel
from shared.text import _SENTENCE_END, _is_nonterminal_period

@dataclass(frozen=True)
class PlannedEdit:
    start: int
    end: int
    replacement: str
    score_start: int
    score_end: int

def plan_temporal_edit(
    answer: str,
    match: TemporalMatch,
    value: TemporalValue,
) -> PlannedEdit:
    prefix = answer[:match.start]
    prep_match = re.search(
        r"(?<!\w)(?P<leading_marker>\*\*|__|\*|_)?"
        r"(?P<prep>(?:in|on|at|from|since|until|by)\s+"
        r"(?P<inner_marker>\*\*|__|\*|_)?"
        r"(?:around|about|circa|approximately|roughly|c\.)"
        r"(?:\s+the\s+year)?|"
        r"(?:around|about|circa|approximately|roughly)\s+"
        r"(?:\*\*|__|\*|_)?c\.|"
        r"(?:around|about|circa|approximately|roughly)"
        r"(?:\s+the\s+year)?|c\.|during\s+the\s+year|"
        r"in\s+the\s+year|of\s+the\s+year|in\s+the|on\s+the|"
        r"from\s+the\s+year|since\s+the\s+year|"
        r"until\s+the\s+year|by\s+the\s+year|"
        r"from\s+the|since\s+the|until\s+the|by\s+the|"
        r"during\s+the|of\s+the|in|on|at|during|of|from|since|until|by)"
        r"\s+(?P<trailing_marker>\*\*|__|\*|_)?$",
        prefix,
        flags=re.IGNORECASE,
    )
    old_prep = (
        re.sub(r"\*\*|__|\*|_", "", prep_match.group("prep")).lower()
        if prep_match
        else ""
    )
    marker_match = (
        re.search(r"\*\*|__|\*|_", prep_match.group(0))
        if prep_match
        else None
    )
    marker = marker_match.group(0) if marker_match else ""
    prep = adapted_temporal_preposition(old_prep, value.level)
    marker_precedes_prep = bool(
        prep_match
        and (
            prep_match.group("leading_marker")
            or prep_match.group("inner_marker")
        )
    )
    if marker_precedes_prep:
        replacement = f"{marker}{prep + ' ' if prep else ''}{value.to_text()}"
    else:
        replacement = f"{prep + ' ' if prep else ''}{marker}{value.to_text()}"
    start = prep_match.start() if prep_match else match.start
    replacement, score_end = replacement_with_terminal_punctuation(
        answer,
        start,
        match.end,
        replacement,
    )
    return PlannedEdit(
        start=start,
        end=match.end,
        replacement=replacement,
        score_start=0,
        score_end=score_end,
    )

def replacement_with_terminal_punctuation(
    answer: str,
    start: int,
    end: int,
    replacement: str,
) -> tuple[str, int]:
    """Normalize sentence punctuation consumed by an extracted span."""
    if not re.fullmatch(r"[\"'”’)}\]*_]*", answer[end:]):
        return replacement, len(replacement)

    punctuation_index = end - 1
    while (
        punctuation_index >= start
        and answer[punctuation_index] in "\"'”’)}]*_"
    ):
        punctuation_index -= 1
    if punctuation_index < start:
        return replacement, len(replacement)

    punctuation = answer[punctuation_index]
    if punctuation not in ".!?":
        return replacement, len(replacement)
    if not replacement.endswith(punctuation):
        replacement += punctuation
    return replacement, len(replacement) - 1

def adapted_temporal_preposition(old: str, level: TemporalLevel) -> str:
    uncertainty_words = (
        "around",
        "about",
        "circa",
        "approximately",
        "roughly",
    )
    uncertainty = next((word for word in uncertainty_words if word in old), None)
    if uncertainty is None and re.search(r"(?:^|\s)c\.(?:\s|$)", old):
        uncertainty = "circa"
    directional = next(
        (word for word in ("from", "since", "until", "by") if old.startswith(word)),
        None,
    )
    coarse = {
        TemporalLevel.DECADE,
        TemporalLevel.EARLYMIDDECADE,
        TemporalLevel.CENTURY,
        TemporalLevel.EARLYMIDCENTURY,
    }
    if level in coarse:
        if uncertainty is not None:
            phrase = f"{uncertainty} the"
            return f"{directional} {phrase}" if directional else phrase
        if directional is not None:
            return f"{directional} the"
        if "during" in old:
            return "during the"
        if old.startswith("of"):
            return "of the"
        return "in the"
    if level is TemporalLevel.DAY:
        return "on"
    if old == "in the year" and level is TemporalLevel.YEAR:
        return old
    if old == "of the year" and level is TemporalLevel.YEAR:
        return old
    if (
        directional is not None
        and old.endswith("the year")
        and level is TemporalLevel.YEAR
    ):
        return old
    if uncertainty is not None:
        return (
            f"{directional} {uncertainty}"
            if directional is not None
            else uncertainty
        )
    if directional is not None:
        return directional
    if old.startswith("during"):
        return "during"
    if old.startswith("of"):
        return "of"
    return "in"

def first_complete_sentence_end(text: str, minimum_end: int) -> int | None:
    """Return the first safe boundary ending at ``minimum_end`` or later."""
    if not 0 <= minimum_end <= len(text):
        raise ValueError("minimum_end must be inside the text")

    for match in _SENTENCE_END.finditer(text):
        if match.end() < minimum_end:
            continue
        if _is_nonterminal_period(text, match):
            continue
        return match.end()
    return None
