"""Select birth/death values from generated answers, without scoring them."""
from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Literal
from .parser import parse_temporal_text
from .values import TemporalValue
from .relations import Relation, has_relation_anchor_before, relation_match_is_relevant
from .sentences import sentence_bounds_containing

ParseStatus = Literal["parsed", "idk", "unparsed", "multiple"]


_TERMINAL_PUNCTUATION = re.compile(r"[.!?]+$")


_MARKDOWN_WRAPPER = re.compile(r"^[*_`#\s]+|[*_`#\s]+$")


_BRACKET_CHARACTERS = str.maketrans({character: " " for character in "()[]{}"})


_HYPHENATED_TEMPORAL_FORM = re.compile(
    r"(\b(?:early|mid|late)|\b\d{1,2}(?:st|nd|rd|th))"
    r"[-‐‑‒–—](?=(?:\d{3,4}s?\b|century\b))",
    flags=re.IGNORECASE,
)


_TEMPORAL_ALTERNATIVE_CUE = re.compile(
    r"\b(?:although|alternatively|another source|could be|estimate[sd]?|"
    r"however|instead|or|possibly|range[sd]?|between|some sources?|"
    r"suggest(?:s|ed)?)\b",
    flags=re.IGNORECASE,
)


_REFERENTIAL_TEMPORAL_ALTERNATIVE = re.compile(
    r"\b(?:another source|could be|different (?:date|year)|later (?:date|year)|"
    r"range[sd]?|between|some sources?|suggest(?:s|ed)?|"
    r"alternative (?:date|year))\b",
    flags=re.IGNORECASE,
)


_GREGORIAN_CALENDAR_CUE = re.compile(
    r"\b(?:Gregorian|modern calendar)\b",
    flags=re.IGNORECASE,
)


_CALENDAR_EQUIVALENCE_CUE = re.compile(
    r"\b(?:corresponds?|equivalent|modern calendar|Gregorian|Julian)\b",
    flags=re.IGNORECASE,
)


_STANDALONE_ANSWER_PREFIX = re.compile(
    r"^\s*(?:(?:correct|incorrect)\s+)?(?:aggregated\s+)?answer\s*:\s*",
    flags=re.IGNORECASE,
)


_STANDALONE_EXPLANATION_BOUNDARY = re.compile(
    r"\n+\s*(?:explanation|reasoning)\s*:",
    flags=re.IGNORECASE,
)


_STANDALONE_OTHER_EVENT = re.compile(
    r"\b(?:became|served|graduated|married|divorced|published|released|"
    r"founded|established|joined|moved|relocated|settled|appointed|elected|"
    r"retired|enlisted|worked|started|began|ended|won|received|awarded|"
    r"announced|reported|created|reigned|ascended|abdicated|lasted)\b",
    flags=re.IGNORECASE,
)


_STANDALONE_PROSE_VERB = re.compile(
    r"\b(?:is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"fought|lived|ruled|occurred|happened)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedTemporalResponse:
    status: ParseStatus
    value: TemporalValue | None
    matched_values: tuple[TemporalValue, ...]


def normalize_idk_response(response: object) -> str:
    if not isinstance(response, str):
        return ""
    normalized = " ".join(
        response.strip().casefold().replace("’", "'").split()
    )
    normalized = _MARKDOWN_WRAPPER.sub("", normalized)
    return _TERMINAL_PUNCTUATION.sub("", normalized).strip()


def is_idk_response(response: object) -> bool:
    """Recognize only a complete abstention, never a hedged concrete answer."""
    return normalize_idk_response(response) in {
        "idk",
        "i don't know",
        "i do not know",
    }


def parse_temporal_response(
    response: str,
    relation,
) -> ParsedTemporalResponse:
    """Select one compatible date anchored to the requested event.

    Repeating the same event at increasing precision does not create multiple
    answers: ``1428`` followed by ``June 1428`` expresses one value at month
    precision.  Incompatible retained values remain non-evaluable rather than
    being hidden by an arbitrary preference for the finer one.
    """
    relation = Relation(relation)
    if is_idk_response(response):
        return ParsedTemporalResponse("idk", None, ())
    # Preparation deliberately removes bracketed qualifications. Final
    # evaluation must instead observe them: a finer date in parentheses still
    # makes the generated answer finer. Replacing only bracket characters keeps
    # source offsets stable while exposing their contents to the shared parser.
    parseable_response = response.translate(_BRACKET_CHARACTERS)
    parseable_response = _HYPHENATED_TEMPORAL_FORM.sub(
        lambda match: f"{match[1]} ",
        parseable_response,
    )
    raw_matches = parse_temporal_text(parseable_response).matches
    relevant_matches = [
        match
        for match in raw_matches
        if relation_match_is_relevant(
            response,
            match.start,
            match.end,
            relation,
        )
        and not _is_attributive_hyphenated_period(response, match)
    ]
    relevant_matches = _include_standalone_answer_matches(
        response,
        raw_matches,
        relevant_matches,
    )
    relevant_matches = _include_referential_alternatives(
        response,
        raw_matches,
        relevant_matches,
    )
    calendar_canonical = _gregorian_calendar_matches(
        response,
        raw_matches,
        relation,
    )
    relevant_matches = _scope_temporal_matches(
        response,
        relevant_matches,
        calendar_canonical,
    )
    relevant = [match.value for match in relevant_matches]
    unique = tuple(dict.fromkeys(relevant))
    if not unique:
        return ParsedTemporalResponse("unparsed", None, ())
    if calendar_canonical:
        canonical = calendar_canonical[-1].value
        later_alternatives = [
            value
            for match, value in zip(relevant_matches, relevant)
            if match.start > calendar_canonical[-1].end
            and not canonical.consistent_with(value)
        ]
        if not later_alternatives:
            return ParsedTemporalResponse("parsed", canonical, unique)
    if len(unique) > 1:
        finest = max(unique, key=lambda value: value.precision)
        if all(finest.consistent_with(value) for value in unique):
            return ParsedTemporalResponse("parsed", finest, unique)
        return ParsedTemporalResponse("multiple", None, unique)
    return ParsedTemporalResponse("parsed", unique[0], unique)


def _include_standalone_answer_matches(
    response: str,
    raw_matches,
    relevant_matches,
):
    """Accept temporal fragments emitted as direct aggregation answers.

    DRAG asks for an answer rather than a full sentence, so outputs such as
    ``around 1401`` contain no repeated birth/death predicate. This path is
    intentionally limited to a short answer segment and rejects prose about a
    different event. Normal biographical answers continue through the stricter
    relation-anchor checks.
    """
    if relevant_matches:
        return relevant_matches
    boundary = _STANDALONE_EXPLANATION_BOUNDARY.search(response)
    answer_end = boundary.start() if boundary is not None else len(response)
    segment = response[:answer_end]
    explicit_answer_prefix = _STANDALONE_ANSWER_PREFIX.match(segment)
    content = _STANDALONE_ANSWER_PREFIX.sub("", segment).strip()
    if not content or len(re.findall(r"\b\w+\b", content)) > 24:
        return relevant_matches
    if _STANDALONE_OTHER_EVENT.search(content):
        return relevant_matches
    if (
        explicit_answer_prefix is None
        and _STANDALONE_PROSE_VERB.search(content)
    ):
        return relevant_matches
    candidates = [
        match
        for match in raw_matches
        if match.start < answer_end
        and not _is_attributive_hyphenated_period(response, match)
    ]
    if not candidates:
        return relevant_matches
    return sorted(
        {*relevant_matches, *candidates},
        key=lambda match: (match.start, match.end),
    )


def _is_attributive_hyphenated_period(response: str, match) -> bool:
    """Reject adjectives such as ``16th-century poet`` as event dates."""
    source = response[match.start:match.end]
    if not re.search(r"\d(?:st|nd|rd|th)-century$", source, re.IGNORECASE):
        return False
    suffix = response[match.end:]
    return re.match(r"\s+[A-Za-z]", suffix) is not None


def _include_referential_alternatives(
    response: str,
    raw_matches,
    relevant_matches,
):
    """Include explicit later alternatives that refer back to the answer.

    A follow-up sentence such as ``Some sources suggest a later date, such as
    576 or 577`` need not repeat the word ``birth``. Its dates are nevertheless
    alternatives to the preceding requested-event claim. Ordinary biography
    sentences do not satisfy this stricter referential cue.
    """
    if not relevant_matches:
        return relevant_matches
    first_start = min(match.start for match in relevant_matches)
    included = set(relevant_matches)
    for match in raw_matches:
        if match in included or match.start <= first_start:
            continue
        sentence_start, sentence_end = sentence_bounds_containing(
            response,
            match.start,
        )
        sentence = response[sentence_start:sentence_end]
        if _REFERENTIAL_TEMPORAL_ALTERNATIVE.search(sentence):
            included.add(match)
    return sorted(included, key=lambda match: (match.start, match.end))


def _scope_temporal_matches(
    response: str,
    relevant_matches,
    calendar_canonical,
):
    """Keep the primary requested-event claim and meaningful refinements.

    Generated answers normally state the requested date first and then add
    biographical dates. Later compatible refinements and explicit alternatives
    remain visible; unrelated incompatible narrative dates do not inherit the
    earlier birth/death anchor.
    """
    combined = sorted(
        {*(relevant_matches), *(calendar_canonical)},
        key=lambda match: (match.start, match.end),
    )
    if not combined:
        return []
    _, primary_end = sentence_bounds_containing(response, combined[0].start)
    selected = [match for match in combined if match.start < primary_end]
    finest = max(selected, key=lambda match: match.value.precision).value

    for match in combined[len(selected):]:
        sentence_start, sentence_end = sentence_bounds_containing(
            response,
            match.start,
        )
        sentence = response[sentence_start:sentence_end]
        if match in calendar_canonical:
            selected.append(match)
            finest = match.value
            continue
        if finest.consistent_with(match.value):
            selected.append(match)
            if match.value.precision > finest.precision:
                finest = match.value
            continue
        if _TEMPORAL_ALTERNATIVE_CUE.search(sentence):
            selected.append(match)
    return selected


def _gregorian_calendar_matches(response: str, raw_matches, relation):
    """Return explicit Gregorian equivalents of a requested event date."""
    by_sentence: dict[tuple[int, int], list] = {}
    for match in raw_matches:
        bounds = sentence_bounds_containing(response, match.start)
        by_sentence.setdefault(bounds, []).append(match)

    canonical = []
    for (start, end), matches in by_sentence.items():
        sentence = response[start:end]
        if not _GREGORIAN_CALENDAR_CUE.search(sentence):
            continue
        if not _CALENDAR_EQUIVALENCE_CUE.search(sentence):
            continue
        if not has_relation_anchor_before(response, end, relation):
            continue
        canonical.append(max(matches, key=lambda match: match.start))
    return sorted(canonical, key=lambda match: match.start)
