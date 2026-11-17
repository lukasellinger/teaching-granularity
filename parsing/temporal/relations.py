"""Check whether temporal mentions answer a birth/death question."""
from __future__ import annotations
from enum import StrEnum
import re
from .sentences import sentence_bounds_containing


class Relation(StrEnum):
    BIRTH = "birth"
    DEATH = "death"


_BIRTH_ANCHOR = re.compile(
    r"\b(?:born|birthdate|birthday|birthplace|"
    r"birth (?:date|year|place)|(?:date|year|place) of birth|"
    r"(?:his|her|their|the) birth|"
    r"came into the world|hails? from|hailed from|native of)\b",
    flags=re.IGNORECASE,
)


_DEATH_ANCHOR = re.compile(
    r"\b(?:die|dies|died|death(?:date|place)?|place of death|"
    r"pass(?:ed)? away|passed|"
    r"deceased|killed|murdered|assassinated|executed|execution|succumbed|"
    r"perished|drowned|dead|committed suicide|took (?:his|her|their) "
    r"own life|murder victim|fatal end|mortally wounded|fatally wounded|"
    r"fatally shot|shot dead|"
    r"lost (?:his|her|their) life)\b",
    flags=re.IGNORECASE,
)


_DEATH_DENIAL = re.compile(
    r"\b(?:did(?: not|n't)(?: actually)? die|"
    r"has(?: not|n't)(?: actually)? died|had not died|never died|"
    r"never actually dies?|is (?:still |currently )?alive|"
    r"was (?:still )?alive|is not dead|was not dead|living person|"
    r"not (?:been )?reported as (?:dead|deceased))\b",
    flags=re.IGNORECASE,
)


_MISLEADING_TEMPORAL_PREFIX = re.compile(
    r"(?:\bas of\b|knowledge cutoff|last (?:knowledge )?update|"
    r"(?:primary |historical )?(?:records?|sources?) from|"
    r"\b(?:movie|film|book|album|song)\b.{0,35}\b(?:in|from|during)|"
    r"\(\s*r\.\s*(?:\d{1,4}\s*[-–—]\s*)?)"
    r"\s*$",
    flags=re.IGNORECASE,
)


_MISLEADING_TEMPORAL_BRIDGE = re.compile(
    # A birth/death word does not license every later date in its sentence.
    # Once the bridge introduces another dated event, the later date belongs
    # to that event instead.  Match event families rather than complete
    # phrases so that paraphrases receive the same treatment.
    r"\b(?:became|served|graduated|married|divorced|published|released|"
    r"founded|established|joined|moved|relocated|settled|appointed|"
    r"elected|retired|enlisted|worked|started|began|ended|won|received|"
    r"awarded|announc(?:e[ds]?|ed|ing|ement)|report(?:s|ed|ing)?|"
    r"mark(?:s|ed|ing)?|creat(?:e[ds]?|ed|ing|ion)|"
    r"reign(?:s|ed|ing)?|ascend(?:s|ed|ing)?|"
    r"abdica(?:te[ds]?|ted|ting|tion)|"
    r"last(?:s|ed|ing)?)\b[^.!?;\n]*$",
    flags=re.IGNORECASE,
)


def has_relation_anchor_before(
    answer: str,
    position: int,
    relation: Relation,
) -> bool:
    """Return whether the requested event is mentioned before ``position``."""
    if not 0 <= position <= len(answer):
        raise ValueError("anchor position must be inside the answer")
    pattern = (
        _BIRTH_ANCHOR
        if relation == Relation.BIRTH
        else _DEATH_ANCHOR
    )
    return pattern.search(answer[:position]) is not None


def _closest_event_anchor(
    sentence: str,
    start: int,
    end: int,
    requested: re.Pattern[str],
    opposite: re.Pattern[str],
) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    for kind, pattern in (("requested", requested), ("opposite", opposite)):
        for match in pattern.finditer(sentence):
            if match.end() <= start:
                distance = start - match.end()
                side = 0
            elif match.start() >= end:
                distance = match.start() - end
                side = 1
            else:
                distance = 0
                side = 0
            candidates.append((distance, side, kind))
    if not candidates:
        return None
    return min(candidates)[2]


def relation_match_is_relevant(answer: str, start: int, end: int, relation: Relation | str) -> bool:
    """Check event anchoring without using external factual knowledge."""
    relation = Relation(relation)
    if not 0 <= start < end <= len(answer):
        raise ValueError("relation span must be inside the answer")
    is_birth = relation is Relation.BIRTH
    if not is_birth and _DEATH_DENIAL.search(answer):
        return False
    requested = _BIRTH_ANCHOR if is_birth else _DEATH_ANCHOR
    opposite = _DEATH_ANCHOR if is_birth else _BIRTH_ANCHOR
    sentence_start, sentence_end = sentence_bounds_containing(answer, start)
    sentence = answer[sentence_start:sentence_end]
    local_start, local_end = start-sentence_start, end-sentence_start
    anchors = list(requested.finditer(sentence))
    if not anchors:
        span = re.sub(r"[*_`]", "", sentence[local_start:local_end])
        if len(re.findall(r"\b\w+\b", span)) > 6:
            return False
        outside = f"{sentence[:local_start]} {sentence[local_end:]}".strip(" \t\n.,;:!?*_`-'\"")
        outside = re.sub(r"\s+", " ", outside).casefold()
        return not outside or bool(re.fullmatch(
            r"(?:(?:the )?(?:correct )?(?:answer|date|year|place|location) (?:is|was)|it (?:is|was))", outside))
    before = [anchor for anchor in anchors if anchor.end() <= local_start]
    if not before or _closest_event_anchor(sentence,local_start,local_end,requested,opposite) != "requested":
        return False
    bridge = sentence[before[-1].end():local_start]
    if _MISLEADING_TEMPORAL_BRIDGE.search(bridge.rstrip()):
        return False
    prefix = sentence[max(0,local_start-120):local_start]
    return _MISLEADING_TEMPORAL_PREFIX.search(prefix.rstrip()) is None
