"""Shared conservative period/abbreviation handling."""
import re

_SENTENCE_END = re.compile(
    r"(?P<punctuation>[.!?]+)(?P<closers>[\"'”’)}\]*_]*)"
    r"(?=\s|$)"
)


_COMMON_ABBREVIATIONS = {
    "ca",
    "dr",
    "e.g",
    "etc",
    "i.e",
    "jr",
    "mr",
    "mrs",
    "ms",
    "mt",
    "no",
    "prof",
    "sr",
    "st",
    "vs",
}


def _is_nonterminal_period(text: str, match: re.Match[str]) -> bool:
    punctuation = match.group("punctuation")
    if punctuation != ".":
        return False

    remainder = text[match.end():].lstrip()
    if not remainder:
        return False

    before = text[:match.end("punctuation")]
    token_match = re.search(r"([A-Za-z.]+)\.$", before)
    if token_match is None:
        return False
    token = token_match.group(1).rstrip(".")
    normalized = token.casefold()
    if normalized in _COMMON_ABBREVIATIONS:
        return True
    if len(token) == 1 and token.isalpha():
        return True
    return bool(re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]?", token))

