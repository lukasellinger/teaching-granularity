"""Conservative sentence boundaries around temporal mentions."""
from __future__ import annotations
import re
from shared.text import _SENTENCE_END, _COMMON_ABBREVIATIONS, _is_nonterminal_period


def sentence_bounds_containing(text: str, position: int) -> tuple[int, int]:
    """Return conservative sentence/paragraph bounds around a position."""
    if not 0 <= position <= len(text):
        raise ValueError("position must be inside the text")

    boundaries = [
        match.end()
        for match in _SENTENCE_END.finditer(text)
        if not _is_nonterminal_period(text, match)
    ]
    boundaries.extend(match.end() for match in re.finditer(r"\n+", text))

    start = 0
    for boundary in sorted(set(boundaries)):
        if boundary >= position:
            return start, boundary
        start = boundary
    return start, len(text)


