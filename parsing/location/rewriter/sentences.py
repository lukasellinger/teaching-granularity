"""Conservative complete-sentence output boundary detection."""

import re
from shared.text import _SENTENCE_END, _COMMON_ABBREVIATIONS, _is_nonterminal_period


def complete_sentence_prefix(text: str) -> str:
    """Keep the longest conservative sentence-complete prefix, without edits.

    A trailing abbreviation/ellipsis is not proof of sentence completion.
    D.C. can close a sentence before an unmistakable new sentence opener.
    """
    end = 0
    for match in _SENTENCE_END.finditer(text):
        if match.group("punctuation") in {"...", ".."}:
            continue
        before = text[:match.end("punctuation")]
        token = re.search(r"([A-Za-z.]+)\.$", before)
        if token and token[1].rstrip(".").casefold() in _COMMON_ABBREVIATIONS:
            continue
        nonterminal = _is_nonterminal_period(text, match)
        if nonterminal and re.search(r"\bD\.C\.$", before) and re.match(
                r"(?:He|She|They|His|Her|It|The|Later|However)\s", text[match.end():].lstrip()):
            nonterminal = False
        if not nonterminal:
            end = match.end()
    return text[:end].rstrip()

