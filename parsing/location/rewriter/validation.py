"""Validate a supplied replacement as one surface expression, not literal CSV."""
from __future__ import annotations

import re

VERSION = "location-rewriter-expression-validation-v2"


def replacement_expression_spans(row: dict, prediction: str) -> list[tuple[int, int]]:
    """Locate the validated expression in ORIGINAL character coordinates.

    Formatting and connecting prepositions may differ from the supplied string.
    Preserve coordinates while stripping emphasis/collapsing whitespace, and
    exclude the protected person's name rather than counting its place tokens.
    """
    hidden = set()
    for match in re.finditer(r"\*\*|\*|__", prediction):
        hidden.update(range(match.start(), match.end()))
    text, positions = [], []
    for index, char in enumerate(prediction):
        if index in hidden:
            continue
        if char.isspace():
            if text and text[-1] != " ":
                text.append(" ")
                positions.append(index)
        else:
            text.append(char)
            positions.append(index)
    clean = "".join(text)
    subject = re.fullmatch(r"Where (?:did|was) (.+?) (?:die|born)\?", row.get("question", "").strip())
    if subject and subject[1] in row.get("answer", ""):
        for match in re.finditer(re.escape(plain_text(subject[1])), clean):
            clean = clean[:match.start()] + " " * (match.end() - match.start()) + clean[match.end():]
    metadata = {**(row.get("normalized_value") or {}), "source_components": row.get("original_components", [])}
    parts = replacement_parts(row["replacement_location"], metadata)
    if not parts:
        return []
    joiner = r"(?:\s*,\s*(?:(?:in|at|on)\s+(?:the\s+)?)?|\s+(?:in|at|on)\s+(?:the\s+)?)"
    pattern = r"(?<!\w)" + joiner.join(re.escape(part) for part in parts) + r"(?!\w)"
    return [(positions[match.start()], positions[match.end()-1]+1)
            for match in re.finditer(pattern, clean, re.IGNORECASE)]


def plain_text(text: str) -> str:
    return " ".join(text.replace("**", "").replace("*", "").replace("__", "").split())


def replacement_parts(replacement: str, normalized_value: dict | None = None) -> list[str]:
    """Split hierarchy separators, preserving known commas inside entity names."""
    text = plain_text(replacement)
    protected = []
    for value in (normalized_value or {}).values():
        if isinstance(value, str):
            protected.append(plain_text(value))
        elif isinstance(value, list):
            protected.extend(plain_text(part) for part in value if isinstance(part, str))
    # PLACE can itself be a rendered comma list (home, address). Treat it as
    # an expression rather than an atomic Wikidata name.
    place = (normalized_value or {}).get("place")
    if isinstance(place, str):
        protected = [name for name in protected if name != plain_text(place)]
    internal_commas = set()
    for name in protected:
        for match in re.finditer(re.escape(name), text, re.IGNORECASE):
            internal_commas.update(match.start() + i for i, char in enumerate(name) if char == ",")
    # This common punctuated name must also work without normalization metadata.
    for match in re.finditer(r"\bWashington,\s*D\.C\.", text, re.IGNORECASE):
        internal_commas.add(text.index(",", match.start(), match.end()))
    starts, parts = 0, []
    for index, char in enumerate(text):
        if char == "," and index not in internal_commas:
            parts.append(text[starts:index].strip())
            starts = index + 1
    parts.append(text[starts:].strip())
    return [part for part in parts if part]


def replacement_expression_count(row: dict, prediction: str) -> int:
    """Count complete ordered mentions; flag repeated residual components too.

    Accept only commas or in/at/on (+ optional 'the') between supplied parts.
    Do not accept reordered, missing, renamed, near/outside-qualified parts,
    or components scattered across sentences. This is not a grammar checker.
    """
    text = plain_text(prediction)
    metadata = {**(row.get("normalized_value") or {})}
    if row.get("original_components"):
        metadata["source_components"] = row["original_components"]
    parts = replacement_parts(row["replacement_location"], metadata)
    if not parts:
        return 0
    subject = re.fullmatch(r"Where (?:did|was) (.+?) (?:die|born)\?", row.get("question", "").strip())
    if subject and subject[1] in row.get("answer", ""):
        # A geographic token inside the protected person's name is not a venue.
        text = text.replace(plain_text(subject[1]), "[PERSON]")
    joiner = r"(?:\s*,\s*(?:(?:in|at|on)\s+(?:the\s+)?)?|\s+(?:in|at|on)\s+(?:the\s+)?)"
    pattern = r"(?<!\w)" + joiner.join(re.escape(part) for part in parts) + r"(?!\w)"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if len(matches) != 1:
        return len(matches)
    match = matches[0]
    residual = text[:match.start()] + " [REPLACEMENT] " + text[match.end():]
    if any(re.search(r"(?<!\w)" + re.escape(part) + r"(?!\w)", residual, re.IGNORECASE)
           for part in parts):
        return 2
    return 1
