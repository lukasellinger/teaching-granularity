"""Conservative reference matching, not a semantic correctness judge."""
from __future__ import annotations

import re
from .validation import replacement_expression_count

SCORING_VERSION = "location-rewriter-tolerant-v1"


def formatting_normalize(text: str, replacement: str) -> str:
    # Preserve words, numbers, hyphens, apostrophes and commas inside names.
    text = text.replace("**", "").replace("*", "")
    text = " ".join(text.split())
    # A comma before a date/time preposition is optional in these answers.
    text = re.sub(r",\s+(?=(?:on|in)\s+(?:\*?\d|January|February|March|April|May|June|July|"
                  r"August|September|October|November|December))", " ", text)
    # Accept sentence punctuation versus the original unfinished punctuation.
    text = re.sub(r"[.,]+$", "", text).strip()
    # Accept 'the' only before these geographic unit names, not arbitrary words.
    if re.match(r"(?:State of |Community of |Greater .+ Region$|.+ Capital Area$|Western Cape$)", replacement):
        text = re.sub(r"\bin\s+(?:the\s+)?(?=" + re.escape(replacement) + r"(?:\W|$))",
                      "in ", text)
    return text


def formatting_match(case: dict, reference: str | None = None) -> bool:
    if case["status"] != "ok" or replacement_expression_count(case, case["prediction"]) != 1:
        return False
    reference = case["expected_output"] if reference is None else reference
    return formatting_normalize(case["prediction"], case["replacement_location"]) == \
        formatting_normalize(reference, case["replacement_location"])


def reference_match(case: dict) -> bool:
    """Match primary or explicitly approved reference variants; no free-form judge."""
    return any(formatting_match(case, reference)
               for reference in [case.get("corrected_expected_output", case["expected_output"]),
                                 *case.get("accepted_outputs", [])])


def scoring_description() -> dict:
    return {
        "version": SCORING_VERSION,
        "formatting_tolerant_exact_match": "Whitespace, Markdown emphasis, final period/comma, "
            "optional comma before dates, and optional article before specified geographic units. "
            "Words, dates, internal name punctuation and prepositions otherwise remain significant.",
        "accepted_reference_match": "Same formatting tolerance applied to the primary reference "
            "or explicitly corrected reference and any approved accepted_outputs. Corrections are case-specific; "
            "this is not an automatic semantic correctness metric.",
        "posthoc_warning": "Reference alternatives added after seeing predictions are adjudication, "
            "not an untouched heldout exact-match result. Always retain original strict scores.",
    }
