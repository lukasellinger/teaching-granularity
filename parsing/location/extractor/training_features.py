"""Source/reference feature tags used for extractor training weights."""

import math

import re

VERSION = "extractor-difficulty-features-v1"

DEFAULT_WEIGHT = 1.5

CATEGORIES = ("venue_or_address", "spatial_qualifier", "alias_or_historical_context",
              "event_anchoring", "multi_sentence", "truncated_source")

FOCUS_CATEGORIES = CATEGORIES[:4]

_VENUE = re.compile(r"\b(?:home|house|hospital|palace|palazzo|prison|asylum|lodge|estate|"
                    r"castle|fortress|hall|clinic|street|road|room|cell|wing|barracks|"
                    r"residence|hotel|townhouse|headquarters|homestead)\b", re.I)

_SPATIAL = re.compile(r"\b(?:near|outside|off the coast|southwest|southeast|northeast|"
                      r"northwest|northern|southern|eastern|western|mountains of|"
                      r"coast of|slope of|vicinity|around)\b", re.I)

_ALIASES = re.compile(r"\b(?:also known|also spelled|also referred|sometimes spelled|"
                      r"now known|modern-day|present-day|formerly|historically|"
                      r"empire|kingdom|sultanate|governorate|duchy|dynasty)\b", re.I)

_ANCHORING = re.compile(r"\b(?:buried|burial|resting place|lived|living|residing|raised|"
                        r"residence|admitted|admission|hospitalized|exiled|imprisoned|"
                        r"held in|while staying|bishop of|archbishop of|still alive|"
                        r"has not died|did not die)\b", re.I)

def categorize(row: dict) -> list[str]:
    source = row["answer"]
    target = row["expected_output"]
    tags = set()
    if _VENUE.search(target) or re.search(r"\b\d+\s+\w+\s+(?:street|road)\b", target, re.I):
        tags.add("venue_or_address")
    if _SPATIAL.search(target):
        tags.add("spatial_qualifier")
    if _ALIASES.search(source) or "/" in target or "'s " in target:
        tags.add("alias_or_historical_context")
    both_events = (re.search(r"\bborn\b", source, re.I)
                   and re.search(r"\b(?:died|passed away|death)\b", source, re.I))
    if _ANCHORING.search(source) or both_events:
        tags.add("event_anchoring")
    # Features only: punctuation in abbreviations can over-tag a case. This is
    # exposed for review, not presented as a semantic multiple-location label.
    if "\n\n" in source or len(re.split(r"[.!?]\s+(?=[A-Z])", source)) > 1:
        tags.add("multi_sentence")
    plain_end = source.rstrip().rstrip('*_`"\u201d\u2019')
    if (not plain_end.endswith((".", "!", "?"))
            or source.count("(") != source.count(")")
            or source.count("[") != source.count("]")):
        tags.add("truncated_source")
    return [category for category in CATEGORIES if category in tags]
