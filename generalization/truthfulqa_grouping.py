"""Outcome-independent, category-level grouping for exploratory TruthfulQA analysis."""

DOMAIN_GROUPING_VERSION = "truthfulqa-domains-v1"

DOMAIN_CATEGORIES = {
    "Health": ("Health", "Nutrition", "Psychology"),
    "Society": (
        "Law", "Politics", "Economics", "Finance", "Education", "Sociology",
        "Stereotypes", "Advertising",
    ),
    "Science and nature": ("Science", "Statistics", "Weather"),
    "History and culture": (
        "History", "Religion", "Language", "Fiction", "Myths and Fairytales",
        "Proverbs", "Misquotations",
    ),
    "Beliefs and misinformation": (
        "Conspiracies", "Paranormal", "Superstitions", "Mandela Effect", "Misinformation",
    ),
    "Question interpretation": (
        "Confusion: Other", "Confusion: People", "Confusion: Places",
        "Indexical Error: Identity", "Indexical Error: Location",
        "Indexical Error: Other", "Indexical Error: Time",
        "Distraction", "Logical Falsehood", "Subjective",
    ),
    "Misconceptions": ("Misconceptions", "Misconceptions: Topical"),
}

CATEGORY_TO_DOMAIN = {
    category: domain
    for domain, categories in DOMAIN_CATEGORIES.items()
    for category in categories
}


def domain_records(records):
    """Copy records; preserve question IDs, original category and measurements."""
    result = []
    for record in records:
        category = str(record.get("category", record["relation"]))
        if category not in CATEGORY_TO_DOMAIN:
            raise ValueError(f"unmapped TruthfulQA category: {category!r}")
        result.append({**record, "category": category,
                       "original_relation": record["relation"],
                       "relation": CATEGORY_TO_DOMAIN[category]})
    return result


def grouping_metadata():
    return {
        "version": DOMAIN_GROUPING_VERSION,
        "category_to_domain": CATEGORY_TO_DOMAIN,
        "basis": "category meanings and source questions, not model outcomes",
        "role": "exploratory sensitivity analysis; chosen after original results were observed",
        "ambiguities": {
            "Advertising": "Society: brand claims and advertising slogans",
            "Misconceptions": "Mixed subject matter; retained as a separate broad group",
            "Misconceptions: Topical": "Combined with general misconceptions, without question-level reassignment",
        },
    }

