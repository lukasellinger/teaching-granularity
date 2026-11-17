"""Reference and structural metrics; not a semantic correctness judge."""
from .scoring import formatting_match, reference_match
from .validation import replacement_expression_count

def metrics(cases: list[dict]) -> dict:
    if not cases:
        raise ValueError("Cannot evaluate an empty split")
    normalize = lambda text: " ".join(text.split())
    exact = sum(c["status"] == "ok" and c["prediction"] == c["expected_output"] for c in cases)
    normalized = sum(c["status"] == "ok" and normalize(c["prediction"]) == normalize(c["expected_output"])
                     for c in cases)
    replacement = sum(c["status"] == "ok" and c["prediction"].count(c["replacement_location"]) == 1
                      for c in cases)
    return {"items": len(cases), "strict_exact_match": exact / len(cases),
            "whitespace_normalized_exact_match": normalized / len(cases),
            "formatting_tolerant_exact_match": sum(formatting_match(c) for c in cases) / len(cases),
            "accepted_reference_match": sum(reference_match(c) for c in cases) / len(cases),
            "replacement_exactly_once_rate": replacement / len(cases),
            "replacement_expression_exactly_once_rate": sum(
                c["status"] == "ok" and replacement_expression_count(c, c["prediction"]) == 1
                for c in cases) / len(cases),
            "valid_output_rate": sum(c["status"] == "ok" for c in cases) / len(cases),
            "person_name_preservation_rate": sum(
                "person_name_changed" not in c.get("validation_errors", []) for c in cases
            ) / len(cases),
            "output_validation_failures": sum(
                c["status"] == "output_validation_failed" for c in cases
            ),
            "truncated_outputs": sum(c["status"] == "generation_truncated" for c in cases),
            "empty_outputs": sum(not c["prediction"] for c in cases)}
