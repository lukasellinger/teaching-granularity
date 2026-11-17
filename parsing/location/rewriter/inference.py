"""Generate and validate a focused event sentence for one supplied replacement."""

import re
from pathlib import Path

from parsing.location._inference import generate, load_model, prompt, settings
from .sentences import complete_sentence_prefix
from .validation import replacement_expression_count

DIRECTORY = Path(__file__).resolve().parent
SETTINGS = settings(DIRECTORY)


def format_prompt(row):
    return prompt(DIRECTORY, SETTINGS, **{key: row[key] for key in (
        "question", "answer", "original_location", "replacement_location")})


def protected_person_name(question, answer):
    match = re.fullmatch(r"Where (?:did|was) (.+?) (?:die|born)\?", question.strip())
    if match is None:
        return None
    subject = match.group(1).strip()
    return subject if subject and subject in answer else None


def validate(row, prediction):
    errors = []
    if replacement_expression_count(row, prediction) != 1:
        errors.append("replacement_count")
    protected = protected_person_name(row["question"], row["answer"])
    if protected is not None and protected not in prediction:
        errors.append("person_name_changed")
    if not prediction or complete_sentence_prefix(prediction) != prediction.rstrip():
        errors.append("incomplete_output")
    return errors


def postprocess(row, result):
    raw = result["raw_prediction"]
    prediction = complete_sentence_prefix(raw)
    errors = validate(row, prediction) if result["status"] == "ok" else []
    return {**result, "status": "output_validation_failed" if errors else result["status"],
            "prediction": prediction, "output_tail_trimmed": prediction != raw,
            "validation_errors": errors}


class LocationRewriter:
    """Rewrite one supplied location replacement per full-answer input."""

    def __init__(self, *, batch_size=16, **kwargs):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.tokenizer = load_model(SETTINGS, **kwargs)
        self.batch_size = batch_size

    def process_many(self, rows):
        results = generate(self.model, self.tokenizer, [format_prompt(row) for row in rows], SETTINGS, self.batch_size)
        return [postprocess(row, result) for row, result in zip(rows, results, strict=True)]
