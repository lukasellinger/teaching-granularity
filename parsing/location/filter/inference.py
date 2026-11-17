"""Frozen Single/Multiple label-logprob safety classifier."""

import math
import json
from pathlib import Path

from parsing.location._inference import encode, load_model, padded, prompt, settings

DIRECTORY = Path(__file__).resolve().parent
SETTINGS = settings(DIRECTORY)


def classify(scores, threshold=SETTINGS["multiple_log_odds_threshold"]):
    if not math.isfinite(threshold):
        raise ValueError("Filter threshold must be finite")
    if any(not math.isfinite(scores[label]) for label in ("Single", "Multiple")):
        return {"filter_status": "invalid_scores", "filter_prediction": None}
    odds = scores["Multiple"] - scores["Single"]
    return {"filter_status": "ok", "filter_prediction": "Multiple" if odds >= threshold else "Single",
            "multiple_log_odds": odds, "label_logprob": scores, "multiple_log_odds_threshold": threshold}


def input_location(row):
    extraction = row.get("extraction", row.get("event_extraction"))
    if not isinstance(extraction, dict):
        raise ValueError("Provide extraction or event_extraction with its location")
    status = extraction.get("status", "ok")
    if status != "ok":
        return None, f"upstream_{status}"
    location = extraction.get("location")
    if not isinstance(location, str) or not location.strip() or location == "IDK":
        return None, "no_event_location"
    return location, None


def format_prompt(row, location):
    return prompt(DIRECTORY, SETTINGS, question=row["question"], answer=row["answer"], event_location=location)


class LocationFilter:
    """Classify training-answer safety using the supplied event extraction."""

    def __init__(self, *, batch_size=16, adapter_path=None, **kwargs):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.tokenizer = load_model(SETTINGS, adapter_path=adapter_path, **kwargs)
        self.batch_size = batch_size
        self.threshold = SETTINGS["multiple_log_odds_threshold"]
        if adapter_path is not None:
            # A newly trained filter requires its own dev-selected threshold.
            self.threshold = json.loads((Path(adapter_path) / "threshold.json").read_text())["multiple_log_odds_threshold"]
        if not math.isfinite(self.threshold):
            raise ValueError("Filter threshold must be finite")

    def process_many(self, rows):
        return score_rows(self.model, self.tokenizer, rows, self.batch_size, self.threshold)


def score_rows(model, tokenizer, rows, batch_size=16, threshold=SETTINGS["multiple_log_odds_threshold"]):
    targets = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("Single", "Multiple")}
    if any(len(tokens) != 1 for tokens in targets.values()):
        raise ValueError("Single and Multiple must each be one token")
    targets = {label: tokens[0] for label, tokens in targets.items()}
    import torch

    results, pending = [None] * len(rows), []
    for index, row in enumerate(rows):
        location, reason = input_location(row)
        if reason:
            results[index] = {"filter_status": reason, "filter_prediction": None}
            continue
        sequence = encode(tokenizer, format_prompt(row, location))
        if len(sequence) > SETTINGS["max_input_tokens"]:
            results[index] = {"filter_status": "input_too_long", "filter_prediction": None, "filter_input_tokens": len(sequence)}
        else:
            pending.append((index, sequence))
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        ids, mask = padded([sequence for _, sequence in batch], tokenizer.pad_token_id)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor(ids, device=model.device),
                                attention_mask=torch.tensor(mask, device=model.device), use_cache=False).logits[:, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
        for offset, (index, sequence) in enumerate(batch):
            results[index] = {**classify({label: log_probs[offset, token].item() for label, token in targets.items()}, threshold),
                              "filter_input_tokens": len(sequence)}
    return results
