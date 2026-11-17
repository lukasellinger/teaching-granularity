"""Greedy Transformers inference with the trained location-component adapter."""

import hashlib
import json
from pathlib import Path

from parsing.location._inference import load_model

DIRECTORY = Path(__file__).resolve().parent
SETTINGS = json.loads((DIRECTORY / "model.json").read_text(encoding="utf-8"))


def format_prompt(question, answer):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Provide the requested birth/death location question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Provide a non-empty generated answer")
    content = (DIRECTORY / "prompt.txt").read_bytes()
    if hashlib.sha256(content).hexdigest() != SETTINGS["prompt_sha256"]:
        raise ValueError("prompt.txt differs from the trained extractor prompt")
    return content.decode("utf-8").strip().format(question=question, answer=answer)


def deduplicate(text):
    # Identical components only: no spelling, hierarchy or entity inference.
    parts, start, depth = [], 0, 0
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return text
        elif character == ";" and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if depth:
        return text
    parts.append(text[start:])
    if any(not part.strip() for part in parts):
        return text
    seen, unique = set(), []
    for part in parts:
        key = " ".join(part.split()).casefold()
        if key not in seen:
            seen.add(key)
            unique.append(part.strip())
    return "; ".join(unique) if len(unique) != len(parts) else text


def parse_output(raw, stopped=True):
    """Keep failures separate from the extractor's successful IDK output."""
    raw = raw.strip()
    status = "generation_truncated" if not stopped else ("ok" if raw else "empty_output")
    location = deduplicate(raw) if status == "ok" else None
    components = [] if location in (None, "IDK") else [part.strip() for part in location.split(";") if part.strip()]
    return {"status": status, "raw_location": raw, "location": location, "components": components}


def decode_output(tokens, eos_ids, decode):
    stop = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
    content = tokens if stop is None else tokens[:stop]
    return {
        **parse_output(decode(content, skip_special_tokens=True), stopped=stop is not None),
        "generated_tokens": len(content) + (stop is not None),
    }


class LocationExtractor:
    """Load once, then extract single answers or batches. No normalization/filtering."""

    def __init__(self, *, device="cuda", dtype="bfloat16", batch_size=16,
                 attention_implementation="sdpa", local_files_only=False, adapter_path=None):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model, self.tokenizer = load_model(
            SETTINGS, device=device, dtype=dtype,
            attention_implementation=attention_implementation, local_files_only=local_files_only,
            adapter_path=adapter_path,
        )
        self.batch_size = batch_size

    def extract(self, question, answer):
        return self.extract_many([{"question": question, "answer": answer}])[0]

    def extract_many(self, rows):
        """Return one result per input, preserving order and the full answer."""
        import torch

        results = [None] * len(rows)
        pending = []
        for index, row in enumerate(rows):
            sequence = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": format_prompt(row["question"], row["answer"])}],
                tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False,
            )
            if len(sequence) > SETTINGS["max_input_tokens"]:
                results[index] = {"status": "input_too_long", "raw_location": None,
                                  "location": None, "components": [], "input_tokens": len(sequence)}
            else:
                pending.append((index, sequence))
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            width = max(len(sequence) for _, sequence in batch)
            ids = [[self.tokenizer.pad_token_id] * (width - len(sequence)) + sequence for _, sequence in batch]
            mask = [[0] * (width - len(sequence)) + [1] * len(sequence) for _, sequence in batch]
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids=torch.tensor(ids, device=self.model.device),
                    attention_mask=torch.tensor(mask, device=self.model.device),
                    do_sample=False, num_beams=1, use_cache=True,
                    max_new_tokens=SETTINGS["max_new_tokens"],
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            eos_ids.add(self.tokenizer.eos_token_id)
            for (index, _), tokens in zip(batch, generated[:, width:].tolist(), strict=True):
                results[index] = decode_output(tokens, eos_ids, self.tokenizer.decode)
        return results
