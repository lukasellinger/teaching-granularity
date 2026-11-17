from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Sequence


PROMPTS = json.loads(Path(__file__).with_name("prompts.json").read_text())

DEFAULT_TOKEN_OVERLAP_THRESHOLD = 0.5
DEFAULT_NGRAM_OVERLAP_THRESHOLD = 0.8
DEFAULT_NGRAM_SIZE = 3


def normalize_answer(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text).rstrip(". ")


def answer_tokens(value: object) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", normalize_answer(value)))


def character_ngrams(value: object, size: int) -> frozenset[str]:
    text = re.sub(r"[^\w]+", " ", normalize_answer(value)).strip()
    if not text:
        return frozenset()
    if len(text) <= size:
        return frozenset({text})
    return frozenset(
        text[index:index + size]
        for index in range(len(text) - size + 1)
    )


def overlap_coefficient(
    left: frozenset[str],
    right: frozenset[str],
) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def build_hierarchy_conflicts(
    rows: Sequence[dict],
) -> dict[str, set[str]]:
    """Map answers to other answers observed in the same hierarchy."""
    conflicts: dict[str, set[str]] = {}
    for row in rows:
        answers = {
            normalize_answer(answer)
            for _, answer in validate_hierarchy(row)
        }
        for left, right in combinations(sorted(answers), 2):
            conflicts.setdefault(left, set()).add(right)
            conflicts.setdefault(right, set()).add(left)
    return conflicts


def answers_are_confusable(
    left: object,
    right: object,
    hierarchy_conflicts: dict[str, set[str]] | None = None,
    *,
    token_overlap_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
    ngram_overlap_threshold: float = DEFAULT_NGRAM_OVERLAP_THRESHOLD,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> bool:
    """Return whether two answers are too related for one candidate set."""
    left_normalized = normalize_answer(left)
    right_normalized = normalize_answer(right)
    if left_normalized == right_normalized:
        return True

    if hierarchy_conflicts and (
        right_normalized in hierarchy_conflicts.get(left_normalized, set())
        or left_normalized in hierarchy_conflicts.get(right_normalized, set())
    ):
        return True

    token_overlap = overlap_coefficient(
        answer_tokens(left_normalized),
        answer_tokens(right_normalized),
    )
    if token_overlap >= token_overlap_threshold:
        return True

    ngram_overlap = overlap_coefficient(
        character_ngrams(left_normalized, ngram_size),
        character_ngrams(right_normalized, ngram_size),
    )
    return ngram_overlap >= ngram_overlap_threshold


def stable_random(seed: int, *parts: object) -> random.Random:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def row_key(row: dict, fallback_index: int) -> object:
    return row.get("id", row.get("row_index", fallback_index))


def validate_hierarchy(row: dict) -> list[tuple[float, str]]:
    answers = list(row.get("granola_answers") or [])
    levels = list(row.get("normalized_levels") or [])
    if len(answers) != len(levels):
        raise ValueError(
            "granola_answers and normalized_levels have different lengths "
            f"for row {row.get('id', row.get('row_index'))!r}."
        )
    hierarchy = []
    for level, answer in zip(levels, answers, strict=True):
        answer = str(answer).strip()
        if not answer:
            continue
        hierarchy.append((float(level), answer))
    return sorted(hierarchy, key=lambda pair: pair[0], reverse=True)


def build_candidate_pools(
    rows: Sequence[dict],
) -> dict[tuple[str, float], list[str]]:
    pools: dict[tuple[str, float], dict[str, str]] = {}
    for row in rows:
        relation = str(row["relation"])
        for level, answer in validate_hierarchy(row):
            normalized = normalize_answer(answer)
            pools.setdefault((relation, level), {}).setdefault(
                normalized,
                answer,
            )
    return {
        key: sorted(values.values(), key=lambda value: normalize_answer(value))
        for key, values in pools.items()
    }


def create_prompt_variants(
    row: dict,
    count: int,
    seed: int,
    fallback_index: int = 0,
) -> list[str]:
    if count < 1:
        raise ValueError("Prompt variant count must be positive.")

    original = str(row["question"]).strip()
    relation = str(row["relation"])
    entity = str(row.get("question_entity", row.get("entity", "")))
    variants = [original]
    templates = PROMPTS.get(relation, [])
    rendered = [template.format(entity=entity) for template in templates]
    # A valid relation template should interpolate the entity exactly once.
    # This also protects scoring from accidentally concatenated templates.
    rendered = [prompt for prompt in rendered if prompt.count(entity) == 1]
    rendered = [
        prompt
        for prompt in rendered
        if normalize_answer(prompt) != normalize_answer(original)
    ]
    rng = stable_random(
        seed,
        "prompts",
        row_key(row, fallback_index),
        relation,
    )
    rng.shuffle(rendered)
    for prompt in rendered:
        if all(
            normalize_answer(prompt) != normalize_answer(existing)
            for existing in variants
        ):
            variants.append(prompt)
        if len(variants) == count:
            break
    if len(variants) < count:
        raise ValueError(
            f"Only {len(variants)} distinct prompts are available for "
            f"relation {relation!r}; requested {count}."
        )
    return variants


def build_distractor_panel(
    *,
    row: dict,
    fallback_index: int,
    level: float,
    correct: str,
    pool: Sequence[str],
    panel_size: int,
    seed: int,
    hierarchy_conflicts: dict[str, set[str]] | None = None,
    token_overlap_threshold: float = DEFAULT_TOKEN_OVERLAP_THRESHOLD,
    ngram_overlap_threshold: float = DEFAULT_NGRAM_OVERLAP_THRESHOLD,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> list[str] | None:
    candidates = [
        candidate
        for candidate in pool
        if not answers_are_confusable(
            correct,
            candidate,
            hierarchy_conflicts,
            token_overlap_threshold=token_overlap_threshold,
            ngram_overlap_threshold=ngram_overlap_threshold,
            ngram_size=ngram_size,
        )
    ]
    rng = stable_random(
        seed,
        "distractors",
        row_key(row, fallback_index),
        row["relation"],
        level,
    )
    rng.shuffle(candidates)
    sampled = []
    for candidate in candidates:
        if any(
            answers_are_confusable(
                candidate,
                existing,
                hierarchy_conflicts,
                token_overlap_threshold=token_overlap_threshold,
                ngram_overlap_threshold=ngram_overlap_threshold,
                ngram_size=ngram_size,
            )
            for existing in sampled
        ):
            continue
        sampled.append(candidate)
        if len(sampled) == panel_size:
            break
    if len(sampled) < panel_size:
        return None
    return sampled


class ContinuationEncoder:
    def __init__(self, tokenizer, assistant_prefix: str):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "Candidate scoring requires a fast tokenizer with offsets."
            )
        self.tokenizer = tokenizer
        self.assistant_prefix = assistant_prefix.rstrip()

    def _render_question(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def encode(self, question: str, candidate: str) -> dict:
        candidate = candidate.strip()
        if not candidate:
            raise ValueError("Candidates must not be empty.")
        prefix = self._render_question(question) + self.assistant_prefix
        # With no verbal prefix, score the candidate exactly as the assistant's
        # first output. A space is only needed to separate an explicit prefix
        # such as "The answer is" from the answer itself.
        continuation = f" {candidate}" if self.assistant_prefix else candidate
        full_text = prefix + continuation
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        candidate_start = len(prefix)
        candidate_end = len(full_text)
        candidate_positions = [
            index
            for index, (start, end) in enumerate(encoded["offset_mapping"])
            if end > candidate_start and start < candidate_end
        ]
        if not candidate_positions or candidate_positions[0] == 0:
            raise ValueError(
                "Could not identify candidate tokens after the prompt."
            )
        return {
            "input_ids": encoded["input_ids"],
            "candidate_positions": candidate_positions,
        }


def summarize_token_logprobs(
    token_logprobs: Sequence[float],
) -> dict[str, float | int]:
    if not token_logprobs:
        raise ValueError("No candidate token log probabilities were returned.")
    total = float(sum(token_logprobs))
    return {
        "sum_logprob": total,
        "mean_logprob": total / len(token_logprobs),
        "token_count": len(token_logprobs),
    }


def score_level(
    *,
    scorer,
    prompts: Sequence[str],
    correct: str,
    distractors: Sequence[str],
) -> dict:
    candidates = [
        ("correct", correct),
        *(("distractor", value) for value in distractors),
    ]
    questions = []
    answers = []
    for _, candidate in candidates:
        for prompt in prompts:
            questions.append(prompt)
            answers.append(candidate)
    scores = scorer.score(questions, answers)
    score_index = 0
    candidate_records = []
    for panel, candidate in candidates:
        prompt_scores = scores[score_index:score_index + len(prompts)]
        score_index += len(prompts)
        candidate_records.append({
            "panel": panel,
            "answer": candidate,
            "prompt_scores": prompt_scores,
        })
    return {
        "correct": candidate_records[0],
        "distractors": candidate_records[1:],
    }
