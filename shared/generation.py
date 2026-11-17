"""Minimal greedy generation with explicit truncation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from shared.chat import render_chat

from functools import lru_cache


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    generated_tokens: int
    finish_reason: str
    token_logprob_sum: float | None = None
    mean_token_logprob: float | None = None
    generated_token_ids: tuple[int, ...] = ()
    token_logprobs: tuple[float, ...] = ()
    first_sentence: str | None = None
    first_sentence_tokens: int = 0
    first_sentence_logprob_sum: float | None = None
    first_sentence_mean_token_logprob: float | None = None


def _sentence_token_count(tokenizer, token_ids: list[int], sentence: str) -> int:
    """Map a decoded first sentence back to its emitted token prefix."""
    if not sentence:
        return 0
    for count in range(1, len(token_ids) + 1):
        decoded = tokenizer.batch_decode(
            [torch.tensor(token_ids[:count])],
            skip_special_tokens=True,
        )[0].strip()
        if decoded == sentence or decoded.startswith(sentence):
            return count
    raise ValueError("first sentence cannot be aligned to generated token prefix")


def model_input_device(model) -> torch.device:
    """Find the device holding input embeddings, including sharded models."""
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise ValueError("model input embeddings have no weight")
    if weight.device.type == "meta":
        device = getattr(model, "device", None)
        if device is None or device.type == "meta":
            raise ValueError("cannot determine input device for sharded model")
        return device
    return weight.device


def _eos_ids(model, tokenizer) -> set[int]:
    configured = getattr(model.generation_config, "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    if configured is None:
        return set()
    if isinstance(configured, int):
        return {configured}
    return {int(value) for value in configured}


@torch.inference_mode()
def generate_greedy(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> list[GeneratedAnswer]:
    """Generate without sampling and never truncate an input silently."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompts = list(prompts)
    if not prompts:
        return []
    model.eval()
    input_device = model_input_device(model)
    eos_ids = _eos_ids(model, tokenizer)
    results: list[GeneratedAnswer] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        rendered = [
            render_chat(tokenizer, prompt, None, system_prompt)
            for prompt in batch
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(input_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        prompt_width = inputs["input_ids"].shape[1]
        sequences = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated = sequences[:, prompt_width:]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for token_ids, answer in zip(generated.tolist(), decoded, strict=True):
            eos_position = next(
                (
                    index
                    for index, token_id in enumerate(token_ids)
                    if token_id in eos_ids
                ),
                None,
            )
            if eos_position is not None:
                token_count = eos_position
                finish_reason = "eos"
            else:
                token_count = len(token_ids)
                finish_reason = (
                    "max_tokens"
                    if len(token_ids) >= max_new_tokens
                    else "stopped"
                )
            results.append(
                GeneratedAnswer(
                    answer=answer.strip(),
                    generated_tokens=token_count,
                    finish_reason=finish_reason,
                )
            )
    if len(results) != len(prompts):
        raise RuntimeError("generation output count does not match prompt count")
    return results


@torch.inference_mode()
def generate_greedy_with_logprobs(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    system_prompt: str | None = None,
) -> list[GeneratedAnswer]:
    """Greedily generate and score only the emitted answer tokens.

    The arithmetic mean avoids the length confound of a sequence log-score.
    EOS and padding tokens are excluded, matching ``generated_tokens``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompts = list(prompts)
    if not prompts:
        return []
    model.eval()
    input_device = model_input_device(model)
    eos_ids = _eos_ids(model, tokenizer)
    results: list[GeneratedAnswer] = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        rendered = [
            render_chat(tokenizer, prompt, None, system_prompt)
            for prompt in batch
        ]
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        inputs = {
            key: value.to(input_device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        prompt_width = inputs["input_ids"].shape[1]
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated = output.sequences[:, prompt_width:]
        if len(output.scores) != generated.shape[1]:
            raise RuntimeError("generation scores do not align with emitted tokens")
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row_index, (token_ids, answer) in enumerate(
            zip(generated.tolist(), decoded, strict=True)
        ):
            eos_position = next(
                (
                    index
                    for index, token_id in enumerate(token_ids)
                    if token_id in eos_ids
                ),
                None,
            )
            if eos_position is not None:
                token_count = eos_position
                finish_reason = "eos"
            else:
                token_count = len(token_ids)
                finish_reason = (
                    "max_tokens"
                    if len(token_ids) >= max_new_tokens
                    else "stopped"
                )
            token_logprobs = (
                torch.stack(
                    [
                        torch.log_softmax(
                            output.scores[step][row_index].float(), dim=-1
                        )[token_ids[step]]
                        for step in range(token_count)
                    ]
                )
                if token_count
                else torch.empty(0)
            )
            token_logprob_values = tuple(
                float(value) for value in token_logprobs.tolist()
            )
            score_sum = float(token_logprobs.sum().item()) if token_count else None
            stripped_answer = answer.strip()
            first_sentence = first_nonempty_sentence(stripped_answer)
            first_sentence_tokens = _sentence_token_count(
                tokenizer,
                token_ids[:token_count],
                first_sentence,
            )
            first_sentence_sum = (
                float(token_logprobs[:first_sentence_tokens].sum().item())
                if first_sentence_tokens
                else None
            )
            results.append(
                GeneratedAnswer(
                    answer=stripped_answer,
                    generated_tokens=token_count,
                    finish_reason=finish_reason,
                    token_logprob_sum=score_sum,
                    mean_token_logprob=(
                        score_sum / token_count
                        if score_sum is not None and token_count
                        else None
                    ),
                    generated_token_ids=tuple(token_ids[:token_count]),
                    token_logprobs=token_logprob_values,
                    first_sentence=first_sentence,
                    first_sentence_tokens=first_sentence_tokens,
                    first_sentence_logprob_sum=first_sentence_sum,
                    first_sentence_mean_token_logprob=(
                        first_sentence_sum / first_sentence_tokens
                        if first_sentence_sum is not None and first_sentence_tokens
                        else None
                    ),
                )
            )
    if len(results) != len(prompts):
        raise RuntimeError("generation output count does not match prompt count")
    return results


@lru_cache(maxsize=1)
def _english_sentencizer():
    import spacy

    pipeline = spacy.blank("en")
    pipeline.add_pipe("sentencizer")
    return pipeline


def first_nonempty_sentence(text: str) -> str:
    """Return the first rule-based sentence, or the stripped input."""
    stripped = text.strip()
    if not stripped:
        return ""
    document = _english_sentencizer()(stripped)
    return next(
        (sentence.text.strip() for sentence in document.sents if sentence.text.strip()),
        stripped,
    )

