"""Shared chat-template rendering for training and generation."""

from __future__ import annotations

_GEMMA_ASSISTANT_HEADER = "<|turn>model\n"
_GEMMA_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


def render_chat(
    tokenizer,
    prompt: str,
    assistant: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """Render exactly the same conversation shape in every pipeline stage."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=assistant is None,
        enable_thinking=False,
    )
    if assistant is not None and _GEMMA_ASSISTANT_HEADER in rendered:
        # Some Gemma 4 templates include the no-thinking wrapper only in the
        # generation prompt, not when rendering a completed assistant turn.
        # Repair only that exact asymmetry, preserving the answer and native
        # end-of-turn suffix. Do not weaken token-prefix validation or change
        # templates that already render completed turns consistently.
        prefix = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if prefix.endswith(_GEMMA_EMPTY_THOUGHT) and not rendered.startswith(prefix):
            bare_prefix = prefix[: -len(_GEMMA_EMPTY_THOUGHT)]
            if rendered.startswith(bare_prefix + assistant):
                rendered = prefix + rendered[len(bare_prefix) :]
    return rendered

