from __future__ import annotations
import pytest
from training.tests.test_data import CharacterTokenizer, example
from shared.chat import render_chat
from training.data import WeightedRealization
from training.encoding import encode_realization
EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class GemmaTemplateTokenizer(CharacterTokenizer):
    """Reproduce the upstream generation/completed-turn asymmetry offline."""

    def __init__(self, *, fixed=False, mismatched=False):
        self.fixed = fixed
        self.mismatched = mismatched

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and not enable_thinking
        rendered = "<bos>"
        for message in messages:
            assistant = message["role"] == "assistant"
            role = "model" if assistant else message["role"]
            rendered += f"<|turn>{role}\n"
            if assistant and self.fixed:
                rendered += EMPTY_THOUGHT
            if assistant and self.mismatched:
                rendered += "unexpected header\n"
            rendered += message["content"] + "<turn|>\n"
        if add_generation_prompt:
            rendered += "<|turn>model\n" + EMPTY_THOUGHT
        return rendered

@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("system", [None, "Answer briefly."])
def test_gemma_completed_turn_uses_generation_prefix_once(fixed, system):
    tokenizer = GemmaTemplateTokenizer(fixed=fixed)
    prefix = render_chat(tokenizer, "When was A born?", system_prompt=system)
    completed = render_chat(
        tokenizer, "When was A born?", "A was born in 1900.", system
    )
    assert completed == prefix + "A was born in 1900.<turn|>\n"
    assert completed.count(EMPTY_THOUGHT) == 1

def test_non_gemma_rendering_is_unchanged():
    tokenizer = CharacterTokenizer()
    assert render_chat(tokenizer, "Question", "Answer", "System") == (
        "<system>System<user>Question<assistant>Answer"
    )

def check_encoding(tokenizer):
    positive = "A was born in 1900."
    item = example("case:0", "Q1", positive, rankable=True, prompt_template_id=0)
    # Answer text occurring in the user message must not move assistant spans.
    item = item.model_copy(update={"prompt": f"Rewrite: {positive}"})
    prefix = render_chat(tokenizer, item.prompt)
    prompt_ids = tokenizer(
        prefix,
        add_special_tokens=False,
        return_attention_mask=True,
        return_offsets_mapping=False,
        truncation=False,
    )["input_ids"]
    encoded = encode_realization(
        WeightedRealization(item, 1.0, 1.0),
        tokenizer,
        seed=42,
        epoch=0,
        max_negatives_per_group=None,
        max_length=512,
        system_prompt=None,
    )
    for sequence in encoded["sequences"]:
        assert sequence["input_ids"][: len(prompt_ids)] == list(prompt_ids)
        assert not any(sequence["sft_mask"][: len(prompt_ids)])
        assert all(sequence["sft_mask"][len(prompt_ids) :])
    return encoded

def test_gemma_repair_preserves_sft_and_candidate_score_offsets():
    encoded = check_encoding(GemmaTemplateTokenizer())
    positive = encoded["sequences"][encoded["positive_sequence_index"]]
    supervised = "".join(
        chr(token)
        for token, selected in zip(positive["input_ids"], positive["sft_mask"])
        if selected
    )
    assert supervised == "A was born in 1900.<turn|>\n"
    scored = []
    for index, mask in zip(
        encoded["candidate_sequence_indices"], encoded["candidate_score_masks"]
    ):
        ids = encoded["sequences"][index]["input_ids"]
        scored.append(
            "".join(chr(token) for token, selected in zip(ids, mask) if selected)
        )
    assert scored == ["1900", "the 1900s"]

def test_unrelated_prefix_mismatch_still_fails_instead_of_being_repaired():
    with pytest.raises(ValueError, match="generation prompt is not a token prefix"):
        check_encoding(GemmaTemplateTokenizer(mismatched=True))
