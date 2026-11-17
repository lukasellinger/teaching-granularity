import re

import pytest

from shared.models import question_only_template, NO_DEFAULT_SYSTEM_MARKER
from training.models import MODEL_IDS, lora_target_modules


def test_ministral_registered_and_vision_not_adapted():
    assert MODEL_IDS["MINISTRAL3_3B"].endswith("-BF16")
    expression = lora_target_modules("MINISTRAL3_3B")
    assert re.fullmatch(expression, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(expression, "model.language_model.layers.1.mlp.down_proj")
    assert not re.fullmatch(expression, "model.vision_tower.layers.0.self_attn.q_proj")


def test_implicit_system_prompt_disabled_without_changing_rest():
    template = "{%- set default_system_message = 'Original' %}\n" + NO_DEFAULT_SYSTEM_MARKER + "\n{{ bos_token }}"
    changed = question_only_template(template)
    assert "set default_system_message = ''" in changed
    assert changed.endswith(NO_DEFAULT_SYSTEM_MARKER + "\n{{ bos_token }}")
    with pytest.raises(ValueError):
        question_only_template("unknown template")
