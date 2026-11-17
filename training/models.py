"""Model identifiers and narrowly scoped LoRA targets."""

from __future__ import annotations
from shared.constants import MODEL_IDS


def lora_target_modules(model_name: str) -> list[str] | str:
    """Avoid adapting vision towers that reuse language projection names."""
    if model_name == "MINISTRAL3_3B":
        return (r"(?:.*\.)?language_model\.layers\.\d+\."
                r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
                r"mlp\.(?:gate_proj|up_proj|down_proj))")
    if model_name == "QWEN3_5_9B":
        return (
            r"(?:.*\.)?(?:language_model|model)\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"linear_attn\.(?:in_proj_qkv|in_proj_z|"
            r"in_proj_a|in_proj_b|out_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))"
        )
    if model_name == "GEMMA4_31B":
        return (
            r"(?:.*\.)?language_model\.layers\.\d+\."
            r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
            r"mlp\.(?:gate_proj|up_proj|down_proj))"
        )
    if model_name == "LFM2_5_1_2B":
        return [
            "w1",
            "w2",
            "w3",
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
        ]
    raise ValueError(f"no LoRA targets configured for {model_name!r}")


def validate_lora_coverage(model, model_name: str) -> None:
    if model_name != "QWEN3_5_9B":
        return
    targeted = tuple(getattr(model, "targeted_module_names", ()))
    required = (".linear_attn.in_proj_qkv", ".linear_attn.out_proj")
    missing = [
        fragment
        for fragment in required
        if not any(fragment in name for name in targeted)
    ]
    if missing:
        raise RuntimeError(
            "Qwen LoRA missed required linear-attention modules: "
            f"{missing}"
        )


