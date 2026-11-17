"""Small inference helpers shared by location models; no normalization logic."""

import hashlib
import json
from pathlib import Path


def settings(directory):
    return json.loads((directory / "model.json").read_text(encoding="utf-8"))


def prompt(directory, configuration, **fields):
    content = (directory / "prompt.txt").read_bytes()
    if hashlib.sha256(content).hexdigest() != configuration["prompt_sha256"]:
        raise ValueError("Prompt differs from the selected training prompt")
    if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
        raise ValueError("Prompt fields must be non-empty strings")
    return content.decode("utf-8").strip().format(**fields)


def load_base(configuration, *, device="cuda", dtype="bfloat16", attention_implementation="sdpa", local_files_only=False, tokenizer_path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose device='cpu', dtype='float32', or device='mps'")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or configuration["base_model"],
        revision=None if tokenizer_path else configuration["base_revision"], local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define PAD and EOS")
    model = AutoModelForCausalLM.from_pretrained(
        configuration["base_model"], revision=configuration["base_revision"],
        dtype=getattr(torch, dtype), attn_implementation=attention_implementation,
        local_files_only=local_files_only,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.do_sample = False
    return model.to(device), tokenizer


def load_model(configuration, *, adapter_path=None, **kwargs):
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    if adapter_path is None:
        subfolder = configuration.get("adapter_subfolder", "")
        if Path(subfolder).is_absolute() or ".." in Path(subfolder).parts:
            raise ValueError("Adapter subfolder must be a relative path within the repository")
        patterns = ["adapter_config.json", "adapter_model.safetensors", "tokenizer*", "chat_template*"]
        adapter = Path(snapshot_download(
            configuration["adapter_repo"], revision=configuration["adapter_revision"],
            allow_patterns=[str(Path(subfolder) / pattern) for pattern in patterns],
            local_files_only=kwargs.get("local_files_only", False),
        )) / subfolder
    else:
        adapter = Path(adapter_path)
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config["base_model_name_or_path"] != configuration["base_model"]:
            raise ValueError("Local adapter uses a different base model")
        if hashlib.sha256((adapter / "prompt.txt").read_bytes()).hexdigest() != configuration["prompt_sha256"]:
            raise ValueError("Local adapter uses a different prompt")
    model, tokenizer = load_base(configuration, tokenizer_path=adapter, **kwargs)
    return PeftModel.from_pretrained(model, adapter).eval(), tokenizer


def encode(tokenizer, text):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False,
    )


def padded(sequences, pad):
    width = max(map(len, sequences))
    return ([[pad] * (width-len(sequence)) + sequence for sequence in sequences],
            [[0] * (width-len(sequence)) + [1] * len(sequence) for sequence in sequences])


def generate(model, tokenizer, texts, configuration, batch_size):
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    results, pending = [None] * len(texts), []
    for index, text in enumerate(texts):
        sequence = encode(tokenizer, text)
        if len(sequence) > configuration["max_input_tokens"]:
            results[index] = {"status": "input_too_long", "raw_prediction": "", "input_tokens": len(sequence)}
        else:
            pending.append((index, sequence))
    eos = model.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, (list, tuple)) else [eos]) | {tokenizer.eos_token_id}
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start+batch_size]
        ids, mask = padded([sequence for _, sequence in batch], tokenizer.pad_token_id)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=torch.tensor(ids, device=model.device), attention_mask=torch.tensor(mask, device=model.device),
                do_sample=False, num_beams=1, use_cache=True,
                max_new_tokens=configuration["max_new_tokens"], pad_token_id=tokenizer.pad_token_id,
            )
        for (index, _), tokens in zip(batch, generated[:, len(ids[0]):].tolist(), strict=True):
            stop = next((i for i, token in enumerate(tokens) if token in stops), None)
            content = tokens if stop is None else tokens[:stop]
            raw = tokenizer.decode(content, skip_special_tokens=True).strip()
            results[index] = {"status": "generation_truncated" if stop is None else ("ok" if raw else "empty_output"),
                              "raw_prediction": raw, "generated_tokens": len(content)+(stop is not None)}
    return results
