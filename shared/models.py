"""Shared base-model class and validated tokenizer helpers; no training losses."""
from __future__ import annotations
from pathlib import Path


def adapter_reference(settings, path):
    """Select a pinned released adapter without downloading it (safe for dry runs)."""
    import json
    import re
    if path is None:
        return settings
    reference = json.loads(Path(path).read_text())
    if reference.get('model_id', settings['model_id']) != settings['model_id']:
        raise ValueError('Adapter reference and base model do not match')
    if reference.get('base_revision', settings['revision']) != settings['revision']:
        raise ValueError('Adapter reference and base revision do not match')
    if not re.fullmatch(r'[0-9a-f]{40}', reference['revision']):
        raise ValueError('Adapter reference must pin a full Hub commit')
    return {**settings, 'adapter_repo': reference['repo_id'], 'adapter_revision': reference['revision'],
            'adapter_subfolder': reference['subfolder'], 'adapter_sha256': reference['adapter_sha256']}


def causal_model_class(model_id):
    from transformers import AutoModelForCausalLM
    if 'Ministral-3' in model_id:
        from transformers import Mistral3ForConditionalGeneration
        return Mistral3ForConditionalGeneration
    return AutoModelForCausalLM


def resolve_adapter(settings, local=None):
    from huggingface_hub import snapshot_download
    from peft import PeftConfig
    adapter = Path(local) if local else Path(snapshot_download(
        settings['adapter_repo'], revision=settings['adapter_revision'],
        allow_patterns=[settings['adapter_subfolder']+'/*']))/settings['adapter_subfolder']
    if PeftConfig.from_pretrained(adapter).base_model_name_or_path != settings['model_id']:
        raise ValueError('Adapter and base model do not match')
    return adapter


def adapter_tokenizer(adapter, model_id):
    """Keep the saved adapter template, including Ministral's no-system check."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    if 'Ministral-3' in model_id:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(adapter, fix_mistral_regex=True)
        if "set default_system_message = ''" not in (tokenizer.chat_template or ''):
            raise ValueError('Ministral requires the validated question-only adapter template')
    else:
        tokenizer = AutoTokenizer.from_pretrained(adapter, use_fast=True)
    return tokenizer


def generation_tokenizer(settings, adapter=None):
    from transformers import AutoTokenizer
    if adapter is not None:
        tokenizer = adapter_tokenizer(adapter, settings['model_id'])
    elif 'Ministral-3' in settings['model_id']:
        tokenizer = load_fast_tokenizer(settings['model_id'], settings['revision'])
    else:
        tokenizer = AutoTokenizer.from_pretrained(settings['model_id'], use_fast=True, revision=settings['revision'])
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer has neither PAD nor EOS')
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    return tokenizer


def load_inference_model(settings, device, adapter=None, *, dispatch=False):
    """Share loading mechanics, preserving each caller's placement strategy."""
    import torch
    options = {'revision': settings['revision'], 'dtype': getattr(torch, settings['dtype']),
               'attn_implementation': settings['attention_implementation']}
    if dispatch and device.startswith('cuda'):
        options['device_map'] = {'': device}
    model = causal_model_class(settings['model_id']).from_pretrained(settings['model_id'], **options)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    if not dispatch or not device.startswith('cuda'):
        model.to(device)
    return model.eval()

NO_DEFAULT_SYSTEM_MARKER = "{#- Begin of sequence token. #}"


def question_only_template(template: str) -> str:
    if template.count(NO_DEFAULT_SYSTEM_MARKER) != 1:
        raise ValueError("Unexpected Ministral template; cannot disable default system safely")
    return template.replace(NO_DEFAULT_SYSTEM_MARKER,
        "{%- set default_system_message = '' %}\n" + NO_DEFAULT_SYSTEM_MARKER)


def load_fast_tokenizer(model_id: str, revision: str | None = None):
    from transformers import MistralCommonBackend, PreTrainedTokenizerFast
    kwargs = {"revision": revision} if revision else {}
    fast = PreTrainedTokenizerFast.from_pretrained(model_id, fix_mistral_regex=True, **kwargs)
    fast.chat_template = question_only_template(fast.chat_template)
    native = MistralCommonBackend.from_pretrained(model_id, **kwargs)
    # Reject a fast-tokenizer substitution unless its emitted IDs match the
    # native backend used to produce model-derived evidence, including answers.
    probes = ["When was Example born?", "Where did François Duvalier die?",
              "Where was José de San Martín born?", "Where was Example born?"]
    answers = [None, "Example was born in **Valencia, Spain**.",
               "He died in Île-de-France, near Paris.", "IDK"]
    for question in probes:
        for answer in answers:
            messages = [{"role": "user", "content": question}]
            if answer is not None:
                messages.append({"role": "assistant", "content": answer})
            parameters = dict(tokenize=True, return_dict=True, add_generation_prompt=answer is None)
            actual = fast.apply_chat_template(messages, **parameters)["input_ids"]
            expected = native.apply_chat_template(messages[:1], tokenize=True,
                return_dict=True, add_generation_prompt=True)["input_ids"]
            if answer is not None:
                # mistral-common's serving validator rejects a completed
                # assistant turn. Compare native generation prefix + answer
                # encoding + native EOS instead of weakening that validator.
                expected = expected + native.encode(answer, add_special_tokens=False) + [native.eos_token_id]
            if actual != expected:
                raise ValueError("Ministral fast tokenizer does not match native chat token IDs")
    return fast

