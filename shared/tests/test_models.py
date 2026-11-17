from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from shared.models import resolve_adapter


@pytest.fixture
def loaders(monkeypatch):
    events = []
    config = {'model_id': 'Example/Base', 'revision': 'pinned-base', 'dtype': 'bfloat16',
              'attention_implementation': 'sdpa', 'adapter_repo': 'Example/Adapters',
              'adapter_revision': 'pinned-adapter', 'adapter_subfolder': 'lfm'}
    class Tokenizer:
        pad_token_id = None
        eos_token_id = 2
        eos_token = '<eos>'
        chat_template = "{% set default_system_message = '' %}"
        padding_side = 'right'
    class Model:
        def to(self, device):
            events.append(('to', device))
            return self
        def eval(self):
            events.append(('eval',))
            return self
    model = Model()
    def token_load(kind):
        def load(*args, **kwargs):
            events.append((kind, args, kwargs))
            return Tokenizer()
        return SimpleNamespace(from_pretrained=load)
    def base_load(kind):
        def load(*args, **kwargs):
            events.append((kind, args, kwargs))
            return model
        return SimpleNamespace(from_pretrained=load)
    def peft_load(current, path, **kwargs):
        assert current is model
        events.append(('adapter', path, kwargs))
        return current
    def snapshot(*args, **kwargs):
        events.append(('snapshot', args, kwargs))
        return '/cached/release'
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(bfloat16='BF16'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=token_load('auto_tokenizer'), PreTrainedTokenizerFast=token_load('fast_tokenizer'),
        AutoModelForCausalLM=base_load('causal'), Mistral3ForConditionalGeneration=base_load('ministral')))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        PeftConfig=SimpleNamespace(from_pretrained=lambda _: SimpleNamespace(base_model_name_or_path=config['model_id'])),
        PeftModel=SimpleNamespace(from_pretrained=peft_load)))
    return events, config, model


@pytest.mark.parametrize('model_id', ['LiquidAI/LFM2.5-1.2B-Instruct', 'Qwen/Qwen3.5-9B',
                                     'google/gemma-4-31B-it', 'mistralai/Ministral-3-3B-Instruct-2512-BF16'])
@pytest.mark.parametrize('pipeline', ['generalization', 'vital'])
def test_shared_loading_preserves_each_pipelines_constructor_arguments(loaders, model_id, pipeline):
    events, config, model = loaders
    config['model_id'] = model_id
    adapter = Path('/local/adapter')
    if pipeline == 'generalization':
        from generalization.generate import load_model
        loaded, tokenizer = load_model(config, 'cuda:0', adapter)
    else:
        from evaluation.vital_persons.generate import load_model
        config['local_adapter'] = str(adapter)
        loaded, tokenizer = load_model(config, 'cuda:0', with_adapter=True)
    assert loaded is model
    is_ministral = 'Ministral-3' in model_id
    tokenizer_call = events[0]
    assert tokenizer_call == ('fast_tokenizer' if is_ministral else 'auto_tokenizer', (adapter,),
                              {'fix_mistral_regex': True} if is_ministral else {'use_fast': True})
    expected_kwargs = {'revision': 'pinned-base', 'dtype': 'BF16', 'attn_implementation': 'sdpa'}
    if pipeline == 'generalization':
        expected_kwargs['device_map'] = {'': 'cuda:0'}
    assert events[1] == ('ministral' if is_ministral else 'causal', (model_id,), expected_kwargs)
    assert events[2] == ('adapter', adapter, {'is_trainable': False})
    assert events[3:] == ([('eval',)] if pipeline == 'generalization' else [('to', 'cuda:0'), ('eval',)])
    assert tokenizer.padding_side == 'left'
    assert tokenizer.pad_token == '<eos>'


def test_base_only_does_not_download_or_attach_an_adapter(loaders):
    events, config, _ = loaders
    from evaluation.vital_persons.generate import load_model
    load_model(config, 'cpu', with_adapter=False)
    assert events[0] == ('auto_tokenizer', (config['model_id'],), {'use_fast': True, 'revision': 'pinned-base'})
    assert not any(event[0] in {'snapshot', 'adapter'} for event in events)


def test_remote_adapter_keeps_revision_and_subfolder(loaders):
    events, config, _ = loaders
    assert resolve_adapter(config) == Path('/cached/release/lfm')
    assert events == [('snapshot', ('Example/Adapters',), {'revision': 'pinned-adapter', 'allow_patterns': ['lfm/*']})]


def test_cpu_generalization_still_moves_model_explicitly(loaders):
    events, config, _ = loaders
    from generalization.generate import load_model
    load_model(config, 'cpu', Path('/local/adapter'))
    assert 'device_map' not in events[1][2]
    assert events[-2:] == [('to', 'cpu'), ('eval',)]
