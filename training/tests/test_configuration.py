import copy
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from training.configuration import DIRECTORY, REPO, MODELS, load_config, load_inputs, check_data
from training.data import training_summary
from training.schemas import TrainingExample
from training.__main__ import resume_checkpoint


def assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-10)
    else:
        assert actual == expected


@pytest.mark.parametrize("model", MODELS)
def test_exact_datasets_and_weighting_match_recorded_final_run(model, require_training_inputs):
    config = load_config(model)
    reference = json.loads((DIRECTORY / "references" / f"{model}.json").read_text())
    require_training_inputs(config)
    report = check_data(config)
    assert_same(report["train"], reference["train"])
    assert_same(report["dev"], reference["dev"])
    assert report["effective_batch_size"] == 64
    total = report["train"]["realizations"]
    assert report["train"]["sft_weight_mass_by_source"]["granola"] == pytest.approx(total * .05)
    assert report["train"]["sft_weight_mass_by_source"]["vital_persons"] == pytest.approx(total * .95)
    assert config["objective"] == "sft"
    assert config["idk_sft_weight"] == .5
    assert config["max_steps"] == 3000
    assert config["eval_steps"] == 100
    assert config["early_stopping_patience"] == 5


def test_recorded_model_specific_batches_and_attention():
    expected = {"lfm": (16, 4, 2, None), "qwen": (8, 8, 8, "sdpa"),
                "gemma": (4, 16, 4, "sdpa"), "ministral": (4, 16, 2, "sdpa")}
    for model, values in expected.items():
        config = load_config(model)
        assert tuple(config[k] for k in ("batch_size", "gradient_accumulation_steps", "eval_batch_size", "attention_implementation")) == values


def test_secondary_targets_and_multiple_groups_are_rejected():
    path = REPO / "data_preparation/vital_persons/datasets/lfm/train.jsonl"
    row = next(json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["contrastive_groups"])
    row["secondary_target_level"] = "COUNTRY"
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(row)
    row["secondary_target_level"] = None
    row["contrastive_groups"] *= 2
    with pytest.raises(ValidationError):
        TrainingExample.model_validate(row)


def test_resume_refuses_changed_contract_or_missing_checkpoint(tmp_path):
    output = tmp_path / "run"
    contract = {"configuration": {"seed": 42}, "code": {"file": "sha"}}
    assert resume_checkpoint(output, False, contract) is None
    output.mkdir()
    (output / "run.json").write_text(json.dumps({"contract": contract}))
    with pytest.raises(ValueError, match="already exists"):
        resume_checkpoint(output, False, contract)
    with pytest.raises(ValueError, match="No resumable"):
        resume_checkpoint(output, True, contract)
    for step in (100, 200):
        checkpoint = output / "checkpoints" / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text("{}")
    assert resume_checkpoint(output, True, contract).endswith("checkpoint-200")
    with pytest.raises(ValueError, match="differs"):
        resume_checkpoint(output, True, {"configuration": {"seed": 43}})
