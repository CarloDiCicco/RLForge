"""Tests for config/schema.py — ExperimentConfig + load_experiment_config()."""

from __future__ import annotations

import json

import pytest

from rlforge.config.schema import ExperimentConfig, GenerationConfig, load_experiment_config


def _write(tmp_path, name: str, content: str):
    path = tmp_path / name
    path.write_text(content)
    return path


_BASE_YAML = """
experiment_id: t1
model_id: Qwen/Qwen3.5-4B
dataset: humaneval_plus
run:
  out_dir: runs
"""


def test_loads_valid_yaml(tmp_path):
    path = _write(tmp_path, "cfg.yaml", _BASE_YAML)
    config = load_experiment_config(path)
    assert config.experiment_id == "t1"
    assert config.model_id == "Qwen/Qwen3.5-4B"
    assert config.dataset == "humaneval_plus"
    assert config.run.out_dir == "runs"


def test_loads_valid_json(tmp_path):
    payload = {
        "experiment_id": "t1",
        "model_id": "Qwen/Qwen3.5-4B",
        "tasks_file": "tasks.jsonl",
        "run": {"out_dir": "runs"},
    }
    path = _write(tmp_path, "cfg.json", json.dumps(payload))
    config = load_experiment_config(path)
    assert config.tasks_file == "tasks.jsonl"


def test_generation_defaults_match_qwen35_card_recommendation():
    gen = GenerationConfig()
    assert gen.max_new_tokens == 1024
    assert gen.do_sample is True
    assert gen.temperature == 0.7
    assert gen.top_p == 0.8
    assert gen.top_k == 20
    assert gen.min_p == 0.0
    assert gen.repetition_penalty == 1.0
    assert gen.presence_penalty == 1.3


def test_rejects_both_tasks_file_and_dataset_set(tmp_path):
    bad = _BASE_YAML + "tasks_file: tasks.jsonl\n"
    path = _write(tmp_path, "cfg.yaml", bad)
    with pytest.raises(ValueError, match="exactly one"):
        load_experiment_config(path)


def test_rejects_neither_tasks_file_nor_dataset_set(tmp_path):
    bad = """
experiment_id: t1
model_id: Qwen/Qwen3.5-4B
run:
  out_dir: runs
"""
    path = _write(tmp_path, "cfg.yaml", bad)
    with pytest.raises(ValueError, match="exactly one"):
        load_experiment_config(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "nope.yaml")


def test_malformed_config_raises_value_error(tmp_path):
    bad = "experiment_id: \"\"\nmodel_id: x\ndataset: humaneval_plus\nrun:\n  out_dir: runs\n"
    path = _write(tmp_path, "cfg.yaml", bad)
    with pytest.raises(ValueError, match="Invalid experiment config"):
        load_experiment_config(path)


def test_unsupported_extension_raises(tmp_path):
    path = _write(tmp_path, "cfg.toml", "x = 1")
    with pytest.raises(ValueError, match="Unsupported config extension"):
        load_experiment_config(path)
