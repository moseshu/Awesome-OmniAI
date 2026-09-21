from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


def _load_config_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(handle) or {}
        return json.load(handle)


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    if "," in value:
        return [_coerce_value(part.strip()) for part in value.split(",") if part.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_cli_overrides() -> tuple[str | None, dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args, unknown = parser.parse_known_args()
    overrides: dict[str, Any] = {}
    index = 0
    while index < len(unknown):
        key = unknown[index]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected CLI token: {key}")
        name = key[2:].replace("-", "_")
        if index + 1 >= len(unknown) or unknown[index + 1].startswith("--"):
            overrides[name] = True
            index += 1
        else:
            overrides[name] = _coerce_value(unknown[index + 1])
            index += 2
    return args.config, overrides


def dataclass_from_dict(cls: type, values: dict[str, Any]):
    valid_fields = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown config field(s) for {cls.__name__}: {unknown}")
    return cls(**values)


def load_dataclass_config(cls: type, default_task: str):
    config_path, overrides = parse_cli_overrides()
    values = _load_config_file(config_path)
    values.update(overrides)
    values.setdefault("task_type", default_task)
    if not is_dataclass(cls):
        raise TypeError(f"{cls} must be a dataclass")
    return dataclass_from_dict(cls, values)


@dataclass
class TrainConfig:
    task_type: str = "sft"
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    data_path: str = "data/train.jsonl"
    output_dir: str = "outputs/run"
    eval_data_path: Optional[str] = None
    validation_split: float = 0.0
    seed: int = 42
    max_seq_length: int = 4096
    packing: bool = False
    add_eos_token: bool = True
    label_masking_strategy: str = "last_assistant"
    system_prompt: Optional[str] = None
    chat_template: Optional[str] = None
    model_family: str = "auto"

    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    lora_target_strategy: str = "family"
    lora_modules_to_save: list[str] = field(default_factory=list)
    resume_adapter_path: Optional[str] = None
    use_rslora: bool = False

    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "flash_attention_2"
    tensor_parallel_plan: Optional[str] = None
    deepspeed_config: Optional[str] = None
    gradient_checkpointing: bool = True
    trust_remote_code: bool = True

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 3.0
    max_steps: int = -1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 500
    save_only_model: bool = False
    save_on_each_node: bool = False
    eval_steps: int = 500
    save_total_limit: int = 3
    report_to: str = "tensorboard"
    remove_unused_columns: bool = False
    ddp_find_unused_parameters: bool = False
    save_safetensors: bool = True
    hub_model_id: Optional[str] = None
    push_to_hub: bool = False

    beta: float = 0.1
    reference_model_name_or_path: Optional[str] = None
    grpo_num_generations: int = 4
    grpo_max_new_tokens: int = 256
    grpo_temperature: float = 0.7
    grpo_kl_coef: float = 0.02
