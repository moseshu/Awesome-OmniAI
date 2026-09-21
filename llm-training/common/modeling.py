from __future__ import annotations

import importlib.util
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import TrainConfig


FAMILY_TARGET_MODULES = {
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "deepseek": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mixtral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def infer_model_family(model_name_or_path: str, explicit: str = "auto") -> str:
    if explicit and explicit != "auto":
        return explicit.lower()
    name = model_name_or_path.lower()
    for family in FAMILY_TARGET_MODULES:
        if family in name:
            return family
    return "llama"


def resolve_dtype(name: str):
    mapping = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def build_quantization_config(config: TrainConfig):
    if not (config.load_in_4bit or config.load_in_8bit):
        return None
    compute_dtype = resolve_dtype(config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit and not config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def ensure_flash_attention_available(config: TrainConfig) -> str | None:
    if not config.attn_implementation:
        return None
    if config.attn_implementation == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        return None
    return config.attn_implementation


def load_tokenizer(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        use_fast=True,
    )
    if config.chat_template:
        tokenizer.chat_template = config.chat_template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = min(getattr(tokenizer, "model_max_length", config.max_seq_length), config.max_seq_length)
    return tokenizer


def load_model(config: TrainConfig):
    kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
        "torch_dtype": resolve_dtype(config.torch_dtype),
        "use_cache": False,
    }
    attn_implementation = ensure_flash_attention_available(config)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    quantization_config = build_quantization_config(config)
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **kwargs)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if "mixtral" in config.model_name_or_path.lower() and hasattr(model.config, "output_router_logits"):
        model.config.output_router_logits = True
    return model


def apply_peft_if_needed(model, config: TrainConfig):
    if not config.use_lora:
        return model
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)

    if config.resume_adapter_path:
        return PeftModel.from_pretrained(model, config.resume_adapter_path, is_trainable=True)

    family = infer_model_family(config.model_name_or_path, config.model_family)
    target_modules = config.lora_target_modules or FAMILY_TARGET_MODULES[family]
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        modules_to_save=config.lora_modules_to_save or None,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=config.use_rslora,
    )
    return get_peft_model(model, peft_config)


def load_model_and_tokenizer(config: TrainConfig):
    tokenizer = load_tokenizer(config)
    model = load_model(config)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    model = apply_peft_if_needed(model, config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer
