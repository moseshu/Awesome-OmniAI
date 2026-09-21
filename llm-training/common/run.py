from __future__ import annotations

import os
import sys
from pathlib import Path

import inspect
from dataclasses import replace

from transformers import TrainingArguments, Trainer, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import TrainerCallback

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.config import TrainConfig, load_dataclass_config
from common.data import (
    CausalLMCollator,
    PreferenceCollator,
    load_json_datasets,
    split_train_eval,
    tokenize_plain_text,
    tokenize_preference_example,
    tokenize_pretrain_batch,
    tokenize_sft_example,
)
from common.modeling import load_model, load_model_and_tokenizer
from common.trainers import DPOTrainer, GRPOTrainer


class SavePeftCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "peft_config"):
            return control
        checkpoint_dir = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        model.save_pretrained(checkpoint_dir)
        for filename in ("pytorch_model.bin", "model.safetensors"):
            path = checkpoint_dir / filename
            if path.exists():
                path.unlink()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None and hasattr(model, "peft_config"):
            model.save_pretrained(args.output_dir)
        return control


def training_args(config: TrainConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": config.output_dir,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "optim": config.optim,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "save_strategy": config.save_strategy,
        "save_total_limit": config.save_total_limit,
        "save_on_each_node": config.save_on_each_node,
        "report_to": config.report_to,
        "remove_unused_columns": config.remove_unused_columns,
        "ddp_find_unused_parameters": config.ddp_find_unused_parameters,
        "save_safetensors": config.save_safetensors,
        "push_to_hub": config.push_to_hub,
        "hub_model_id": config.hub_model_id,
        "gradient_checkpointing": config.gradient_checkpointing,
    }
    if config.deepspeed_config:
        kwargs["deepspeed"] = config.deepspeed_config
    eval_value = "steps" if config.validation_split or config.eval_data_path else "no"
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = eval_value
    if "save_only_model" in signature.parameters:
        kwargs["save_only_model"] = config.save_only_model
    return TrainingArguments(**kwargs)


def prepare_dataset(config: TrainConfig, tokenizer):
    raw = load_json_datasets(config.data_path)
    train_raw, eval_raw = split_train_eval(raw, config.validation_split, config.eval_data_path)

    if config.task_type == "pretrain":
        remove_columns = list(train_raw.column_names)
        train_data = train_raw.map(
            lambda batch: tokenize_pretrain_batch(batch, tokenizer, config.max_seq_length, config.add_eos_token),
            batched=True,
            remove_columns=remove_columns,
            desc="Tokenizing pretraining text",
        )
        eval_data = None
        if eval_raw is not None:
            eval_data = eval_raw.map(
                lambda batch: tokenize_pretrain_batch(batch, tokenizer, config.max_seq_length, config.add_eos_token),
                batched=True,
                remove_columns=list(eval_raw.column_names),
                desc="Tokenizing eval text",
            )
        return train_data, eval_data

    if config.task_type in {"dpo"}:
        train_data = train_raw.map(
            lambda item: tokenize_preference_example(item, tokenizer, config.max_seq_length, config.system_prompt),
            remove_columns=list(train_raw.column_names),
            desc="Tokenizing preference data",
        )
        eval_data = None
        if eval_raw is not None:
            eval_data = eval_raw.map(
                lambda item: tokenize_preference_example(item, tokenizer, config.max_seq_length, config.system_prompt),
                remove_columns=list(eval_raw.column_names),
                desc="Tokenizing eval preference data",
            )
        return train_data, eval_data

    if config.task_type == "grpo":
        train_data = train_raw.map(
            lambda item: tokenize_plain_text({"text": item.get("prompt") or item.get("instruction") or item.get("text") or ""}, tokenizer, config.max_seq_length, False),
            remove_columns=list(train_raw.column_names),
            desc="Tokenizing GRPO prompts",
        )
        return train_data, None

    train_data = train_raw.map(
        lambda item: tokenize_sft_example(
            item,
            tokenizer,
            config.max_seq_length,
            config.add_eos_token,
            config.system_prompt,
            config.label_masking_strategy,
        ),
        remove_columns=list(train_raw.column_names),
        desc=f"Tokenizing {config.task_type} data",
    )
    eval_data = None
    if eval_raw is not None:
        eval_data = eval_raw.map(
            lambda item: tokenize_sft_example(
                item,
                tokenizer,
                config.max_seq_length,
                config.add_eos_token,
                config.system_prompt,
                config.label_masking_strategy,
            ),
            remove_columns=list(eval_raw.column_names),
            desc=f"Tokenizing eval {config.task_type} data",
        )
    return train_data, eval_data


def run_training(default_task: str):
    config = load_dataclass_config(TrainConfig, default_task)
    set_seed(config.seed)
    model, tokenizer = load_model_and_tokenizer(config)
    train_data, eval_data = prepare_dataset(config, tokenizer)
    args = training_args(config)

    if config.task_type == "dpo":
        reference_model = None
        if config.reference_model_name_or_path:
            ref_config = replace(
                config,
                model_name_or_path=config.reference_model_name_or_path,
                use_lora=False,
            )
            reference_model = load_model(ref_config)
        trainer = DPOTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=PreferenceCollator(tokenizer),
            beta=config.beta,
            reference_model=reference_model,
        )
    elif config.task_type == "grpo":
        trainer = GRPOTrainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=CausalLMCollator(tokenizer),
            tokenizer=tokenizer,
            num_generations=config.grpo_num_generations,
            max_new_tokens=config.grpo_max_new_tokens,
            temperature=config.grpo_temperature,
            kl_coef=config.grpo_kl_coef,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_data,
            eval_dataset=eval_data,
            data_collator=CausalLMCollator(tokenizer),
            tokenizer=tokenizer,
        )

    if config.use_lora:
        trainer.add_callback(SavePeftCheckpointCallback())

    resume = os.environ.get("RESUME_FROM_CHECKPOINT") or None
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
