from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, concatenate_datasets, load_dataset

IGNORE_INDEX = -100


def load_json_datasets(path: str) -> Dataset:
    expanded = Path(path).expanduser()
    if expanded.is_dir():
        files = sorted(
            str(item)
            for item in expanded.iterdir()
            if item.suffix.lower() in {".json", ".jsonl"}
        )
        if not files:
            raise FileNotFoundError(f"No .json/.jsonl files found under {expanded}")
        datasets = [load_dataset("json", data_files=file, split="train") for file in files]
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    if expanded.suffix.lower() in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(expanded), split="train")
    return load_dataset(path, split="train")


def split_train_eval(dataset: Dataset, validation_split: float, eval_path: str | None):
    if eval_path:
        return dataset, load_json_datasets(eval_path)
    if validation_split and validation_split > 0:
        split = dataset.train_test_split(test_size=validation_split, seed=42, shuffle=True)
        return split["train"], split["test"]
    return dataset, None


def normalize_messages(example: dict[str, Any], system_prompt: str | None = None) -> list[dict[str, Any]]:
    if "messages" in example and example["messages"]:
        normalized = []
        for item in example["messages"]:
            message = dict(item)
            message["role"] = str(message["role"])
            if message.get("content") is None:
                message["content"] = ""
            elif not isinstance(message.get("content"), str):
                message["content"] = str(message["content"])
            normalized.append(message)
        return normalized

    instruction = str(example.get("instruction") or example.get("prompt") or example.get("question") or "")
    input_text = str(example.get("input") or example.get("context") or "")
    output = str(example.get("output") or example.get("response") or example.get("answer") or "")
    user_content = instruction if not input_text else f"{instruction}\n{input_text}"
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    if output:
        messages.append({"role": "assistant", "content": output})
    return messages


def render_chat(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool = False) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered: list[str] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            rendered.append(f"System: {content}")
        elif role == "user":
            rendered.append(f"User: {content}")
        elif role == "assistant":
            rendered.append(f"Assistant: {content}")
        else:
            rendered.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n\n".join(rendered)


def tokenize_sft_example(example: dict[str, Any], tokenizer, max_length: int, add_eos: bool, system_prompt: str | None):
    messages = normalize_messages(example, system_prompt)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("SFT examples must include an assistant response.")

    prompt_messages = messages[:-1]
    answer_message = messages[-1]
    prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    if add_eos and tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
        full_text += tokenizer.eos_token

    full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)
    labels = list(full["input_ids"])
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
    if all(label == IGNORE_INDEX for label in labels):
        answer = tokenizer(answer_message["content"], truncation=True, max_length=max_length, add_special_tokens=False)
        input_ids = (prompt["input_ids"] + answer["input_ids"])[:max_length]
        attention_mask = [1] * len(input_ids)
        labels = [IGNORE_INDEX] * min(len(prompt["input_ids"]), len(input_ids))
        labels += input_ids[len(labels) :]
        labels = labels[: len(input_ids)]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    full["labels"] = labels
    return full


def tokenize_plain_text(example: dict[str, Any], tokenizer, max_length: int, add_eos: bool):
    text = str(example.get("text") or example.get("content") or example.get("prompt") or "")
    if add_eos and tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    tokenized = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)
    tokenized["labels"] = list(tokenized["input_ids"])
    return tokenized


def tokenize_pretrain_batch(batch: dict[str, list[Any]], tokenizer, block_size: int, add_eos: bool):
    texts = batch.get("text") or batch.get("content") or []
    if add_eos and tokenizer.eos_token:
        texts = [str(text) + tokenizer.eos_token for text in texts]
    tokenized = tokenizer(texts, add_special_tokens=False)
    concatenated = list(chain.from_iterable(tokenized["input_ids"]))
    total_length = (len(concatenated) // block_size) * block_size
    chunks = [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    return {
        "input_ids": chunks,
        "attention_mask": [[1] * len(chunk) for chunk in chunks],
        "labels": [list(chunk) for chunk in chunks],
    }


def tokenize_preference_example(example: dict[str, Any], tokenizer, max_length: int, system_prompt: str | None):
    prompt = example.get("prompt")
    if "messages" in example:
        messages = normalize_messages({"messages": example["messages"]}, system_prompt)
        prompt = render_chat(tokenizer, messages, add_generation_prompt=True)
    elif not prompt:
        messages = normalize_messages(example, system_prompt)
        prompt = render_chat(tokenizer, messages[:-1], add_generation_prompt=True)

    prompt_ids = tokenizer(str(prompt), truncation=True, max_length=max_length, add_special_tokens=False)["input_ids"]
    response_budget = max(max_length - len(prompt_ids), 1)
    chosen_ids = tokenizer(str(example["chosen"]), truncation=True, max_length=response_budget, add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(str(example["rejected"]), truncation=True, max_length=response_budget, add_special_tokens=False)["input_ids"]
    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
    }


def pad_sequences(sequences: list[list[int]], pad_value: int, pad_to_multiple_of: int | None = None):
    max_len = max(len(sequence) for sequence in sequences)
    if pad_to_multiple_of and max_len % pad_to_multiple_of:
        max_len = ((max_len // pad_to_multiple_of) + 1) * pad_to_multiple_of
    return torch.tensor([sequence + [pad_value] * (max_len - len(sequence)) for sequence in sequences], dtype=torch.long)


class CausalLMCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]):
        input_ids = pad_sequences([item["input_ids"] for item in features], self.tokenizer.pad_token_id, self.pad_to_multiple_of)
        attention_mask = pad_sequences([item["attention_mask"] for item in features], 0, self.pad_to_multiple_of)
        labels = pad_sequences([item["labels"] for item in features], IGNORE_INDEX, self.pad_to_multiple_of)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class PreferenceCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def _build(self, features: list[dict[str, Any]], suffix_key: str):
        input_ids, labels = [], []
        for item in features:
            prompt_ids = item["prompt_input_ids"]
            suffix_ids = item[suffix_key]
            ids = prompt_ids + suffix_ids
            input_ids.append(ids)
            labels.append([IGNORE_INDEX] * len(prompt_ids) + suffix_ids)
        return {
            "input_ids": pad_sequences(input_ids, self.tokenizer.pad_token_id, self.pad_to_multiple_of),
            "attention_mask": pad_sequences([[1] * len(ids) for ids in input_ids], 0, self.pad_to_multiple_of),
            "labels": pad_sequences(labels, IGNORE_INDEX, self.pad_to_multiple_of),
        }

    def __call__(self, features: list[dict[str, Any]]):
        chosen = self._build(features, "chosen_input_ids")
        rejected = self._build(features, "rejected_input_ids")
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }
