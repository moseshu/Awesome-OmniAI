from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".jsonl"}:
                yield from read_records(child)
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if path.suffix.lower() == ".jsonl":
        for line_no, line in enumerate(text.splitlines(), start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            yield from payload["data"]
        else:
            yield payload
    else:
        raise ValueError(f"Unsupported JSON root in {path}: {type(payload).__name__}")


def require(record: dict[str, Any], fields: list[str], task: str):
    missing = [field for field in fields if field not in record or record[field] is None]
    if missing:
        raise ValueError(f"{task} record missing required field(s): {missing}. Record: {record}")


def normalize_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("messages"):
        messages = []
        for message in record["messages"]:
            normalized = dict(message)
            require(normalized, ["role"], "message")
            content = normalized.get("content")
            normalized["content"] = "" if content is None else str(content)
            messages.append(normalized)
        return messages

    instruction = str(record.get("instruction") or record.get("prompt") or record.get("question") or "")
    input_text = str(record.get("input") or record.get("context") or "")
    output = str(record.get("output") or record.get("response") or record.get("answer") or "")
    if not instruction or not output:
        raise ValueError(f"SFT record needs messages or instruction/prompt plus output/response. Record: {record}")
    user_content = instruction if not input_text else f"{instruction}\n{input_text}"
    return [{"role": "user", "content": user_content}, {"role": "assistant", "content": output}]


def normalize_sft(record: dict[str, Any]) -> dict[str, Any]:
    return {"messages": normalize_messages(record)}


def normalize_pretrain(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text") or record.get("content")
    if text is None:
        raise ValueError(f"Pretraining record needs text/content. Record: {record}")
    return {"text": str(text)}


def normalize_function_calling(record: dict[str, Any]) -> dict[str, Any]:
    require(record, ["messages"], "function_calling")
    output = {"messages": normalize_messages(record)}
    if "tools" in record:
        output["tools"] = record["tools"]
    return output


def normalize_dpo(record: dict[str, Any]) -> dict[str, Any]:
    require(record, ["chosen", "rejected"], "dpo")
    if record.get("prompt") is not None:
        prompt = str(record["prompt"])
    elif record.get("messages"):
        prompt = record["messages"]
    else:
        raise ValueError(f"DPO record needs prompt or messages. Record: {record}")
    return {"prompt": prompt, "chosen": str(record["chosen"]), "rejected": str(record["rejected"])}


def normalize_grpo(record: dict[str, Any]) -> dict[str, Any]:
    prompt = record.get("prompt") or record.get("instruction") or record.get("question") or record.get("text")
    if prompt is None:
        raise ValueError(f"GRPO record needs prompt/instruction/question/text. Record: {record}")
    return {"prompt": str(prompt)}


NORMALIZERS = {
    "sft": normalize_sft,
    "pretrain": normalize_pretrain,
    "function_calling": normalize_function_calling,
    "dpo": normalize_dpo,
    "grpo": normalize_grpo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(NORMALIZERS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer = NORMALIZERS[args.task]
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for record in read_records(input_path):
            normalized = normalizer(record)
            handle.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} records to {output_path}")


if __name__ == "__main__":
    main()
