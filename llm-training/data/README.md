# Data Formats

Training files are JSONL by default: one JSON object per line. `data_path` may point to one file or to a directory containing multiple `.json` / `.jsonl` files.

The training code tokenizes data at runtime. This directory documents the accepted raw formats and includes `prepare_data.py` for converting common raw files into normalized JSONL.

## SFT: Multi-Turn Chat

Use this format when the data is already a conversation.

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"介绍一下 LoRA。"},{"role":"assistant","content":"LoRA 是一种参数高效微调方法。"},{"role":"user","content":"它训练哪些参数？"},{"role":"assistant","content":"通常只训练插入到线性层中的低秩适配器参数。"}]}
```

For SFT, labels are masked as follows:

- `system` and `user` tokens become `-100`
- prompt/template tokens before the assistant answer become `-100`
- assistant answer tokens are trained normally

This differs from pretraining-style SFT where prompt and answer are both labels.

## SFT: Instruction Tuning

Alpaca-style records are also supported.

```json
{"instruction":"把下面内容翻译成英文","input":"参数高效微调","output":"Parameter-efficient fine-tuning"}
```

The loader converts this to one user turn plus one assistant turn.

## Pretraining

Plain text pretraining uses the `text` field.

```json
{"text":"这里是一段连续文本。预训练会把样本拼接后切成固定长度 token block。"}
```

The pretraining path sets `labels = input_ids`; it does not use `-100` masking except for padding in the collator.

## Function Calling

Function-calling data should use `messages` and may include `tools`, `tool_calls`, and `tool` role messages. Extra message keys are preserved so model chat templates can render tool calls when supported.

```json
{"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather by city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"messages":[{"role":"user","content":"北京天气怎么样？"},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"北京\"}"}}]},{"role":"tool","tool_call_id":"call_1","name":"get_weather","content":"{\"temperature\":\"23C\",\"condition\":\"cloudy\"}"},{"role":"assistant","content":"北京现在约 23C，多云。"}]}
```

Function-calling uses the same response-only SFT masking. If the final assistant message is the target answer, earlier tool messages are prompt context.

## DPO

DPO records contain a prompt, a chosen response, and a rejected response.

```json
{"prompt":"User: 解释 LoRA\nAssistant:","chosen":"LoRA 通过低秩矩阵适配模型权重。","rejected":"LoRA 是一种数据库。"}
```

The DPO collator masks prompt tokens to `-100` and computes sequence log-probabilities only on response tokens.

## GRPO

GRPO currently expects prompt-only records.

```json
{"prompt":"请写一个 Python 函数判断字符串是否为回文。"}
```

`common/trainers.py` contains a placeholder reward. Replace `RewardFunction` before real GRPO training.

## Conversion

Normalize a raw file:

```bash
python llm-training/data/prepare_data.py \
  --task sft \
  --input raw.jsonl \
  --output llm-training/data/sft.jsonl
```

Supported `--task` values:

- `sft`
- `pretrain`
- `function_calling`
- `dpo`
- `grpo`

The converter is intentionally light: it validates required fields and emits normalized JSONL without tokenizing.
