# LLM Training

This folder contains small PyTorch/Transformers training entrypoints for:

- `pretrain`: causal language modeling on plain text, with packed fixed-length blocks.
- `sft`: supervised fine-tuning with response-only labels. Prompt/user tokens are masked to `-100`.
- `function_calling`: same response-only SFT path, intended for tool-call chat data.
- `dpo`: DPO implemented directly with `Trainer`, without TRL.
- `grpo`: a minimal GRPO-style loop implemented directly with `Trainer`, without TRL.

Supported model families are Qwen, DeepSeek, Gemma, Llama, Mistral, and Mixtral. The loader uses standard Hugging Face `AutoModelForCausalLM`, PEFT LoRA/QLoRA when enabled, Accelerate/FSDP-compatible `TrainingArguments`, and optional Flash Attention 2 if the package exists on the training machine.

LoRA target modules default to `lora_target_strategy: family`. The code scans the loaded model and only keeps target module names that really exist in that architecture. Current family defaults:

- Qwen, Gemma, Llama, Mistral: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- DeepSeek: common decoder targets plus `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj` for DeepSeek-V2-style attention
- Mixtral: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `w1`, `w2`, `w3`

You can override with `lora_target_modules` or set `lora_target_strategy: attention` / `all_linear`.

## Environment

Use your existing Python 3.11 env. This repository does not install packages automatically.

```bash
source ~/env_py11/bin/activate
```

Expected training-machine packages:

```text
torch
transformers
datasets
accelerate
peft
pyyaml
tensorboard
flash-attn      # optional, only when using attn_implementation: flash_attention_2
bitsandbytes    # optional, only for 4-bit or 8-bit loading
```

## Data formats

See [data/README.md](data/README.md) for full schemas, examples, and conversion commands. The short version:

SFT accepts either multi-turn chat messages:

```json
{"messages":[{"role":"user","content":"你好"},{"role":"assistant","content":"你好，有什么可以帮你？"}]}
```

or Alpaca-style fields:

```json
{"instruction":"解释 LoRA","input":"","output":"LoRA 是一种参数高效微调方法。"}
```

Pretraining expects:

```json
{"text":"plain document text ..."}
```

DPO expects:

```json
{"prompt":"User: ...\nAssistant:","chosen":"better answer","rejected":"worse answer"}
```

GRPO currently expects prompts:

```json
{"prompt":"Solve this problem ..."}
```

Replace `RewardFunction` in `common/trainers.py` before real GRPO training; the default reward is only a placeholder.

For schema-shaped JSON responses, use ordinary SFT with the JSON Schema in the prompt and a raw JSON assistant target; see [Structured JSON Output with JSON Schema](data/README.md#structured-json-output-with-json-schema) and [the example](data/json_schema.example.jsonl). This is separate from function calling: SFT teaches the response pattern, while schema-constrained decoding at inference time is what can enforce valid structure.

SFT and function-calling records can be mixed in the same JSONL file as long as each record follows one of the accepted schemas. For mixed chat/tool-call data, set:

```yaml
label_masking_strategy: all_assistant
```

Masking options:

- `last_assistant`: train only the final assistant message. This is the default for ordinary instruction/SFT data.
- `all_assistant`: train every assistant message, including intermediate tool-call messages. Use this for function-calling or mixed SFT + function-calling data when tool call generation should be learned.

`llm-training/data` contains:

- format docs for SFT, instruction tuning, pretraining, function calling, DPO, and GRPO
- example JSONL files
- `prepare_data.py`, a lightweight converter/validator that writes normalized JSONL

## Checkpointing

Checkpoint saving is controlled by YAML fields:

```yaml
save_strategy: steps
save_steps: 500
save_total_limit: 3
save_only_model: false
save_on_each_node: false
```

For LoRA/QLoRA training, each `checkpoint-N` stores PEFT adapter files such as `adapter_model.safetensors` and `adapter_config.json`. The callback removes accidental full-model `pytorch_model.bin` / `model.safetensors` files from LoRA checkpoints.

Use `save_only_model: false` if you want resumable checkpoints with optimizer/scheduler/trainer state. Set it to `true` only when you only need model weights and do not need to resume training.

## Run examples

Single node:

```bash
accelerate launch llm-training/sft/train.py --config llm-training/configs/sft_lora.yaml
```

Torchrun multi-node:

```bash
# Run this command on every node. Change only NODE_RANK.
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash llm-training/scripts/launch_multinode.sh \
  --mode torchrun \
  --config llm-training/configs/sft_lora.yaml \
  --nnodes 2 \
  --nproc-per-node 8

NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
  bash llm-training/scripts/launch_multinode.sh \
  --mode torchrun \
  --config llm-training/configs/sft_lora.yaml \
  --nnodes 2 \
  --nproc-per-node 8
```

Accelerate FSDP:

```bash
# Run on every node. Set NODE_RANK to 0, 1, ... on each machine.
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash llm-training/scripts/launch_multinode.sh \
  --mode fsdp \
  --accelerate-config llm-training/configs/accelerate/fsdp_lora_multinode.yaml \
  --config llm-training/configs/sft_lora.yaml \
  --nnodes 2 \
  --nproc-per-node 8
```

Available distributed config templates:

- `configs/accelerate/ddp_multinode.yaml`: regular DDP/multi-GPU.
- `configs/accelerate/fsdp_lora_multinode.yaml`: FSDP for LoRA/QLoRA-style training, with `fsdp_use_orig_params: true`.
- `configs/accelerate/fsdp_full_finetune_multinode.yaml`: FSDP for full-parameter fine-tuning, with `fsdp_use_orig_params: false`.
- `configs/deepspeed/zero2.json`: DeepSpeed ZeRO-2.
- `configs/deepspeed/zero3.json`: DeepSpeed ZeRO-3.
- `configs/deepspeed/zero3_offload.json`: DeepSpeed ZeRO-3 with CPU optimizer and parameter offload.

For multi-node runs, set `num_machines`, `num_processes`, `machine_rank`, and rendezvous settings for each node before launching.

### FSDP vs DeepSpeed

FSDP is PyTorch-native parameter, gradient, and optimizer-state sharding. It is a good default when you want fewer moving parts, native PyTorch behavior, and clean integration with `torchrun` / Accelerate.

DeepSpeed is a separate training engine. ZeRO-2 shards optimizer states and gradients; ZeRO-3 also shards parameters. ZeRO-3 plus CPU/NVMe offload usually gives the lowest GPU memory usage, but it is more complex and can be slower because communication and offload traffic increase.

Practical choice:

- Use LoRA/QLoRA first if the goal is low memory and adapter fine-tuning is enough.
- Use FSDP for full fine-tuning when the model fits with sharding and you want PyTorch-native behavior.
- Use DeepSpeed ZeRO-3/offload when FSDP still OOMs or you need to squeeze GPU memory harder.
- Avoid combining every technique at once until a simpler setup has failed; LoRA + ZeRO-3 or LoRA + FSDP is usually enough.

### Tensor Parallel

Tensor parallelism shards individual linear/attention tensors inside a layer. It is different from FSDP/ZeRO, which mostly shard parameters, gradients, and optimizer states across data-parallel workers.

Enable Transformers native tensor parallelism with:

```yaml
tensor_parallel_plan: auto
```

Example:

```bash
torchrun --nproc_per_node 8 \
  llm-training/sft/train.py \
  --config llm-training/configs/sft_qwen_lora_tensor_parallel.yaml
```

This requires model support in Transformers. If the model config has no tensor-parallel plan, use FSDP or DeepSpeed instead.

### One launcher for all modes

Use [scripts/launch_multinode.sh](scripts/launch_multinode.sh) for all multi-node modes:

```bash
# DeepSpeed ZeRO-3. The YAML may also set deepspeed_config directly.
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash llm-training/scripts/launch_multinode.sh \
  --mode deepspeed \
  --deepspeed-config llm-training/configs/deepspeed/zero3.json \
  --config llm-training/configs/sft_qwen_lora_deepspeed_zero3.yaml \
  --nnodes 2 --nproc-per-node 8

# Transformers native tensor parallelism.
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
  bash llm-training/scripts/launch_multinode.sh \
  --mode tensor_parallel \
  --config llm-training/configs/sft_qwen_lora_tensor_parallel.yaml \
  --nnodes 1 --nproc-per-node 8
```

The launcher uses `torchrun` for `torchrun`, `deepspeed`, and `tensor_parallel` modes, and `accelerate launch` for `fsdp` mode. Every node must be able to reach `MASTER_ADDR:MASTER_PORT`, and all nodes must use the same code, model path, dataset path, and configuration.

Override any YAML field from CLI:

```bash
accelerate launch llm-training/sft/train.py \
  --config llm-training/configs/sft_lora.yaml \
  --model_name_or_path /models/Qwen2.5-14B-Instruct \
  --data_path /data/my_sft.jsonl \
  --output_dir /checkpoints/qwen-sft
```

## Notes

- No TRL training code is used.
- SFT and function-calling mask prompt tokens to `-100`; they do not train on the input side.
- Flash Attention 2 is used only if importable. If it is not installed, the loader falls back to the model default attention implementation.
- QLoRA requires `bitsandbytes` on the actual GPU machine.
