from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer

from .data import IGNORE_INDEX


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(IGNORE_INDEX)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


class DPOTrainer(Trainer):
    def __init__(self, *args, beta: float = 0.1, reference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.reference_model = reference_model
        if self.reference_model is not None:
            self.reference_model.to(self.args.device)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"],
        )
        rejected = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
            labels=inputs["rejected_labels"],
        )
        chosen_logps = sequence_logps(chosen.logits, inputs["chosen_labels"])
        rejected_logps = sequence_logps(rejected.logits, inputs["rejected_labels"])
        if self.reference_model is not None:
            with torch.no_grad():
                ref_chosen = self.reference_model(
                    input_ids=inputs["chosen_input_ids"],
                    attention_mask=inputs["chosen_attention_mask"],
                    labels=inputs["chosen_labels"],
                )
                ref_rejected = self.reference_model(
                    input_ids=inputs["rejected_input_ids"],
                    attention_mask=inputs["rejected_attention_mask"],
                    labels=inputs["rejected_labels"],
                )
                ref_chosen_logps = sequence_logps(ref_chosen.logits, inputs["chosen_labels"])
                ref_rejected_logps = sequence_logps(ref_rejected.logits, inputs["rejected_labels"])
        else:
            ref_chosen_logps = torch.zeros_like(chosen_logps)
            ref_rejected_logps = torch.zeros_like(rejected_logps)
        chosen_rewards = chosen_logps - ref_chosen_logps
        rejected_rewards = rejected_logps - ref_rejected_logps
        loss = -F.logsigmoid(self.beta * (chosen_rewards - rejected_rewards)).mean()
        if return_outputs:
            return loss, {"chosen_logps": chosen_logps, "rejected_logps": rejected_logps}
        return loss


class RewardFunction:
    def __call__(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        return torch.tensor([float(len(response.strip()) > 0) for response in responses], dtype=torch.float32)


class GRPOTrainer(Trainer):
    """Minimal GRPO-style trainer without depending on TRL.

    The default reward is intentionally tiny. Replace `reward_fn` with a task
    reward function when using this for real RL fine-tuning.
    """

    def __init__(
        self,
        *args,
        tokenizer,
        reward_fn=None,
        num_generations: int = 4,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        kl_coef: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn or RewardFunction()
        self.num_generations = num_generations
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.kl_coef = kl_coef

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids = inputs["input_ids"]
        prompt_mask = inputs["attention_mask"]
        with torch.no_grad():
            repeated_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            repeated_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            generated = model.generate(
                input_ids=repeated_ids,
                attention_mask=repeated_mask,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = prompt_ids.shape[1]
        labels = generated.clone()
        labels[:, :prompt_len] = IGNORE_INDEX
        attention_mask = generated.ne(self.tokenizer.pad_token_id).long()
        outputs = model(input_ids=generated, attention_mask=attention_mask, labels=labels)
        logps = sequence_logps(outputs.logits, labels)

        prompts = self.tokenizer.batch_decode(repeated_ids, skip_special_tokens=True)
        responses = self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        rewards = self.reward_fn(prompts, responses).to(logps.device)
        grouped = rewards.view(-1, self.num_generations)
        advantages = (grouped - grouped.mean(dim=1, keepdim=True)).reshape(-1)
        denom = grouped.std(dim=1, keepdim=True).clamp_min(1e-6).reshape(-1).repeat_interleave(self.num_generations)
        advantages = advantages / denom[: advantages.shape[0]]
        loss = -(advantages.detach() * logps).mean() + self.kl_coef * outputs.loss
        return (loss, outputs) if return_outputs else loss
