from time import time
from typing import Callable

import ray
import torch
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.ppo_types import PPORLElement
from trlx.trainer import BaseRLTrainer
from trlx.orchestrator import register_orchestrator
from trlx.orchestrator.ppo_orchestrator import PPOOrchestrator
from trlx.pipeline import BasePipeline
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_from_logits


@register_orchestrator
class PPOSoftpromptOrchestrator(PPOOrchestrator):
    def __init__(
        self,
        model: BaseRLTrainer,
        pipeline: BasePipeline,
        chunk_size: int = 512,
    ):
        super().__init__(model, pipeline, chunk_size)
        self.n_soft_tokens = model.model.base_model.get_input_embeddings().n_tokens

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model, computes KL againts a reference model appends PPOElements to model's `store`

        Modified to handle indices containing soft prompts
        """
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            # Get next batch in prompt dataset and refresh if exhausted
            try:
                batch: PromptBatch = next(self.pipeline_iterator)
            except StopIteration:
                self.pipeline_iterator = iter(self.pipeline_loader)
                batch = next(self.pipeline_iterator)

            exp_generate_time = time()
            samples = self.trainer.generate(**batch)
            stats["exp_generate_time"] = time() - exp_generate_time

            # here, we take care to handle additional softprompt indices
            query_len = batch.input_ids.shape[1] + self.n_soft_tokens
            query_tensors = samples[:, :query_len]
            response_tensors = samples[
                :, query_len:
            ]  # ignore softprompt padding index tokens
            texts = self.trainer.tokenizer.batch_decode(
                samples, skip_special_tokens=True
            )
            exp_score_time = time()
            scores = torch.as_tensor(self.score(texts), device=samples.device)
            stats["exp_score_time"] = time() - exp_score_time

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running.update(scores)
            stats["exp_scores_mean"] = all_scores_mean
            stats["exp_scores_std"] = all_scores_std
            stats["running_mean"] = self.running.mean
            stats["running_std"] = self.running.std

            if self.trainer.config.method.scale_reward == "running":
                scores /= self.running.std
            elif self.trainer.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.trainer.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            all_tokens, attention_mask, position_ids = self.trainer.get_model_inputs(
                query_tensors.to(response_tensors.device), response_tensors
            )
            with torch.no_grad():
                logits, *_, v = self.trainer.model(
                    all_tokens, attention_mask=attention_mask, position_ids=position_ids
                )
                # TODO(dahoas): When hydra model works need to also support generation on hydra head
                if hasattr(self.trainer.model, "frozen_head"):
                    ref_logits = self.trainer.model.forward_hydra(
                        all_tokens,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        return_dict=False,
                    )
                else:
                    ref_logits, _, *_ = self.ref_model(
                        all_tokens.cpu(),
                        attention_mask=attention_mask.cpu(),
                        position_ids=position_ids.cpu(),
                    )

            ref_logits = ref_logits.to(self.trainer.accelerator.device)
            logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
            ref_logprobs = logprobs_from_logits(
                ref_logits[:, :-1, :], all_tokens[:, 1:]
            )
            start = query_tensors.size()[1] - 1
            end = query_tensors.size()[1] + response_tensors.size()[1] - 1
            all_values = v[:, start:end]
            all_logprobs = logprobs[:, start:end]
            all_ref_logprobs = ref_logprobs[:, start:end]

            # Compute rewards
            kls = all_logprobs - all_ref_logprobs
            non_score_rewards = -self.trainer.kl_ctl.value * kls
            all_rewards = non_score_rewards.clone()
            all_rewards[:, -1] += scores.to(self.trainer.accelerator.device)

            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            all_logprobs = all_logprobs.cpu()
            all_values = all_values.cpu()
            all_rewards = all_rewards.cpu()

            exp_time = clock.tick()

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i, :],
                    response_tensor=response_tensors[i, :],
                    logprobs=all_logprobs[i, :],
                    values=all_values[i, :],
                    rewards=all_rewards[i, :],
                )
                for i in range(query_tensors.size()[0])
            ]
            ppo_rl_elements += new_ppo_rl_elements

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["exp_time"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to model's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)
