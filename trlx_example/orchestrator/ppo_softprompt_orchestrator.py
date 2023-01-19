from time import time
from typing import Callable

import ray
import torch
import torch.nn.functional as F

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

    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """
        Takes `num_rollouts` prompts from `pipeline`, samples model and computes the
        KL againts a reference model. It then appends PPOElements to trainer's `store`

        Modified override to handle indices containing soft prompts
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
            stats["time/exp_generate"] = time() - exp_generate_time

            # here, we take care to handle additional softprompt indices
            query_len = batch.input_ids.shape[1] + self.n_soft_tokens
            query_tensors = samples[:, :query_len]
            device = samples.device
            str_samples, str_prompts, str_outputs = self.trainer.decode(
                query_tensors, samples
            )

            # Convert trimmed samples back into tensors for another head pass
            # This can be defered, instead letting the pass to made over the original samples
            # after unbinding and truncating operations lower are fixed
            outputs = self.trainer.tokenizer(str_outputs).input_ids
            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.trainer.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            response_tensors = torch.vstack(outputs).to(device)

            exp_score_time = time()

            scores = torch.tensor(
                self.trainer.reward_fn(
                    samples=str_samples,
                    prompts=str_prompts,
                    outputs=str_outputs,
                ),
                dtype=float,
            ).to(device)
            stats["time/exp_score"] = time() - exp_score_time

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = scores.mean(), scores.std()
            all_scores_mean, all_scores_std = self.running.update(scores)
            stats["exp_scores/mean"] = all_scores_mean
            stats["exp_scores/std"] = all_scores_std
            stats["exp_scores/running_mean"] = self.running.mean
            stats["exp_scores/running_std"] = self.running.std

            if self.trainer.config.method.scale_reward == "running":
                scores /= self.running.std
            elif self.trainer.config.method.scale_reward == "ref":
                scores /= self.ref_std

            clip_reward = self.trainer.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)

            # Precompute logprobs, values
            if self.trainer.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                query_tensors = batch.input_ids.to(device)
                with torch.no_grad():
                    outputs = self.trainer.model(
                        input_ids=query_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=response_tensors,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.trainer.model, "frozen_head"):
                        ref_logits = self.trainer.model.forward_hydra(
                            input_ids=query_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=response_tensors,
                        )
                    else:
                        ref_logits = self.ref_model(
                            input_ids=query_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=response_tensors,
                        ).logits
            else:
                all_tokens = torch.cat(
                    (query_tensors.to(device), response_tensors), dim=1
                )
                attention_mask = (
                    all_tokens.not_equal(self.trainer.tokenizer.pad_token_id)
                    .long()
                    .to(device)
                )
                with torch.no_grad():
                    logits, *_, values = self.trainer.model(
                        all_tokens,
                        attention_mask=attention_mask,
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.trainer.model, "frozen_head"):
                        ref_logits = self.trainer.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=False,
                        )
                    else:
                        ref_logits, _, *_ = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            return_dict=False,
                        )
                        ref_logits = ref_logits.to(device)

            if self.trainer.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_from_logits(
                    logits[:, :-1, :], response_tensors[:, 1:]
                )
                ref_logprobs = logprobs_from_logits(
                    ref_logits[:, :-1, :], response_tensors[:, 1:]
                )
            else:
                logprobs = logprobs_from_logits(logits, all_tokens)
                ref_logprobs = logprobs_from_logits(ref_logits, all_tokens)

            n = samples.shape[0]
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            query_tensors = query_tensors.cpu()
            response_tensors = response_tensors.cpu()
            if self.trainer.config.model.model_arch_type == "seq2seq":
                start = 1  # skip the <s> token
                ends = (response_tensors[:, start:] != 0).sum(1)
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n)]
                all_values = [values[ix, start - 1 : ends[ix] - 1] for ix in range(n)]
                rewards = [
                    -self.trainer.kl_ctl.value
                    * (
                        logprobs[ix, start : ends[ix]]
                        - ref_logprobs[ix, start : ends[ix]]
                    )
                    for ix in range(n)
                ]
            else:
                logprobs = logprobs_from_logits(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_from_logits(
                    ref_logits[:, :-1, :], all_tokens[:, 1:]
                )

                n = samples.shape[0]
                values = values.cpu()[:, :-1]
                logprobs = logprobs.cpu()
                ref_logprobs = ref_logprobs.cpu()
                query_tensors = query_tensors.cpu()
                response_tensors = response_tensors.cpu()

                start = query_tensors.shape[1] - 1
                ends = start + attention_mask[:, start:].sum(1)
                all_values = [values[ix, start : ends[ix]] for ix in range(n)]
                all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n)]

                rewards = -self.trainer.kl_ctl.value * (logprobs - ref_logprobs)
                rewards = [rs[start : ends[ix]] for ix, rs in enumerate(rewards)]

            # Compute rewards
            all_rewards = [None] * n

            for ix in range(n):
                rs = rewards[ix]
                if len(rs) == 0:
                    rs = torch.tensor([0.0])
                rs[-1] += scores[ix].cpu()
                all_rewards[ix] = rs

            new_ppo_rl_elements = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=response_tensors[i],
                    logprobs=all_logprobs[i],
                    values=all_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n)
            ]
            ppo_rl_elements += new_ppo_rl_elements
            exp_time = clock.tick()

        stats["kl_ctl_value"] = self.trainer.kl_ctl.value
        stats["time/exp"] = exp_time

        if not ray.is_initialized():
            self.trainer.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.trainer.push_to_store(ppo_rl_elements)