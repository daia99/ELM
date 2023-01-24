import copy
from time import time
from typing import Tuple

import ray
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torchtyping import TensorType
from rich.console import Console
from rich.table import Table
from trlx.data.configs import TRLConfig
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.nn.ppo_models import CausalLMHydraWithValueHead
from trlx.utils import significant


class SoftEmbedding(nn.Module):
    def __init__(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """
        appends learned embedding as prefix

        From: https://github.com/kipgparker/soft-prompt-tuning

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super().__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.padding_token_id = 50256  # used when input tensors are prefix padded
        self.learned_embedding = (
            nn.parameter.Parameter(  # dim: (n_tokens, embedding_dim)
                self.initialize_embedding(
                    wte, n_tokens, random_range, initialize_from_vocab
                )
            )
        )
        self.init_embedding = copy.deepcopy(self.learned_embedding)

    def initialize_embedding(
        self,
        wte: nn.Embedding,
        n_tokens: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
    ):
        """
        initializes learned embedding

        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(
            -random_range, random_range
        )

    def forward(self, tokens):
        """
        run forward pass

        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            seq_embedding (torch.float): encoding of text concatenated with learned task specifc embedding
        """
        prompt_tokens = tokens[
            :, self.n_tokens :
        ]  # dim: (batch_size, seq_length) - without soft prompt padding indices
        if self.padding_token_id in prompt_tokens[:, 0]:  # padding is applied as prefix
            seq_embedding = self.wte(
                tokens
            )  # dim: (batch_size, seq_length, embedding_dim)
            padding_tensor = torch.tensor([self.padding_token_id]).to(
                seq_embedding.device
            )

            # index in each sequence in tokens just after last prefix padding
            # this would be where the (first) soft prompt embedding should be set
            first_prompt_indices = (
                (prompt_tokens == padding_tensor).int().argmin(axis=1)
            )

            # for asserting that the first main sequence token embedding isn't modified by accident
            first_prompt_indices_full_seq = (
                (tokens == padding_tensor).int().argmin(axis=1)
            )
            first_item_idx = first_prompt_indices_full_seq[0]
            main_embedding_before_soft_prompt_assign = seq_embedding[0, first_item_idx]

            # for each batch sequence, replace embeddings at soft prompt indices with correct soft embeddings
            for batch_idx, first_prompt_idx in enumerate(first_prompt_indices.tolist()):
                # indices for assigning soft embeddings
                start = first_prompt_idx
                end = first_prompt_idx + self.n_tokens

                seq_embedding[batch_idx, start:end] = self.learned_embedding

                # debug only
                if batch_idx == 0:
                    first_main_embedding_after_soft_embedding_assign = seq_embedding[
                        0, first_item_idx
                    ]
                    assert torch.equal(
                        main_embedding_before_soft_prompt_assign,
                        first_main_embedding_after_soft_embedding_assign,
                    ), "Error: soft prompt overwrote main prompt embeddings"
        else:
            input_embedding = self.wte(prompt_tokens)
            learned_embedding = self.learned_embedding.repeat(
                input_embedding.size(0), 1, 1
            )
            seq_embedding = torch.cat([learned_embedding, input_embedding], 1)

        assert (
            seq_embedding.shape[1] == prompt_tokens.shape[1] + self.n_tokens
        ), "Number of token embeddings with soft prompts should be number of prompt tokens + number of soft tokens"

        return seq_embedding


@register_trainer
class AcceleratePPOSoftpromptTrainer(AcceleratePPOTrainer):
    def __init__(self, config, train_mode=True, **kwargs):
        # account for extra prefix tokens
        config.method.gen_kwargs["max_new_tokens"] += config.method.n_soft_tokens

        super().__init__(config, **kwargs)

        assert (
            config.method.n_soft_tokens > 0
        ), "Number of soft prompt tokens should be >=1"

        self.soft_dummy_token_id = 50256  # dummy token for padding soft prompts
        self.measure_soft_embedding_drift = config.method.measure_soft_embedding_drift

    def get_arch(self, config: TRLConfig):
        """
        Load model, and set Soft Prompt module for input embeddings
        """
        model = CausalLMHydraWithValueHead(
            config.model.model_path, config.model.num_layers_unfrozen
        )

        # if all layers are frozen, freeze all params. Softprompt will still be tuned
        if config.model.num_layers_unfrozen == 0:
            model.requires_grad_(False)

            if config.method.tune_v_head:
                model.v_head.requires_grad_(True)  # unfreeze value head

        # here, we setup softprompts by initializing learned softprompt embedding(s)
        # and the model's input embeddings.
        # the model will always concatenate learned softprompt embeddings as prefix to the prompt/query after it's set
        # use config option to initialize embedding from existing vocab, or random
        self.n_soft_tokens = (
            config.method.n_soft_tokens
        )  # number of prefix tokens added to prompt, with learned embeddings

        s_wte = SoftEmbedding(
            model.base_model.get_input_embeddings(),
            n_tokens=self.n_soft_tokens,
            initialize_from_vocab=config.method.initialize_from_vocab,
        )

        model.base_model.set_input_embeddings(s_wte)

        return model

    def generate(
        self,
        input_ids: TensorType["batch_size", "seq_length"],
        attention_mask: TensorType["batch_size", "seq_length"] = None,
        **kwargs,
    ):
        """
        Wraps hf's `generate` adding some specific method's defaults

        Modified to handle indices containing soft prompts
        """
        # pad for soft prompt indices (using same token as for padding)
        input_ids_padded, attention_mask_padded = self.softprompt_pad_input_and_mask(input_ids, attention_mask)

        if self.generate_experience_kwargs is not None:
            kwargs = dict(self.generate_experience_kwargs, **kwargs)
        else:
            kwargs = dict(self.generate_kwargs, **kwargs)
        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            # disable cache needed for softprompt compatibility
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                use_cache=False,
                **kwargs,
            )

    def generate_eval(self, input_ids, attention_mask=None, **kwargs):
        """Wraps hf's `generate` adding some specific method's defaults"""
        # pad for soft prompt indices (using same token as for padding)
        input_ids_padded, attention_mask_padded = self.softprompt_pad_input_and_mask(input_ids, attention_mask)

        kwargs = dict(self.generate_kwargs, **kwargs)

        with torch.no_grad():
            # disable cache needed for softprompt compatibility
            return self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids_padded,
                attention_mask=attention_mask_padded,
                use_cache=False,
                **kwargs,
            )

    def softprompt_pad_input_and_mask(self, input_ids, attention_mask):
        input_ids = torch.cat(
            [
                torch.full(
                    (input_ids.shape[0], self.n_soft_tokens), self.soft_dummy_token_id
                ).to(input_ids.device),
                input_ids,
            ],
            1,
        )
        input_ids = input_ids.to(self.accelerator.device)
        if attention_mask is not None:
            # extend for soft prompt indices (by extending mask at the end of tensor)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.full((attention_mask.shape[0], self.n_soft_tokens), 1).to(
                        attention_mask.device
                    ),
                ],
                1,
            )
            attention_mask = attention_mask.to(self.accelerator.device)
        
        return input_ids,attention_mask
    
    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Used in orchestrator and loss calculation, to compute logprobs and values

        Modified to handle indices containing soft prompts
        """
        tokens = torch.cat((query_tensors, response_tensors), dim=1)
        attention_mask = (
            tokens.not_equal(self.tokenizer.pad_token_id).long().to(tokens.device)
        )
        # to handle extra softprompts, set attention at softprompt indices
        first_non_pad_indices = torch.argmax(attention_mask, dim=1)
        for batch_idx, first_non_pad_idx in enumerate(first_non_pad_indices.tolist()):
            start = first_non_pad_idx - self.n_soft_tokens
            end = first_non_pad_idx
            attention_mask[batch_idx, start:end] = 1.0

        # For a proper positional encoding in case of left padding
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)
        return tokens, attention_mask, position_ids

    def evaluate(self):  # noqa: C901
        """
        Samples model on `eval_prompts`, logs stats with `reward_fn` or `metric_fn` if provided
        
        Modified to support plotting of metrics involving soft prompts
        """
        stats = {}
        table = []

        # Do multiple evaluations over a single list in `gen_kwargs` if present
        if self.generate_sweep_kwarg is not None:
            gen_sweep_arg, gen_sweep_values = self.generate_sweep_kwarg
        else:
            gen_sweep_values = [None]

        for gen_sweep_value in gen_sweep_values:
            # A dedicated suffix for wandb logging
            if gen_sweep_value is not None:
                sweep_suffix = f"@{gen_sweep_arg}={gen_sweep_value}"
            else:
                sweep_suffix = ""

            all_samples = []
            all_prompts = []
            prompt_sizes = []
            generate_time = time()
            for prompts in self.eval_dataloader:
                if self.generate_sweep_kwarg:
                    samples = self.generate_eval(
                        **prompts, **{gen_sweep_arg: gen_sweep_value}
                    )
                else:
                    samples = self.generate_eval(**prompts)

                if self.config.model.model_arch_type == "seq2seq":
                    samples = samples[:, 1:]

                all_samples.append(
                    F.pad(
                        samples,
                        (0, self.max_length - samples.shape[1]),
                        value=self.tokenizer.pad_token_id,
                    )
                )
                all_prompts.append(
                    F.pad(
                        prompts.input_ids,
                        (0, self.max_length - prompts.input_ids.shape[1]),
                        value=self.tokenizer.pad_token_id,
                    ).to(samples.device)
                )
                prompt_sizes.append(
                    torch.tensor(
                        prompts.input_ids.shape[1], device=samples.device
                    ).repeat(len(prompts.input_ids))
                )

            stats["time/generate"] = time() - generate_time

            samples = self.accelerator.gather(torch.vstack(all_samples))
            prompts = self.accelerator.gather(torch.vstack(all_prompts))
            prompt_sizes = self.accelerator.gather(torch.hstack(prompt_sizes))

            if self.accelerator.is_main_process:
                str_samples, str_prompts, str_outputs = self.decode(
                    prompts, samples, prompt_sizes
                )

                columns = ["prompt", "output"]
                columns_data = [str_prompts, str_outputs]

                # in online setting, compute the reward for validation
                if self.reward_fn:
                    rewards = torch.tensor(
                        self.reward_fn(
                            samples=str_samples,
                            prompts=str_prompts,
                            outputs=str_outputs,
                        ),
                        dtype=float,
                    )
                    mean_reward = rewards.mean().item()
                    columns.append("reward")
                    if not isinstance(rewards, list):
                        rewards = rewards.tolist()
                    columns_data.append(rewards)
                    stats[f"reward/mean{sweep_suffix}"] = mean_reward

                # log Euclidean distance between init and current Soft Prompt embedding parameters
                if self.measure_soft_embedding_drift:
                    softprompt = self.model.base_model.get_input_embeddings()
                    stats["softprompt_drift_dist"] = (
                        (softprompt.init_embedding - softprompt.learned_embedding)
                        .pow(2)
                        .sum(1)
                        .sqrt()
                        .mean()
                    )
                
                # additionally log any other metrics
                if self.metric_fn:
                    metric_time = time()
                    metrics = self.metric_fn(str_samples)
                    stats["time/metric"] = time() - metric_time

                    mean_metrics = {
                        f"metrics/{k}{sweep_suffix}": torch.as_tensor(xs).mean(-1)
                        for k, xs in metrics.items()
                    }

                    stats.update(mean_metrics)

                    for metric, values in metrics.items():
                        columns.append(metric)
                        if not isinstance(values, list):
                            values = values.tolist()
                        columns_data.append(values)

                # Prepend the sweep argument along with samples
                if self.generate_sweep_kwarg:
                    columns.insert(0, gen_sweep_arg)
                    columns_data.insert(0, [gen_sweep_value] * len(samples))

                table.append(list(zip(*columns_data)))

        # Log and display evaluation metrics
        if self.accelerator.is_main_process:
            rows = sum(list(map(list, zip(*table))), [])

            # Add metrics/rewards to the table's title
            table_title = f"Evaluation #{self.nth_evaluation}"
            for k, x in stats.items():
                if k.startswith("reward") or k.startswith("metrics"):
                    table_title += f" {k}: {significant(x)}"

            rich_table = Table(*columns, title=table_title, show_lines=True)

            for ix in range(max(min(3, len(rows)), len(gen_sweep_values))):
                rich_table.add_row(*[str(significant(x)) for x in rows[ix]])

            if not ray.is_initialized():
                if "wandb" in self.config.train.tracker:
                    import wandb

                    stats["samples"] = wandb.Table(columns, rows)

            Console().print(rich_table)

        self.nth_evaluation += 1
        return stats
