from dataclasses import dataclass

import trlx
from datasets import load_dataset


# to register the added softprompt model, and supported orchestrator, need to import here
from model.accelerate_ppo_softprompt_model import AcceleratePPOSoftpromptModel
from orchestrator.ppo_softprompt_orchestrator import PPOSoftpromptOrchestrator
from transformers import AutoTokenizer, pipeline
from trlx.data.configs import TRLConfig
from trlx.data.method_configs import register_method
from trlx.model.nn.ppo_models import PPOConfig
import sys


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None
    initialize_from_vocab: bool = True  # of softprompt
    tune_v_head: bool = True  # set in case whole model is frozen (except softprompt)
    measure_soft_embedding_drift: bool = True  # for debugging purposes


if __name__ == "__main__":
    config = TRLConfig.load_yaml(sys.argv[1])
    config.train.seed = int(sys.argv[2])

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)

    max_gen_length = config.method.gen_kwargs[
        "max_length"
    ]  # set reward as 0 if max length is reached.

    def reward_fn(samples):
        whitelist_words = ["class", "def", "os", "return"]
        max_possible_count = 30
        reward = []
        for item in samples:
            reward_item = 0
            for keyword in whitelist_words:
                reward_item = item.count(keyword)
            reward.append(min(reward_item, max_possible_count)/max_possible_count)

        return reward  # list of scalar reward scores for each response

    # Take few tokens off of python script examples as prompts
    code_data = load_dataset("tomekkorbak/python-github-code", split="train")
    tokenized_prompts = tokenizer(code_data["text"]).data["input_ids"]
    tokenized_prompts_truncated = [item_tokens[:4] for item_tokens in tokenized_prompts]
    prompts = tokenizer.batch_decode(tokenized_prompts_truncated)

    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["from . import "] * 64,
        config=config,
    )

    print("DONE!")
