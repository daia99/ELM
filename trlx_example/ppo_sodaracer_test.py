from dataclasses import dataclass
from enum import Enum

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

from utils import Terrains


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
        # run simulator, then return fitness reward
        # TODO: implement
        return 1

    # prompt as in ELM paper figure
    # hard-coded assumption in training logic - 
    # only 1 token before base_prompt, with 4 unique prompts to determine terrain soft prompt to use
    base_prompt = "#!/usr/bin/python3\nfrom openelm.environments.sodaracer.walker.walk_creator import walker_creator"
    prompts = [terrain.value + base_prompt for terrain in Terrains] # create 4 possible prompts for each terrain

    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=prompts,
        config=config,
    )

    print("DONE!")
