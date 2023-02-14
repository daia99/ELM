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


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None
    initialize_from_vocab: bool = True  # of softprompt
    tune_v_head: bool = True  # set in case whole model is frozen (except softprompt)
    measure_soft_embedding_drift: bool = True  # for debugging purposes


if __name__ == "__main__":
    config = TRLConfig.load_yaml("/nfs/scratch_2/marco/OpenELM/trlx_example/configs/ppo_softprompt_config.yml")

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)

    max_gen_length = config.method.gen_kwargs[
        "max_length"
    ]  # set reward as 0 if max length is reached.

    def reward_fn(samples):
        samples_tokenized = tokenizer(samples)
        samples_token_ids = samples_tokenized.data["input_ids"]
        reward = [
            (1 - len(item_ids) / max_gen_length) for item_ids in samples_token_ids
        ]

        return reward  # list of scalar reward scores for each response

    # Take few words off of movies reviews as prompts
    # imdb = load_dataset("imdb", split="train+test")
    code_data = load_dataset("tomekkorbak/python-github-code", split="train")
    prompts = [" ".join(code.split()[:4]) for code in code_data["text"]]

    model = trlx.train(
        config.model.model_path,
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["from . import "] * 64,
        config=config,
    )

    print("DONE!")
