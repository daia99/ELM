from dataclasses import dataclass
from typing import List
import os

import trlx
from datasets import load_dataset
import torch

# to register the added softprompt model, and supported orchestrator, need to import here
from trainer.accelerate_ppo_softprompt_trainer import AcceleratePPOSoftpromptTrainer
from orchestrator.ppo_softprompt_orchestrator import PPOSoftpromptOrchestrator
from transformers import pipeline
from trlx.data.configs import TRLConfig
from trlx.data.method_configs import register_method
from trlx.trainer.nn.ppo_models import PPOConfig


@dataclass
@register_method
class PPOSoftpromptConfig(PPOConfig):
    n_soft_tokens: int = None
    initialize_from_vocab: bool = True  # of softprompt
    tune_v_head: bool = True  # set in case whole model is frozen (except softprompt)
    measure_soft_embedding_drift: bool = True  # for debugging purposes


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
    
    sentiment_fn = pipeline(
        "sentiment-analysis",
        "lvwerra/distilbert-imdb",
        top_k=2,
        truncation=True,
        batch_size=256,
        device=device,
    )


    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        sentiments = list(map(get_positive_score, sentiment_fn(samples)))
        return sentiments

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]

    config = TRLConfig.load_yaml("/home/andrew_dai/OpenELM/trlx_example/configs/ppo_softprompt_config.yml")

    trlx.train(
        "lvwerra/gpt2-imdb",
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=["I don't know much about Hungarian underground"] * 64,
        config=config,
    )

    print("DONE!")
