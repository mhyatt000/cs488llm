from dataclasses import dataclass
import torch
from enum import Enum
from pathlib import Path
from typing import Dict

import accelerate
import tyro
from rich.pretty import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from webpolicy.deploy.base_policy import BasePolicy
from webpolicy.deploy.server import WebsocketPolicyServer as Server

from cs488llm.cn import CN, default
from cs488llm.models import DummyModel, Model


class Gemma(Model):

    def create(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.path, device_map="auto", torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.path)
        return model, tokenizer


g1 = Gemma(name="gemma3-1b", path="google/gemma-3-1b-it")
dummy = DummyModel()


class LLMPolicy(BasePolicy):

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def infer(self, obs: Dict[str, str]) -> Dict[str, str]:
        text = obs["text"]
        tok = self.tokenizer(text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**tok)
        return {"agent": self.tokenizer.decode(outputs[0])}

    def reset(self) -> None:
        pass


class PipelinePolicy(BasePolicy):
    def __init__(self, url):
        self.url = url
        self.pipeline = pipeline(
            "text-generation",
            model=url,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def infer(self, messages  : Dict[str, str]) -> Dict[str, str]:
        response = self.pipeline(messages, max_new_tokens=512)
        return response

    def reset(self) -> None:
        pass


class TryPolicy(BasePolicy):
    """A policy that tries to run the given policy."""

    def __init__(self, policy: BasePolicy) -> None:
        self.policy = policy

    def infer(self, obs: dict) -> dict:
        try:
            action = self.policy.infer(obs)
            return action
        except Exception as e:
            print("Error in policy: %s", e)
            import traceback

            traceback.print_exc()

        return {}

    def reset(self, *args, **kwargs):
        return self.policy.reset()


@dataclass
class ServerConfig:

    model: Model.E()  # the llm to serve

    host: str = "0.0.0.0"
    port: int = 8080


def main(cfg: ServerConfig):
    print("Hello from cs488llm-hw!")

    # model, tokenizer = cfg.model.value.create()
    # policy = LLMPolicy(model, tokenizer)
    policy = PipelinePolicy(cfg.model.value.path)
    policy = TryPolicy(policy)
    server = Server(
        policy,
        host=cfg.host,
        port=cfg.port,
        metadata=None,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
