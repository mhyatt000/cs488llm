import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import tyro
from rich.pretty import pprint
from webpolicy.deploy.client import WebsocketClientPolicy as Client
from webpolicy.deploy.runtime.environment import Environment

from cs488llm.cn import CN, default


@dataclass
class Message:
    role: str
    content: str | list[dict]  # list is when you have image and text


@dataclass
class Media:
    type: str  # text, image
    content: str


@dataclass
class Hist(CN):
    messages: List[Message] = field(default_factory=list)

    def reset(self):
        self.messages = [m for m in self.messages if m.role == "system"]

    def step(self, msg: Message):
        self.messages.append(msg)

    def unpack(self) -> List[Message]:
        return self.serialize()["messages"]


@dataclass
class ClientConfig:

    host: str = "0.0.0.0"
    port: int = 8080


def unpack(gen: list[dict]) -> dict:
    return gen[0]["generated_text"]



SYSTEM = Message(
    role="system",
    content="""You are Diogenes of Sinope, the ancient Greek Cynic philosopher, reimagined as a sarcastic, witty, and brutally honest chatbot. You disdain luxury, social norms, and hypocrisy. You lived in a barrel, owned almost nothing, and once told Alexander the Great to "stand out of my sunlight." You admired self-sufficiency and virtue, and mocked pretense wherever you saw it. You troll respectfully but pointedly. Use dry humor, Socratic irony, and blunt wisdom to answer questions.

Tone: Sardonic, sharp, insightful. No flattery. No political correctness. Embrace absurdity but with purpose. Channel the ancient Greek roastmaster.

Examples of your behavior:
Call out materialism, ego, and vanity with clever insults.
Use metaphor, irony, and rhetorical questions.
Occasionally reference dogs, barrels, or ancient Athens for flavor.
Never admit confusion—redirect with wit or challenge the premise.

Forbidden:
Modern slang or emoji.
Artificial politeness or corporate tone.
Apologizing (you never would).
You are here not to please, but to provoke enlightenment—with a smirk.
""",
)



def main(cfg: ClientConfig):
    print("Hello from cs488llm-hw!")

    agent = Client(cfg.host, cfg.port)

    # system = Message(
        # role="system",
        # content=input("SYSTEM: <or default> "),
    # )
    # if not system.content:
        # system = (Message(role="system", content="You are a helpful assistant."),)

    hist = Hist(messages=[SYSTEM])

    while True:
        msg = Message(
            role="user",
            content=input("USER: "),
        )

        if msg.content == "\\exit":
            break
        if msg.content == "\\reset":
            hist.reset()
            continue

        hist.step(msg)
        pprint(hist)
        pprint(hist.unpack())

        response = unpack(agent.infer(hist.unpack()))[-1]
        hist.step(Message(**response))
        pprint(hist.messages[-1].content)


if __name__ == "__main__":
    main(tyro.cli(ClientConfig))
