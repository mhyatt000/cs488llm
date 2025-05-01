import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, Optional

from cs488llm.cn import CN, default


@dataclass
class Model(CN):
    name: str
    REGISTRY: ClassVar[Dict[str, "Model"]] = {}
    _E: ClassVar[None | Enum] = None

    path: Optional[str] = None
    version: str = "0.0.1"

    def __post_init__(self):
        self.REGISTRY[self.name] = self

    @classmethod
    def E(cls, force=False):
        # dont recompute enum if it exists
        if not cls._E or force:
            cls._E = Enum(f'{str(cls.__name__)}E', cls.REGISTRY)
        return cls._E

    def info(self):
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
        }

    @abstractmethod
    def create(self):
        raise NotImplementedError("ABC")


@dataclass
class DummyModel(Model):
    name: str = "dummy"
    path: Optional[str] = None

    def create(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.random_text(random.randint(10, 100))

    def random_text(self, length):
        c = string.ascii_letters + string.digits + " "
        return "".join(random.choices(c, k=length))
