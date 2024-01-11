from abc import ABC, abstractmethod

from pydantic import BaseModel


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Response(BaseModel):
    model: str
    outputs: list[list[str]]
    n: int
    usage: Usage


class BaseAPI(ABC):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def __call__(self):
        pass
