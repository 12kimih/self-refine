import logging
import os
import random
import time

import openai
from openai import OpenAI
from pydantic import BaseModel, Field

MODEL = {
    "gpt-3.5-turbo": "gpt-3.5",
    "gpt-3.5-turbo-instruct": "gpt-3.5-instruct",
    "gpt-3.5-turbo-0125": "gpt-3.5",
    "gpt-4": "gpt-4",
    "gpt-4-32k": "gpt-4",
    "gpt-4-turbo-preview": "gpt-4",
    "gpt-4-vision-preview": "gpt-4-vision",
    "gpt-4-0125-preview": "gpt-4",
}


class APIConfig(BaseModel):
    model: str = Field(default="gpt-3.5-turbo-0125", description="[api] text generation model code", json_schema_extra={"choices": list(MODEL)})
    max_new_tokens: int | None = Field(default=1024, description="[api] maximum number of tokens to generate")
    temperature: float = Field(default=0.5, description="[api] value used to modulate the next token probabilities")
    top_p: float = Field(default=0.5, description="[api] only the smallest set of most probable tokens with probabilities that add up to top_p or higher is kept for generation")
    frequency_penalty: float = Field(default=0.0, description="[api] number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far")
    presence_penalty: float = Field(default=0.0, description="[api] number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far")
    n_sequences: int = Field(default=1, description="[api] number of independently computed returned sequences for each element in the batch")
    verbose: int = Field(default=0, description="[api] verbosity level; 0: no info, 1: model post-usage, 2: model response")


def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_delay: float = 60,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = tuple(),
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay: float = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise Exception(f"Maximum number of retries ({max_retries}) exceeded.")
                    delay *= min(exponential_base * (1 + jitter * random.random()), max_delay)
                    time.sleep(delay)
                    logging.error(msg=f"Retrying #{num_retries}...")
                except Exception as e:
                    raise e

        return wrapper

    return decorator


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Response(BaseModel):
    model: str
    outputs: list[list[str]]
    n: int
    usage: Usage


class OpenAIAPI:
    def __init__(self, config) -> None:
        self.config = config
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry_with_exponential_backoff(errors=(openai.RateLimitError,))
    def _completions_with_backoff(self, *args, **kwargs):
        return self.client.chat.completions.create(*args, **kwargs)

    def _build_messages(
        self,
        messages: list[dict[str, str]] | None = None,
        system: str | None = None,
        user: str | None = None,
        assistant: str | None = None,
    ) -> list[dict[str, str]]:
        if messages is None:
            messages = list()
        if system:
            messages.append({"role": "system", "content": system})
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def __call__(
        self,
        inputs: str | list[str] | None = None,
        messages: list[dict[str, str]] | None = None,
        stop: list[str] | None = None,
    ) -> Response:
        if inputs is None and messages is None:
            raise ValueError("API call with neither inputs nor messages.")
        if messages is None:
            assert isinstance(inputs, str)
            messages = self._build_messages(user=inputs)
        outputs = self._completions_with_backoff(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
            top_p=self.config.top_p,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            stop=stop,
            n=self.config.n_sequences,
        )
        assert outputs.usage
        usage = Usage(input_tokens=outputs.usage.prompt_tokens, output_tokens=outputs.usage.completion_tokens, total_tokens=outputs.usage.total_tokens)
        if self.config.verbose == 1:
            print(usage, flush=True)

        def extract(choice) -> str:
            message = choice.message.content
            assert message
            return message

        outputs = [list(map(extract, outputs.choices))]
        response = Response(model=self.config.model, outputs=outputs, n=self.config.n_sequences, usage=usage)
        if self.config.verbose == 2:
            print(response, flush=True)
        return response


def get_api(args):
    config = APIConfig(**vars(args))
    return OpenAIAPI(config)


def get_model_name(model: str) -> str:
    if model not in MODEL:
        raise ValueError(f"{model} is not supported.")
    return MODEL[model]
