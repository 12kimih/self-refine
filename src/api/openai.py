import os

import openai
from openai import OpenAI

from .base import Usage, BaseAPI, Response
from .utils import retry_with_exponential_backoff


class OpenAIAPI(BaseAPI):
    def __init__(self, config) -> None:
        super().__init__(config=config)
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
        if self.config.verbose == 2:
            print(usage, flush=True)

        def extract(choice) -> str:
            message = choice.message.content
            assert message
            return message

        outputs = [list(map(extract, outputs.choices))]
        response = Response(model=self.config.model, outputs=outputs, n=self.config.n_sequences, usage=usage)
        if self.config.verbose == 3:
            print(response, flush=True)
        return response
