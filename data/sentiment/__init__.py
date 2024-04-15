import itertools
import json
import re
import sys

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseGenerator


class SentimentFeedback(BaseModel):
    effective: str = str()
    effective_score: int = 0
    logical: str = str()
    logical_score: int = 0
    total_score: int = 0


class SentimentInput(BaseModel):
    review: str = Field(title="Review", validation_alias="text")


class SentimentTrainOutput(BaseModel):
    review: str
    reversed_review_a: str
    feedback_a: SentimentFeedback
    reversed_review_b: str
    feedback_b: SentimentFeedback


class SentimentTestOutput(BaseModel):
    review: str
    reversed_review: str


class SentimentTrainGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, review: str) -> str:
        prompt: str = str()
        prompt += self.template.train_context.format(review=review)
        prompt += self.template.train_instruction
        return prompt

    def _parse_response(self, response: str) -> dict | None:
        pattern = re.compile(self.template.train_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedbacks = {
            "reversed_review_a": match.group(1),
            "feedback_a": SentimentFeedback(
                effective=match.group(2),
                effective_score=int(match.group(3)),
                logical=match.group(4),
                logical_score=int(match.group(5)),
                total_score=int(match.group(3)) + int(match.group(5)),
            ),
            "reversed_review_b": match.group(6),
            "feedback_b": SentimentFeedback(
                effective=match.group(7),
                effective_score=int(match.group(8)),
                logical=match.group(9),
                logical_score=int(match.group(10)),
                total_score=int(match.group(8)) + int(match.group(10)),
            ),
        }
        return feedbacks

    def _input(self) -> SentimentInput:
        return SentimentInput.model_validate(self._input_schema(schema=SentimentInput.model_json_schema()))

    def _generate(self, _input: SentimentInput, output_file, prompt_file) -> SentimentTrainOutput:
        prompt: str = self._build_prompt(review=_input.review)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedbacks: dict | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            feedbacks = self._parse_response(response=response.outputs[0][0])
            if not feedbacks:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedbacks
        print(self.template.train_output.format(review=_input.review, **feedbacks), file=output_file, flush=True)
        output = SentimentTrainOutput(**_input.model_dump(), **feedbacks)
        return output

    def __call__(self, input_file, output_file, record_file, config_file, prompt_file) -> None:
        config = {
            "task": self.config.model_dump(),
            "api": self.api.config.model_dump(),
            "template": self.template.model_dump(),
        }
        json.dump(config, config_file)
        config_file.flush()

        if input_file and record_file:
            for i, line in tqdm(itertools.islice(enumerate(input_file), self.config.start, self.config.end)):
                _input: SentimentInput = SentimentInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentimentTrainOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class SentimentTestGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, review: str) -> str:
        prompt: str = str()
        prompt += self.template.test_context.format(review=review)
        prompt += self.template.test_instruction
        return prompt

    def _parse_response(self, response: str) -> str | None:
        pattern = re.compile(self.template.test_regex)
        match = pattern.search(response)
        if not match:
            return None
        reversed_review: str = match.group(1)
        return reversed_review

    def _input(self) -> SentimentInput:
        return SentimentInput.model_validate(self._input_schema(schema=SentimentInput.model_json_schema()))

    def _generate(self, _input: SentimentInput, output_file, prompt_file) -> SentimentTestOutput:
        prompt: str = self._build_prompt(review=_input.review)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        reversed_review: str | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            reversed_review = self._parse_response(response=response.outputs[0][0])
            if not reversed_review:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert reversed_review
        print(self.template.test_output.format(review=_input.review, reversed_review=reversed_review), file=output_file, flush=True)
        output = SentimentTestOutput(**_input.model_dump(), reversed_review=reversed_review)
        return output

    def __call__(self, input_file, output_file, record_file, config_file, prompt_file) -> None:
        config = {
            "task": self.config.model_dump(),
            "api": self.api.config.model_dump(),
            "template": self.template.model_dump(),
        }
        json.dump(config, config_file)
        config_file.flush()

        if input_file and record_file:
            for i, line in tqdm(itertools.islice(enumerate(input_file), self.config.start, self.config.end)):
                _input: SentimentInput = SentimentInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentimentTestOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
