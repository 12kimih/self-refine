import re
import sys
import json
import itertools

from tqdm import tqdm
from pydantic import Field, BaseModel

from ..base import BaseGenerator


class SentenceFeedback(BaseModel):
    inclusion: str = str()
    inclusion_score: int = 0
    logical: str = str()
    logical_score: int = 0
    total_score: int = 0


class SentenceInput(BaseModel):
    concepts: list[str] = Field(title="Concepts")


class SentenceTrainOutput(BaseModel):
    concepts: list[str]
    sentence_a: str
    feedback_a: SentenceFeedback
    sentence_b: str
    feedback_b: SentenceFeedback


class SentenceTestOutput(BaseModel):
    concepts: list[str]
    sentence: str


class SentenceTrainGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, concepts: list[str]) -> str:
        prompt: str = str()
        prompt += self.template.train_context.format(concepts=", ".join(concepts))
        prompt += self.template.train_instruction
        return prompt

    def _parse_response(self, response: str) -> dict | None:
        pattern = re.compile(self.template.train_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedbacks = {
            "sentence_a": match.group(1),
            "feedback_a": SentenceFeedback(
                inclusion=match.group(2),
                inclusion_score=int(match.group(3)),
                logical=match.group(4),
                logical_score=int(match.group(5)),
                total_score=int(match.group(3)) + int(match.group(5)),
            ),
            "sentence_b": match.group(6),
            "feedback_b": SentenceFeedback(
                inclusion=match.group(7),
                inclusion_score=int(match.group(8)),
                logical=match.group(9),
                logical_score=int(match.group(10)),
                total_score=int(match.group(8)) + int(match.group(10)),
            ),
        }
        return feedbacks

    def _input(self) -> SentenceInput:
        return SentenceInput.model_validate(self._input_schema(schema=SentenceInput.model_json_schema()))

    def _generate(self, _input: SentenceInput, output_file, prompt_file) -> SentenceTrainOutput:
        prompt: str = self._build_prompt(concepts=_input.concepts)
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
        print(self.template.train_output.format(concepts=", ".join(_input.concepts), **feedbacks), file=output_file, flush=True)
        output = SentenceTrainOutput(**_input.model_dump(), **feedbacks)
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
                _input: SentenceInput = SentenceInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentenceTrainOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class SentenceTestGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, concepts: list[str]) -> str:
        prompt: str = str()
        prompt += self.template.test_context.format(concepts=", ".join(concepts))
        prompt += self.template.test_instruction
        return prompt

    def _parse_response(self, response: str) -> str | None:
        pattern = re.compile(self.template.test_regex)
        match = pattern.search(response)
        if not match:
            return None
        sentence: str = match.group(1)
        return sentence

    def _input(self) -> SentenceInput:
        return SentenceInput.model_validate(self._input_schema(schema=SentenceInput.model_json_schema()))

    def _generate(self, _input: SentenceInput, output_file, prompt_file) -> SentenceTestOutput:
        prompt: str = self._build_prompt(concepts=_input.concepts)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        sentence: str | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            sentence = self._parse_response(response=response.outputs[0][0])
            if not sentence:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert sentence
        print(self.template.test_output.format(concepts=", ".join(_input.concepts), sentence=sentence), file=output_file, flush=True)
        output = SentenceTestOutput(**_input.model_dump(), sentence=sentence)
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
                _input: SentenceInput = SentenceInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentenceTestOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
