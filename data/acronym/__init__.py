import itertools
import json
import re
import sys

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseGenerator


class AcronymFeedback(BaseModel):
    relevance: str = str()
    relevance_score: int = 0
    pronunciation: str = str()
    pronunciation_score: int = 0
    spelling: str = str()
    spelling_score: int = 0
    familiarity: str = str()
    familiarity_score: int = 0
    total_score: int = 0


class AcronymInput(BaseModel):
    title: str = Field(title="Title")


class AcronymTrainOutput(BaseModel):
    title: str
    acronym_a: str
    feedback_a: AcronymFeedback
    acronym_b: str
    feedback_b: AcronymFeedback


class AcronymTestOutput(BaseModel):
    title: str
    acronym: str


class AcronymTrainGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, title: str) -> str:
        prompt: str = str()
        prompt += self.template.train_context.format(title=title)
        prompt += self.template.train_instruction
        return prompt

    def _parse_response(self, response: str) -> dict | None:
        pattern = re.compile(self.template.train_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedbacks = {
            "acronym_a": match.group(1),
            "feedback_a": AcronymFeedback(
                relevance=match.group(2),
                relevance_score=int(match.group(3)),
                pronunciation=match.group(4),
                pronunciation_score=int(match.group(5)),
                spelling=match.group(6),
                spelling_score=int(match.group(7)),
                familiarity=match.group(8),
                familiarity_score=int(match.group(9)),
                total_score=int(match.group(3)) + int(match.group(5)) + int(match.group(7)) + int(match.group(9)),
            ),
            "acronym_b": match.group(10),
            "feedback_b": AcronymFeedback(
                relevance=match.group(11),
                relevance_score=int(match.group(12)),
                pronunciation=match.group(13),
                pronunciation_score=int(match.group(14)),
                spelling=match.group(15),
                spelling_score=int(match.group(16)),
                familiarity=match.group(17),
                familiarity_score=int(match.group(18)),
                total_score=int(match.group(12)) + int(match.group(14)) + int(match.group(16)) + int(match.group(18)),
            ),
        }
        return feedbacks

    def _input(self) -> AcronymInput:
        return AcronymInput.model_validate(self._input_schema(schema=AcronymInput.model_json_schema()))

    def _generate(self, _input: AcronymInput, output_file, prompt_file) -> AcronymTrainOutput:
        prompt: str = self._build_prompt(title=_input.title)
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
        print(self.template.train_output.format(title=_input.title, **feedbacks), file=output_file, flush=True)
        output = AcronymTrainOutput(**_input.model_dump(), **feedbacks)
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
                _input: AcronymInput = AcronymInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: AcronymTrainOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class AcronymTestGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, title: str) -> str:
        prompt: str = str()
        prompt += self.template.test_context.format(title=title)
        prompt += self.template.test_instruction
        return prompt

    def _parse_response(self, response: str) -> str | None:
        pattern = re.compile(self.template.test_regex)
        match = pattern.search(response)
        if not match:
            return None
        acronym: str = match.group(1)
        return acronym

    def _input(self) -> AcronymInput:
        return AcronymInput.model_validate(self._input_schema(schema=AcronymInput.model_json_schema()))

    def _generate(self, _input: AcronymInput, output_file, prompt_file) -> AcronymTestOutput:
        prompt: str = self._build_prompt(title=_input.title)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        acronym: str | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            acronym = self._parse_response(response=response.outputs[0][0])
            if not acronym:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert acronym
        print(self.template.test_output.format(title=_input.title, acronym=acronym), file=output_file, flush=True)
        output = AcronymTestOutput(**_input.model_dump(), acronym=acronym)
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
                _input: AcronymInput = AcronymInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: AcronymTestOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
