import re
import sys
import json
import itertools

from tqdm import tqdm
from pydantic import Field, BaseModel

from ..base import BaseGenerator


class DialogFeedback(BaseModel):
    consistency: str = str()
    consistency_score: int = 0
    understand: str = str()
    understand_score: int = 0
    sustain: str = str()
    sustain_score: int = 0
    total_score: int = 0


class DialogInput(BaseModel):
    dialog: list[str] = Field(title="Conversation history")


class DialogTrainOutput(BaseModel):
    dialog: list[str]
    response_a: str
    feedback_a: DialogFeedback
    response_b: str
    feedback_b: DialogFeedback


class DialogTestOutput(BaseModel):
    dialog: list[str]
    response: str


class DialogTrainGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_dialog(self, dialog: list[str]) -> str:
        tag = {0: "A: ", 1: "B: "}
        tagged = [tag[i % 2] + s for i, s in enumerate(dialog)]
        return "\n".join(tagged)

    def _build_prompt(self, dialog: list[str]) -> str:
        prompt: str = str()
        prompt += self.template.train_context.format(dialog=self._build_dialog(dialog))
        prompt += self.template.train_instruction
        return prompt

    def _parse_response(self, response: str) -> dict | None:
        pattern = re.compile(self.template.train_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedbacks = {
            "response_a": match.group(1),
            "feedback_a": DialogFeedback(
                consistency=match.group(2),
                consistency_score=int(match.group(3)),
                understand=match.group(4),
                understand_score=int(match.group(5)),
                sustain=match.group(6),
                sustain_score=int(match.group(7)),
                total_score=int(match.group(3)) + int(match.group(5)) + int(match.group(7)),
            ),
            "response_b": match.group(8),
            "feedback_b": DialogFeedback(
                consistency=match.group(9),
                consistency_score=int(match.group(10)),
                understand=match.group(11),
                understand_score=int(match.group(12)),
                sustain=match.group(13),
                sustain_score=int(match.group(14)),
                total_score=int(match.group(10)) + int(match.group(12)) + int(match.group(14)),
            ),
        }
        return feedbacks

    def _input(self) -> DialogInput:
        return DialogInput.model_validate(self._input_schema(schema=DialogInput.model_json_schema()))

    def _generate(self, _input: DialogInput, output_file, prompt_file) -> DialogTrainOutput:
        prompt: str = self._build_prompt(dialog=_input.dialog)
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
        print(self.template.train_output.format(dialog=self._build_dialog(_input.dialog), **feedbacks), file=output_file, flush=True)
        output = DialogTrainOutput(**_input.model_dump(), **feedbacks)
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
                _input: DialogInput = DialogInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: DialogTrainOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class DialogTestGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_dialog(self, dialog: list[str]) -> str:
        tag = {0: "A: ", 1: "B: "}
        tagged = [tag[i % 2] + s for i, s in enumerate(dialog)]
        return "\n".join(tagged)

    def _build_prompt(self, dialog: list[str]) -> str:
        prompt: str = str()
        prompt += self.template.test_context.format(dialog=self._build_dialog(dialog))
        prompt += self.template.test_instruction
        return prompt

    def _parse_response(self, response: str) -> str | None:
        pattern = re.compile(self.template.test_regex)
        match = pattern.search(response)
        if not match:
            return None
        _response: str = match.group(1)
        return _response

    def _input(self) -> DialogInput:
        return DialogInput.model_validate(self._input_schema(schema=DialogInput.model_json_schema()))

    def _generate(self, _input: DialogInput, output_file, prompt_file) -> DialogTestOutput:
        prompt: str = self._build_prompt(dialog=_input.dialog)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        _response: str | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            _response = self._parse_response(response=response.outputs[0][0])
            if not _response:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert _response
        print(self.template.test_output.format(dialog=self._build_dialog(_input.dialog), response=_response), file=output_file, flush=True)
        output = DialogTestOutput(**_input.model_dump(), response=_response)
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
                _input: DialogInput = DialogInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: DialogTestOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
