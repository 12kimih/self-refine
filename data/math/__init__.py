import re
import sys
import json
import itertools

from tqdm import tqdm
from pydantic import Field, BaseModel

from ..base import BaseGenerator


class MathSolution(BaseModel):
    steps: str = str()
    answer: int = 0


class MathFeedback(BaseModel):
    adequacy: str = str()
    adequacy_score: int = 0
    validity: str = str()
    validity_score: int = 0
    total_score: int = 0


class MathInput(BaseModel):
    question: str = Field(title="Math problem")


class MathTrainOutput(BaseModel):
    question: str
    solution_a: MathSolution
    feedback_a: MathFeedback
    solution_b: MathSolution
    feedback_b: MathFeedback


class MathTestOutput(BaseModel):
    question: str
    solution: MathSolution


class MathTrainGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, question: str) -> str:
        prompt: str = str()
        prompt += self.template.train_context.format(question=question)
        prompt += self.template.train_instruction
        return prompt

    def _parse_response(self, response: str) -> dict | None:
        pattern = re.compile(self.template.train_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedbacks = {
            "solution_a": MathSolution(steps=match.group(1), answer=int(match.group(2))),
            "feedback_a": MathFeedback(
                adequacy=match.group(3),
                adequacy_score=int(match.group(4)),
                validity=match.group(5),
                validity_score=int(match.group(6)),
                total_score=int(match.group(4)) + int(match.group(6)),
            ),
            "solution_b": MathSolution(steps=match.group(7), answer=int(match.group(8))),
            "feedback_b": MathFeedback(
                adequacy=match.group(9),
                adequacy_score=int(match.group(10)),
                validity=match.group(11),
                validity_score=int(match.group(12)),
                total_score=int(match.group(10)) + int(match.group(12)),
            ),
        }
        return feedbacks

    def _input(self) -> MathInput:
        return MathInput.model_validate(self._input_schema(schema=MathInput.model_json_schema()))

    def _generate(self, _input: MathInput, output_file, prompt_file) -> MathTrainOutput:
        prompt: str = self._build_prompt(question=_input.question)
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
        print(self.template.train_output.format(question=_input.question, **feedbacks), file=output_file, flush=True)
        output = MathTrainOutput(**_input.model_dump(), **feedbacks)
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
                _input: MathInput = MathInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: MathTrainOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class MathTestGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, question: str) -> str:
        prompt: str = str()
        prompt += self.template.test_context.format(question=question)
        prompt += self.template.test_instruction
        return prompt

    def _parse_response(self, response: str) -> MathSolution | None:
        pattern = re.compile(self.template.test_regex)
        match = pattern.search(response)
        if not match:
            return None
        solution = MathSolution(steps=match.group(1), answer=int(match.group(2)))
        return solution

    def _input(self) -> MathInput:
        return MathInput.model_validate(self._input_schema(schema=MathInput.model_json_schema()))

    def _generate(self, _input: MathInput, output_file, prompt_file) -> MathTestOutput:
        prompt: str = self._build_prompt(question=_input.question)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        solution: MathSolution | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt, stop=[self.template.stop])
            solution = self._parse_response(response=response.outputs[0][0])
            if not solution:
                print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert solution
        print(self.template.test_output.format(question=_input.question, solution=solution), file=output_file, flush=True)
        output = MathTestOutput(**_input.model_dump(), solution=solution)
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
                _input: MathInput = MathInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: MathTestOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
