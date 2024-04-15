import itertools
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseEvaluator, BaseGenerator

MAX_SCORE = 36


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


class AcronymIteration(BaseModel):
    n: int = 0
    acronym: str = str()
    feedback: AcronymFeedback = AcronymFeedback()


class AcronymGenInput(BaseModel):
    title: str = Field(title="Title")


class AcronymGenOutput(BaseModel):
    title: str
    best: AcronymIteration
    n: int
    iterations: list[AcronymIteration]


class AcronymEvalInput(BaseModel):
    title: str = Field(title="Title")
    acronym: str = Field(title="Acronym")


class AcronymEvalOutput(BaseModel):
    title: str
    acronym_a: str
    feedback_a: AcronymFeedback
    acronym_b: str
    feedback_b: AcronymFeedback


class AcronymGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)
        self._setup_initial_examples()
        self._setup_feedback_examples()
        self._setup_refine_examples()

    def _setup_initial_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.initial_prompt: str = str()
            for line in itertools.islice(f, self.config.n_initial):
                example = AcronymEvalOutput.model_validate_json(line)
                self.initial_prompt += self.template.initial_context.format(title=example.title)
                self.initial_prompt += self.template.initial_instruction
                self.initial_prompt += self.template.initial_response.format(acronym=example.acronym_b)
                self.initial_prompt += self.template.delim

    def _setup_feedback_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.feedback_prompt: str = str()
            for line in itertools.islice(f, self.config.n_feedback):
                example = AcronymEvalOutput.model_validate_json(line)
                self.feedback_prompt += self.template.feedback_context.format(title=example.title, acronym=example.acronym_a)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_a.model_dump())
                self.feedback_prompt += self.template.delim
                self.feedback_prompt += self.template.feedback_context.format(title=example.title, acronym=example.acronym_b)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_b.model_dump())
                self.feedback_prompt += self.template.delim

    def _setup_refine_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.refine_prompt: str = str()
            for line in itertools.islice(f, self.config.n_refine):
                example = AcronymEvalOutput.model_validate_json(line)
                if self.config.refine == 0:
                    self.refine_prompt += self.template.refine_context_0.format(title=example.title, acronym=example.acronym_a, **example.feedback_a.model_dump())
                    self.refine_prompt += self.template.refine_instruction_0
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_0.format(acronym=example.acronym_b, **example.feedback_b.model_dump())
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 1:
                    self.refine_prompt += self.template.refine_context_1.format(title=example.title, acronym=example.acronym_a)
                    self.refine_prompt += self.template.refine_instruction_1
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_1.format(acronym=example.acronym_b)
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 2:
                    self.refine_prompt += self.template.refine_context_2.format(title=example.title, acronym=example.acronym_a)
                    self.refine_prompt += self.template.refine_instruction_2
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_2.format(acronym=example.acronym_b)
                    self.refine_prompt += self.template.delim

    def _build_initial_prompt(self, title: str) -> str:
        prompt: str = self.initial_prompt
        prompt += self.template.initial_context.format(title=title)
        prompt += self.template.initial_instruction
        prompt += self.template.initial_query
        return prompt

    def _build_feedback_prompt(self, title: str, acronym: str) -> str:
        prompt: str = self.feedback_prompt
        prompt += self.template.feedback_context.format(title=title, acronym=acronym)
        prompt += self.template.feedback_instruction
        prompt += self.template.feedback_query
        return prompt

    def _build_refine_prompt(self, title: str, best: AcronymIteration) -> str:
        prompt: str = self.refine_prompt
        if self.config.refine == 0:
            prompt += self.template.refine_context_0.format(title=title, acronym=best.acronym, **best.feedback.model_dump())
            prompt += self.template.refine_instruction_0
        elif self.config.refine == 1:
            prompt += self.template.refine_context_1.format(title=title, acronym=best.acronym)
            prompt += self.template.refine_instruction_1
        elif self.config.refine == 2:
            prompt += self.template.refine_context_2.format(title=title, acronym=best.acronym)
            prompt += self.template.refine_instruction_2
        if self.config.auto:
            prompt += self.template.refine_instruction_auto
        prompt += "\n\n"
        prompt += self.template.refine_query
        return prompt

    def _parse_initial_response(self, response: str) -> str | None:
        pattern = re.compile(self.template.initial_regex)
        match = pattern.search(response)
        if not match:
            return None
        acronym = match.group(1)
        return acronym

    def _parse_feedback_response(self, response: str) -> AcronymFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = AcronymFeedback(
            relevance=match.group(1),
            relevance_score=int(match.group(2)),
            pronunciation=match.group(3),
            pronunciation_score=int(match.group(4)),
            spelling=match.group(5),
            spelling_score=int(match.group(6)),
            familiarity=match.group(7),
            familiarity_score=int(match.group(8)),
            total_score=int(match.group(2)) + int(match.group(4)) + int(match.group(6)) + int(match.group(8)),
        )
        return feedback

    def _input(self) -> AcronymGenInput:
        return AcronymGenInput.model_validate(self._input_schema(schema=AcronymGenInput.model_json_schema()))

    def _generate(self, _input: AcronymGenInput, output_file, prompt_file) -> AcronymGenOutput:
        output = AcronymGenOutput(**_input.model_dump(), best=AcronymIteration(), n=0, iterations=list())

        for n in range(self.config.max_iter):
            prompt: str = self._build_initial_prompt(title=_input.title) if n == 0 else self._build_refine_prompt(title=_input.title, best=output.best)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            acronym: str | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop, "Feedback"])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                acronym = self._parse_initial_response(response=response.outputs[0][0])
                if not acronym:
                    print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not acronym or self.template.sign in acronym:
                break
            print(self.template.initial_output.format(n=n + 1, title=_input.title, acronym=acronym), file=output_file, flush=True)

            prompt: str = self._build_feedback_prompt(title=_input.title, acronym=acronym)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            feedback: AcronymFeedback | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                feedback = self._parse_feedback_response(response=response.outputs[0][0])
                if not feedback:
                    print(f"ParseError: Feedback format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not feedback:
                break
            print(self.template.feedback_output.format(n=n + 1, **feedback.model_dump()), file=output_file, flush=True)

            iteration = AcronymIteration(n=n + 1, acronym=acronym, feedback=feedback)
            output.n += 1
            output.iterations.append(iteration)
            if iteration.feedback.total_score > output.best.feedback.total_score:
                output.best = iteration
            if output.best.feedback.total_score >= MAX_SCORE:
                break

        print("Best Generation", file=output_file, flush=True)
        print(self.template.initial_output.format(n=output.best.n, title=output.title, acronym=output.best.acronym), file=output_file, flush=True)
        print(self.template.feedback_output.format(n=output.best.n, **output.best.feedback.model_dump()), file=output_file, flush=True)
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
                _input: AcronymGenInput = AcronymGenInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: AcronymGenOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class AcronymEvaluator(BaseEvaluator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, title: str, acronym: str) -> str:
        prompt: str = str()
        prompt += self.template.evaluation_context.format(title=title, acronym=acronym)
        prompt += self.template.evaluation_instruction
        return prompt

    def _parse_response(self, response: str) -> AcronymFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = AcronymFeedback(
            relevance=match.group(1),
            relevance_score=int(match.group(2)),
            pronunciation=match.group(3),
            pronunciation_score=int(match.group(4)),
            spelling=match.group(5),
            spelling_score=int(match.group(6)),
            familiarity=match.group(7),
            familiarity_score=int(match.group(8)),
            total_score=int(match.group(2)) + int(match.group(4)) + int(match.group(6)) + int(match.group(8)),
        )
        return feedback

    def _input(self) -> AcronymEvalInput:
        return AcronymEvalInput.model_validate(self._input_schema(schema=AcronymEvalInput.model_json_schema()))

    def _evaluate(self, _input: AcronymEvalInput, output_file, prompt_file) -> AcronymFeedback:
        prompt: str = self._build_prompt(title=_input.title, acronym=_input.acronym)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedback: AcronymFeedback | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt)
            feedback = self._parse_response(response=response.outputs[0][0])
            if not feedback:
                print(f"ParseError: Evaluation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedback
        print(self.template.evaluation_output.format(title=_input.title, acronym=_input.acronym, **feedback.model_dump()), file=output_file, flush=True)
        return feedback

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
                _input = AcronymGenOutput.model_validate_json(line)
                print(f">>> Evaluation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Evaluation #{i + 1} >>>", file=prompt_file, flush=True)
                if _input.best.n == 1 or _input.iterations[0].acronym == _input.best.acronym:
                    input_a = AcronymEvalInput(title=_input.title, acronym=_input.iterations[0].acronym)
                    output_a: AcronymFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    output = AcronymEvalOutput(title=_input.title, acronym_a=input_a.acronym, feedback_a=output_a, acronym_b=input_a.acronym, feedback_b=output_a)
                else:
                    input_a = AcronymEvalInput(title=_input.title, acronym=_input.iterations[0].acronym)
                    output_a: AcronymFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    input_b = AcronymEvalInput(title=_input.title, acronym=_input.best.acronym)
                    output_b: AcronymFeedback = self._evaluate(_input=input_b, output_file=output_file, prompt_file=prompt_file)
                    output = AcronymEvalOutput(title=_input.title, acronym_a=input_a.acronym, feedback_a=output_a, acronym_b=input_b.acronym, feedback_b=output_b)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._evaluate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
