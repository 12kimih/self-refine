import itertools
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseEvaluator, BaseGenerator

MAX_SCORE = 27


class DialogFeedback(BaseModel):
    consistency: str = str()
    consistency_score: int = 0
    understand: str = str()
    understand_score: int = 0
    sustain: str = str()
    sustain_score: int = 0
    total_score: int = 0


class DialogIteration(BaseModel):
    n: int = 0
    response: str = str()
    feedback: DialogFeedback = DialogFeedback()


class DialogGenInput(BaseModel):
    dialog: list[str] = Field(title="Conversation history")


class DialogGenOutput(BaseModel):
    dialog: list[str]
    best: DialogIteration
    n: int
    iterations: list[DialogIteration]


class DialogEvalInput(BaseModel):
    dialog: list[str] = Field(title="Conversation history")
    response: str = Field(title="Response")


class DialogEvalOutput(BaseModel):
    dialog: list[str]
    response_a: str
    feedback_a: DialogFeedback
    response_b: str
    feedback_b: DialogFeedback


class DialogGenerator(BaseGenerator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)
        self._setup_initial_examples()
        self._setup_feedback_examples()
        self._setup_refine_examples()

    def _build_dialog(self, dialog: list[str]) -> str:
        tag = {0: "A: ", 1: "B: "}
        tagged = [tag[i % 2] + s for i, s in enumerate(dialog)]
        return "\n".join(tagged)

    def _setup_initial_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.initial_prompt: str = str()
            for line in itertools.islice(f, self.config.n_initial):
                example = DialogEvalOutput.model_validate_json(line)
                self.initial_prompt += self.template.initial_context.format(dialog=self._build_dialog(example.dialog))
                self.initial_prompt += self.template.initial_instruction
                self.initial_prompt += self.template.initial_response.format(response=example.response_b)
                self.initial_prompt += self.template.delim

    def _setup_feedback_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.feedback_prompt: str = str()
            for line in itertools.islice(f, self.config.n_feedback):
                example = DialogEvalOutput.model_validate_json(line)
                self.feedback_prompt += self.template.feedback_context.format(dialog=self._build_dialog(example.dialog), response=example.response_a)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_a.model_dump())
                self.feedback_prompt += self.template.delim
                self.feedback_prompt += self.template.feedback_context.format(dialog=self._build_dialog(example.dialog), response=example.response_b)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_b.model_dump())
                self.feedback_prompt += self.template.delim

    def _setup_refine_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.refine_prompt: str = str()
            for line in itertools.islice(f, self.config.n_refine):
                example = DialogEvalOutput.model_validate_json(line)
                if self.config.refine == 0:
                    self.refine_prompt += self.template.refine_context_0.format(dialog=self._build_dialog(example.dialog), response=example.response_a, **example.feedback_a.model_dump())
                    self.refine_prompt += self.template.refine_instruction_0
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_0.format(response=example.response_b, **example.feedback_b.model_dump())
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 1:
                    self.refine_prompt += self.template.refine_context_1.format(dialog=self._build_dialog(example.dialog), response=example.response_a)
                    self.refine_prompt += self.template.refine_instruction_1
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_1.format(response=example.response_b)
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 2:
                    self.refine_prompt += self.template.refine_context_2.format(dialog=self._build_dialog(example.dialog), response=example.response_a)
                    self.refine_prompt += self.template.refine_instruction_2
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_2.format(response=example.response_b)
                    self.refine_prompt += self.template.delim

    def _build_initial_prompt(self, dialog: list[str]) -> str:
        prompt: str = self.initial_prompt
        prompt += self.template.initial_context.format(dialog=self._build_dialog(dialog))
        prompt += self.template.initial_instruction
        prompt += self.template.initial_query
        return prompt

    def _build_feedback_prompt(self, dialog: list[str], response: str) -> str:
        prompt: str = self.feedback_prompt
        prompt += self.template.feedback_context.format(dialog=self._build_dialog(dialog), response=response)
        prompt += self.template.feedback_instruction
        prompt += self.template.feedback_query
        return prompt

    def _build_refine_prompt(self, dialog: list[str], best: DialogIteration) -> str:
        prompt: str = self.refine_prompt
        if self.config.refine == 0:
            prompt += self.template.refine_context_0.format(dialog=self._build_dialog(dialog), response=best.response, **best.feedback.model_dump())
            prompt += self.template.refine_instruction_0
        elif self.config.refine == 1:
            prompt += self.template.refine_context_1.format(dialog=self._build_dialog(dialog), response=best.response)
            prompt += self.template.refine_instruction_1
        elif self.config.refine == 2:
            prompt += self.template.refine_context_2.format(dialog=self._build_dialog(dialog), response=best.response)
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
        response = match.group(1)
        return response

    def _parse_feedback_response(self, response: str) -> DialogFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = DialogFeedback(
            consistency=match.group(1),
            consistency_score=int(match.group(2)),
            understand=match.group(3),
            understand_score=int(match.group(4)),
            sustain=match.group(5),
            sustain_score=int(match.group(6)),
            total_score=int(match.group(2)) + int(match.group(4)) + int(match.group(6)),
        )
        return feedback

    def _input(self) -> DialogGenInput:
        return DialogGenInput.model_validate(self._input_schema(schema=DialogGenInput.model_json_schema()))

    def _generate(self, _input: DialogGenInput, output_file, prompt_file) -> DialogGenOutput:
        output = DialogGenOutput(**_input.model_dump(), best=DialogIteration(), n=0, iterations=list())

        for n in range(self.config.max_iter):
            prompt: str = self._build_initial_prompt(dialog=_input.dialog) if n == 0 else self._build_refine_prompt(dialog=_input.dialog, best=output.best)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            _response: str | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop, "Feedback"])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                _response = self._parse_initial_response(response=response.outputs[0][0])
                if not _response:
                    print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not _response or self.template.sign in _response:
                break
            print(self.template.initial_output.format(n=n + 1, dialog=self._build_dialog(_input.dialog), response=_response), file=output_file, flush=True)

            prompt: str = self._build_feedback_prompt(dialog=_input.dialog, response=_response)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            feedback: DialogFeedback | None = None
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

            iteration = DialogIteration(n=n + 1, response=_response, feedback=feedback)
            output.n += 1
            output.iterations.append(iteration)
            if iteration.feedback.total_score > output.best.feedback.total_score:
                output.best = iteration
            if output.best.feedback.total_score >= MAX_SCORE:
                break

        print("Best Generation", file=output_file, flush=True)
        print(self.template.initial_output.format(n=output.best.n, dialog=self._build_dialog(output.dialog), response=output.best.response), file=output_file, flush=True)
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
                _input: DialogGenInput = DialogGenInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: DialogGenOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class DialogEvaluator(BaseEvaluator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_dialog(self, dialog: list[str]) -> str:
        tag = {0: "A: ", 1: "B: "}
        tagged = [tag[i % 2] + s for i, s in enumerate(dialog)]
        return "\n".join(tagged)

    def _build_prompt(self, dialog: list[str], response: str) -> str:
        prompt: str = str()
        prompt += self.template.evaluation_context.format(dialog=self._build_dialog(dialog), response=response)
        prompt += self.template.evaluation_instruction
        return prompt

    def _parse_response(self, response: str) -> DialogFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = DialogFeedback(
            consistency=match.group(1),
            consistency_score=int(match.group(2)),
            understand=match.group(3),
            understand_score=int(match.group(4)),
            sustain=match.group(5),
            sustain_score=int(match.group(6)),
            total_score=int(match.group(2)) + int(match.group(4)) + int(match.group(6)),
        )
        return feedback

    def _input(self) -> DialogEvalInput:
        return DialogEvalInput.model_validate(self._input_schema(schema=DialogEvalInput.model_json_schema()))

    def _evaluate(self, _input: DialogEvalInput, output_file, prompt_file) -> DialogFeedback:
        prompt: str = self._build_prompt(dialog=_input.dialog, response=_input.response)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedback: DialogFeedback | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt)
            feedback = self._parse_response(response=response.outputs[0][0])
            if not feedback:
                print(f"ParseError: Evaluation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedback
        print(self.template.evaluation_output.format(dialog=self._build_dialog(_input.dialog), response=_input.response, **feedback.model_dump()), file=output_file, flush=True)
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
                _input = DialogGenOutput.model_validate_json(line)
                print(f">>> Evaluation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Evaluation #{i + 1} >>>", file=prompt_file, flush=True)
                if _input.best.n == 1 or _input.iterations[0].response == _input.best.response:
                    input_a = DialogEvalInput(dialog=_input.dialog, response=_input.iterations[0].response)
                    output_a: DialogFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    output = DialogEvalOutput(dialog=_input.dialog, response_a=input_a.response, feedback_a=output_a, response_b=input_a.response, feedback_b=output_a)
                else:
                    input_a = DialogEvalInput(dialog=_input.dialog, response=_input.iterations[0].response)
                    output_a: DialogFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    input_b = DialogEvalInput(dialog=_input.dialog, response=_input.best.response)
                    output_b: DialogFeedback = self._evaluate(_input=input_b, output_file=output_file, prompt_file=prompt_file)
                    output = DialogEvalOutput(dialog=_input.dialog, response_a=input_a.response, feedback_a=output_a, response_b=input_b.response, feedback_b=output_b)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._evaluate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
