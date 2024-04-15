import itertools
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseEvaluator, BaseGenerator

MAX_SCORE = 18


class SentimentFeedback(BaseModel):
    effective: str = str()
    effective_score: int = 0
    logical: str = str()
    logical_score: int = 0
    total_score: int = 0


class SentimentIteration(BaseModel):
    n: int = 0
    reversed_review: str = str()
    feedback: SentimentFeedback = SentimentFeedback()


class SentimentGenInput(BaseModel):
    review: str = Field(title="Review", validation_alias="text")


class SentimentGenOutput(BaseModel):
    review: str
    best: SentimentIteration
    n: int
    iterations: list[SentimentIteration]


class SentimentEvalInput(BaseModel):
    review: str = Field(title="Review")
    reversed_review: str = Field(title="Reversed review")


class SentimentEvalOutput(BaseModel):
    review: str
    reversed_review_a: str
    feedback_a: SentimentFeedback
    reversed_review_b: str
    feedback_b: SentimentFeedback


class SentimentGenerator(BaseGenerator):
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
                example = SentimentEvalOutput.model_validate_json(line)
                self.initial_prompt += self.template.initial_context.format(review=example.review)
                self.initial_prompt += self.template.initial_instruction
                self.initial_prompt += self.template.initial_response.format(reversed_review=example.reversed_review_b)
                self.initial_prompt += self.template.delim

    def _setup_feedback_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.feedback_prompt: str = str()
            for line in itertools.islice(f, self.config.n_feedback):
                example = SentimentEvalOutput.model_validate_json(line)
                self.feedback_prompt += self.template.feedback_context.format(review=example.review, reversed_review=example.reversed_review_a)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_a.model_dump())
                self.feedback_prompt += self.template.delim
                self.feedback_prompt += self.template.feedback_context.format(review=example.review, reversed_review=example.reversed_review_b)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_b.model_dump())
                self.feedback_prompt += self.template.delim

    def _setup_refine_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.refine_prompt: str = str()
            for line in itertools.islice(f, self.config.n_refine):
                example = SentimentEvalOutput.model_validate_json(line)
                if self.config.refine == 0:
                    self.refine_prompt += self.template.refine_context_0.format(review=example.review, reversed_review=example.reversed_review_a, **example.feedback_a.model_dump())
                    self.refine_prompt += self.template.refine_instruction_0
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_0.format(reversed_review=example.reversed_review_b, **example.feedback_b.model_dump())
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 1:
                    self.refine_prompt += self.template.refine_context_1.format(review=example.review, reversed_review=example.reversed_review_a)
                    self.refine_prompt += self.template.refine_instruction_1
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_1.format(reversed_review=example.reversed_review_b)
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 2:
                    self.refine_prompt += self.template.refine_context_2.format(review=example.review, reversed_review=example.reversed_review_a)
                    self.refine_prompt += self.template.refine_instruction_2
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_2.format(reversed_review=example.reversed_review_b)
                    self.refine_prompt += self.template.delim

    def _build_initial_prompt(self, review: str) -> str:
        prompt: str = self.initial_prompt
        prompt += self.template.initial_context.format(review=review)
        prompt += self.template.initial_instruction
        prompt += self.template.initial_query
        return prompt

    def _build_feedback_prompt(self, review: str, reversed_review: str) -> str:
        prompt: str = self.feedback_prompt
        prompt += self.template.feedback_context.format(review=review, reversed_review=reversed_review)
        prompt += self.template.feedback_instruction
        prompt += self.template.feedback_query
        return prompt

    def _build_refine_prompt(self, review: str, best: SentimentIteration) -> str:
        prompt: str = self.refine_prompt
        if self.config.refine == 0:
            prompt += self.template.refine_context_0.format(review=review, reversed_review=best.reversed_review, **best.feedback.model_dump())
            prompt += self.template.refine_instruction_0
        elif self.config.refine == 1:
            prompt += self.template.refine_context_1.format(review=review, reversed_review=best.reversed_review)
            prompt += self.template.refine_instruction_1
        elif self.config.refine == 2:
            prompt += self.template.refine_context_2.format(review=review, reversed_review=best.reversed_review)
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
        reversed_review = match.group(1)
        return reversed_review

    def _parse_feedback_response(self, response: str) -> SentimentFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = SentimentFeedback(
            effective=match.group(1),
            effective_score=int(match.group(2)),
            logical=match.group(3),
            logical_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> SentimentGenInput:
        return SentimentGenInput.model_validate(self._input_schema(schema=SentimentGenInput.model_json_schema()))

    def _generate(self, _input: SentimentGenInput, output_file, prompt_file) -> SentimentGenOutput:
        output = SentimentGenOutput(**_input.model_dump(), best=SentimentIteration(), n=0, iterations=list())

        for n in range(self.config.max_iter):
            prompt: str = self._build_initial_prompt(review=_input.review) if n == 0 else self._build_refine_prompt(review=_input.review, best=output.best)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            reversed_review: str | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop, "Feedback"])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                reversed_review = self._parse_initial_response(response=response.outputs[0][0])
                if not reversed_review:
                    print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not reversed_review or self.template.sign in reversed_review:
                break
            print(self.template.initial_output.format(n=n + 1, review=_input.review, reversed_review=reversed_review), file=output_file, flush=True)

            prompt: str = self._build_feedback_prompt(review=_input.review, reversed_review=reversed_review)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            feedback: SentimentFeedback | None = None
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

            iteration = SentimentIteration(n=n + 1, reversed_review=reversed_review, feedback=feedback)
            output.n += 1
            output.iterations.append(iteration)
            if iteration.feedback.total_score > output.best.feedback.total_score:
                output.best = iteration
            if output.best.feedback.total_score >= MAX_SCORE:
                break

        print("Best Generation", file=output_file, flush=True)
        print(self.template.initial_output.format(n=output.best.n, review=output.review, reversed_review=output.best.reversed_review), file=output_file, flush=True)
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
                _input: SentimentGenInput = SentimentGenInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentimentGenOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class SentimentEvaluator(BaseEvaluator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, review: str, reversed_review: str) -> str:
        prompt: str = str()
        prompt += self.template.evaluation_context.format(review=review, reversed_review=reversed_review)
        prompt += self.template.evaluation_instruction
        return prompt

    def _parse_response(self, response: str) -> SentimentFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = SentimentFeedback(
            effective=match.group(1),
            effective_score=int(match.group(2)),
            logical=match.group(3),
            logical_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> SentimentEvalInput:
        return SentimentEvalInput.model_validate(self._input_schema(schema=SentimentEvalInput.model_json_schema()))

    def _evaluate(self, _input: SentimentEvalInput, output_file, prompt_file) -> SentimentFeedback:
        prompt: str = self._build_prompt(review=_input.review, reversed_review=_input.reversed_review)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedback: SentimentFeedback | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt)
            feedback = self._parse_response(response=response.outputs[0][0])
            if not feedback:
                print(f"ParseError: Evaluation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedback
        print(self.template.evaluation_output.format(review=_input.review, reversed_review=_input.reversed_review, **feedback.model_dump()), file=output_file, flush=True)
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
                _input = SentimentGenOutput.model_validate_json(line)
                print(f">>> Evaluation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Evaluation #{i + 1} >>>", file=prompt_file, flush=True)
                if _input.best.n == 1 or _input.iterations[0].reversed_review == _input.best.reversed_review:
                    input_a = SentimentEvalInput(review=_input.review, reversed_review=_input.iterations[0].reversed_review)
                    output_a: SentimentFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    output = SentimentEvalOutput(review=_input.review, reversed_review_a=input_a.reversed_review, feedback_a=output_a, reversed_review_b=input_a.reversed_review, feedback_b=output_a)
                else:
                    input_a = SentimentEvalInput(review=_input.review, reversed_review=_input.iterations[0].reversed_review)
                    output_a: SentimentFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    input_b = SentimentEvalInput(review=_input.review, reversed_review=_input.best.reversed_review)
                    output_b: SentimentFeedback = self._evaluate(_input=input_b, output_file=output_file, prompt_file=prompt_file)
                    output = SentimentEvalOutput(review=_input.review, reversed_review_a=input_a.reversed_review, feedback_a=output_a, reversed_review_b=input_b.reversed_review, feedback_b=output_b)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._evaluate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
