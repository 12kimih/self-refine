import itertools
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseEvaluator, BaseGenerator

MAX_SCORE = 18


class MathSolution(BaseModel):
    steps: str = str()
    answer: int = 0


class MathFeedback(BaseModel):
    adequacy: str = str()
    adequacy_score: int = 0
    validity: str = str()
    validity_score: int = 0
    total_score: int = 0


class MathIteration(BaseModel):
    n: int = 0
    solution: MathSolution = MathSolution()
    feedback: MathFeedback = MathFeedback()


class MathGenInput(BaseModel):
    question: str = Field(title="Math problem")


class MathGenOutput(BaseModel):
    question: str
    best: MathIteration
    n: int
    iterations: list[MathIteration]


class MathEvalInput(BaseModel):
    question: str = Field(title="Math problem")
    solution: MathSolution = Field(title="Solution")


class MathEvalOutput(BaseModel):
    question: str
    solution_a: MathSolution
    feedback_a: MathFeedback
    solution_b: MathSolution
    feedback_b: MathFeedback


class MathGenerator(BaseGenerator):
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
                example = MathEvalOutput.model_validate_json(line)
                self.initial_prompt += self.template.initial_context.format(question=example.question)
                self.initial_prompt += self.template.initial_instruction
                self.initial_prompt += self.template.initial_response.format(solution=example.solution_b)
                self.initial_prompt += self.template.delim

    def _setup_feedback_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.feedback_prompt: str = str()
            for line in itertools.islice(f, self.config.n_feedback):
                example = MathEvalOutput.model_validate_json(line)
                self.feedback_prompt += self.template.feedback_context.format(question=example.question, solution=example.solution_a)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_a.model_dump())
                self.feedback_prompt += self.template.delim
                self.feedback_prompt += self.template.feedback_context.format(question=example.question, solution=example.solution_b)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_b.model_dump())
                self.feedback_prompt += self.template.delim

    def _setup_refine_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.refine_prompt: str = str()
            for line in itertools.islice(f, self.config.n_refine):
                example = MathEvalOutput.model_validate_json(line)
                if self.config.refine == 0:
                    self.refine_prompt += self.template.refine_context_0.format(question=example.question, solution=example.solution_a, **example.feedback_a.model_dump())
                    self.refine_prompt += self.template.refine_instruction_0
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_0.format(solution=example.solution_b, **example.feedback_b.model_dump())
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 1:
                    self.refine_prompt += self.template.refine_context_1.format(question=example.question, solution=example.solution_a)
                    self.refine_prompt += self.template.refine_instruction_1
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_1.format(solution=example.solution_b)
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 2:
                    self.refine_prompt += self.template.refine_context_2.format(question=example.question, solution=example.solution_a)
                    self.refine_prompt += self.template.refine_instruction_2
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_2.format(solution=example.solution_b)
                    self.refine_prompt += self.template.delim

    def _build_initial_prompt(self, question: str) -> str:
        prompt: str = self.initial_prompt
        prompt += self.template.initial_context.format(question=question)
        prompt += self.template.initial_instruction
        prompt += self.template.initial_query
        return prompt

    def _build_feedback_prompt(self, question: str, solution: MathSolution) -> str:
        prompt: str = self.feedback_prompt
        prompt += self.template.feedback_context.format(question=question, solution=solution)
        prompt += self.template.feedback_instruction
        prompt += self.template.feedback_query
        return prompt

    def _build_refine_prompt(self, question: str, best: MathIteration) -> str:
        prompt: str = self.refine_prompt
        if self.config.refine == 0:
            prompt += self.template.refine_context_0.format(question=question, solution=best.solution, **best.feedback.model_dump())
            prompt += self.template.refine_instruction_0
        elif self.config.refine == 1:
            prompt += self.template.refine_context_1.format(question=question, solution=best.solution)
            prompt += self.template.refine_instruction_1
        elif self.config.refine == 2:
            prompt += self.template.refine_context_2.format(question=question, solution=best.solution)
            prompt += self.template.refine_instruction_2
        if self.config.auto:
            prompt += self.template.refine_instruction_auto
        prompt += "\n\n"
        prompt += self.template.refine_query
        return prompt

    def _parse_initial_response(self, response: str) -> MathSolution | None:
        pattern = re.compile(self.template.initial_regex)
        match = pattern.search(response)
        if not match:
            return None
        solution = MathSolution(steps=match.group(1), answer=int(match.group(2)))
        return solution

    def _parse_feedback_response(self, response: str) -> MathFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = MathFeedback(
            adequacy=match.group(1),
            adequacy_score=int(match.group(2)),
            validity=match.group(3),
            validity_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> MathGenInput:
        return MathGenInput.model_validate(self._input_schema(schema=MathGenInput.model_json_schema()))

    def _generate(self, _input: MathGenInput, output_file, prompt_file) -> MathGenOutput:
        output = MathGenOutput(**_input.model_dump(), best=MathIteration(), n=0, iterations=list())

        for n in range(self.config.max_iter):
            prompt: str = self._build_initial_prompt(question=_input.question) if n == 0 else self._build_refine_prompt(question=_input.question, best=output.best)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            solution: MathSolution | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop, "Feedback"])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                solution = self._parse_initial_response(response=response.outputs[0][0])
                if not solution:
                    print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not solution or self.template.sign in solution:
                break
            print(self.template.initial_output.format(n=n + 1, question=_input.question, solution=solution), file=output_file, flush=True)

            prompt: str = self._build_feedback_prompt(question=_input.question, solution=solution)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            feedback: MathFeedback | None = None
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

            iteration = MathIteration(n=n + 1, solution=solution, feedback=feedback)
            output.n += 1
            output.iterations.append(iteration)
            if iteration.feedback.total_score > output.best.feedback.total_score:
                output.best = iteration
            if output.best.feedback.total_score >= MAX_SCORE:
                break

        print("Best Generation", file=output_file, flush=True)
        print(self.template.initial_output.format(n=output.best.n, question=output.question, solution=output.best.solution), file=output_file, flush=True)
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
                _input: MathGenInput = MathGenInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: MathGenOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class MathEvaluator(BaseEvaluator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, question: str, solution: MathSolution) -> str:
        prompt: str = str()
        prompt += self.template.evaluation_context.format(question=question, solution=solution)
        prompt += self.template.evaluation_instruction
        return prompt

    def _parse_response(self, response: str) -> MathFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = MathFeedback(
            adequacy=match.group(1),
            adequacy_score=int(match.group(2)),
            validity=match.group(3),
            validity_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> MathEvalInput:
        return MathEvalInput.model_validate(self._input_schema(schema=MathEvalInput.model_json_schema()))

    def _evaluate(self, _input: MathEvalInput, output_file, prompt_file) -> MathFeedback:
        prompt: str = self._build_prompt(question=_input.question, solution=_input.solution)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedback: MathFeedback | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt)
            feedback = self._parse_response(response=response.outputs[0][0])
            if not feedback:
                print(f"ParseError: Evaluation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedback
        print(self.template.evaluation_output.format(question=_input.question, solution=_input.solution, **feedback.model_dump()), file=output_file, flush=True)
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
                _input = MathGenOutput.model_validate_json(line)
                print(f">>> Evaluation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Evaluation #{i + 1} >>>", file=prompt_file, flush=True)
                if _input.best.n == 1 or _input.iterations[0].solution == _input.best.solution:
                    input_a = MathEvalInput(question=_input.question, solution=_input.iterations[0].solution)
                    output_a: MathFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    output = MathEvalOutput(question=_input.question, solution_a=input_a.solution, feedback_a=output_a, solution_b=input_a.solution, feedback_b=output_a)
                else:
                    input_a = MathEvalInput(question=_input.question, solution=_input.iterations[0].solution)
                    output_a: MathFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    input_b = MathEvalInput(question=_input.question, solution=_input.best.solution)
                    output_b: MathFeedback = self._evaluate(_input=input_b, output_file=output_file, prompt_file=prompt_file)
                    output = MathEvalOutput(question=_input.question, solution_a=input_a.solution, feedback_a=output_a, solution_b=input_b.solution, feedback_b=output_b)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._evaluate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
