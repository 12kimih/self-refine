import itertools
import json
import re
import sys
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from ..base import BaseEvaluator, BaseGenerator

MAX_SCORE = 18


class SentenceFeedback(BaseModel):
    inclusion: str = str()
    inclusion_score: int = 0
    logical: str = str()
    logical_score: int = 0
    total_score: int = 0


class SentenceIteration(BaseModel):
    n: int = 0
    sentence: str = str()
    feedback: SentenceFeedback = SentenceFeedback()


class SentenceGenInput(BaseModel):
    concepts: list[str] = Field(title="Concepts")


class SentenceGenOutput(BaseModel):
    concepts: list[str]
    best: SentenceIteration
    n: int
    iterations: list[SentenceIteration]


class SentenceEvalInput(BaseModel):
    concepts: list[str] = Field(title="Concepts")
    sentence: str = Field(title="Sentence")


class SentenceEvalOutput(BaseModel):
    concepts: list[str]
    sentence_a: str
    feedback_a: SentenceFeedback
    sentence_b: str
    feedback_b: SentenceFeedback


class SentenceGenerator(BaseGenerator):
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
                example = SentenceEvalOutput.model_validate_json(line)
                self.initial_prompt += self.template.initial_context.format(concepts=", ".join(example.concepts))
                self.initial_prompt += self.template.initial_instruction
                self.initial_prompt += self.template.initial_response.format(sentence=example.sentence_b)
                self.initial_prompt += self.template.delim

    def _setup_feedback_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.feedback_prompt: str = str()
            for line in itertools.islice(f, self.config.n_feedback):
                example = SentenceEvalOutput.model_validate_json(line)
                self.feedback_prompt += self.template.feedback_context.format(concepts=", ".join(example.concepts), sentence=example.sentence_a)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_a.model_dump())
                self.feedback_prompt += self.template.delim
                self.feedback_prompt += self.template.feedback_context.format(concepts=", ".join(example.concepts), sentence=example.sentence_b)
                self.feedback_prompt += self.template.feedback_instruction
                self.feedback_prompt += self.template.feedback_response.format(**example.feedback_b.model_dump())
                self.feedback_prompt += self.template.delim

    def _setup_refine_examples(self) -> None:
        path: Path = Path("data") / self.config.task / self.config.example_file
        with open(path, mode="r") as f:
            self.refine_prompt: str = str()
            for line in itertools.islice(f, self.config.n_refine):
                example = SentenceEvalOutput.model_validate_json(line)
                if self.config.refine == 0:
                    self.refine_prompt += self.template.refine_context_0.format(concepts=", ".join(example.concepts), sentence=example.sentence_a, **example.feedback_a.model_dump())
                    self.refine_prompt += self.template.refine_instruction_0
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_0.format(sentence=example.sentence_b, **example.feedback_b.model_dump())
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 1:
                    self.refine_prompt += self.template.refine_context_1.format(concepts=", ".join(example.concepts), sentence=example.sentence_a)
                    self.refine_prompt += self.template.refine_instruction_1
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_1.format(sentence=example.sentence_b)
                    self.refine_prompt += self.template.delim
                elif self.config.refine == 2:
                    self.refine_prompt += self.template.refine_context_2.format(concepts=", ".join(example.concepts), sentence=example.sentence_a)
                    self.refine_prompt += self.template.refine_instruction_2
                    if self.config.auto:
                        self.refine_prompt += self.template.refine_instruction_auto
                    self.refine_prompt += "\n\n"
                    self.refine_prompt += self.template.refine_response_2.format(sentence=example.sentence_b)
                    self.refine_prompt += self.template.delim

    def _build_initial_prompt(self, concepts: list[str]) -> str:
        prompt: str = self.initial_prompt
        prompt += self.template.initial_context.format(concepts=", ".join(concepts))
        prompt += self.template.initial_instruction
        prompt += self.template.initial_query
        return prompt

    def _build_feedback_prompt(self, concepts: list[str], sentence: str) -> str:
        prompt: str = self.feedback_prompt
        prompt += self.template.feedback_context.format(concepts=", ".join(concepts), sentence=sentence)
        prompt += self.template.feedback_instruction
        prompt += self.template.feedback_query
        return prompt

    def _build_refine_prompt(self, concepts: list[str], best: SentenceIteration) -> str:
        prompt: str = self.refine_prompt
        if self.config.refine == 0:
            prompt += self.template.refine_context_0.format(concepts=", ".join(concepts), sentence=best.sentence, **best.feedback.model_dump())
            prompt += self.template.refine_instruction_0
        elif self.config.refine == 1:
            prompt += self.template.refine_context_1.format(concepts=", ".join(concepts), sentence=best.sentence)
            prompt += self.template.refine_instruction_1
        elif self.config.refine == 2:
            prompt += self.template.refine_context_2.format(concepts=", ".join(concepts), sentence=best.sentence)
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
        sentence = match.group(1)
        return sentence

    def _parse_feedback_response(self, response: str) -> SentenceFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = SentenceFeedback(
            inclusion=match.group(1),
            inclusion_score=int(match.group(2)),
            logical=match.group(3),
            logical_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> SentenceGenInput:
        return SentenceGenInput.model_validate(self._input_schema(schema=SentenceGenInput.model_json_schema()))

    def _generate(self, _input: SentenceGenInput, output_file, prompt_file) -> SentenceGenOutput:
        output = SentenceGenOutput(**_input.model_dump(), best=SentenceIteration(), n=0, iterations=list())

        for n in range(self.config.max_iter):
            prompt: str = self._build_initial_prompt(concepts=_input.concepts) if n == 0 else self._build_refine_prompt(concepts=_input.concepts, best=output.best)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            sentence: str | None = None
            for i in range(self.config.max_calls):
                response = self.api(inputs=prompt, stop=[self.template.stop, "Feedback"])
                if response.usage.output_tokens >= 0.95 * self.api.config.max_new_tokens:
                    continue
                sentence = self._parse_initial_response(response=response.outputs[0][0])
                if not sentence:
                    print(f"ParseError: Generation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                    continue
                break
            if not sentence or self.template.sign in sentence:
                break
            print(self.template.initial_output.format(n=n + 1, concepts=", ".join(_input.concepts), sentence=sentence), file=output_file, flush=True)

            prompt: str = self._build_feedback_prompt(concepts=_input.concepts, sentence=sentence)
            if prompt_file:
                print(prompt, file=prompt_file, flush=True)
                print("=" * 200, file=prompt_file, flush=True)
            feedback: SentenceFeedback | None = None
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

            iteration = SentenceIteration(n=n + 1, sentence=sentence, feedback=feedback)
            output.n += 1
            output.iterations.append(iteration)
            if iteration.feedback.total_score > output.best.feedback.total_score:
                output.best = iteration
            if output.best.feedback.total_score >= MAX_SCORE:
                break

        print("Best Generation", file=output_file, flush=True)
        print(self.template.initial_output.format(n=output.best.n, concepts=", ".join(output.concepts), sentence=output.best.sentence), file=output_file, flush=True)
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
                _input: SentenceGenInput = SentenceGenInput.model_validate_json(line)
                print(f">>> Generation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Generation #{i + 1} >>>", file=prompt_file, flush=True)
                output: SentenceGenOutput = self._generate(_input=_input, output_file=output_file, prompt_file=prompt_file)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._generate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)


class SentenceEvaluator(BaseEvaluator):
    def __init__(self, api, config, template) -> None:
        super().__init__(api=api, config=config, template=template)

    def _build_prompt(self, concepts: list[str], sentence: str) -> str:
        prompt: str = str()
        prompt += self.template.evaluation_context.format(concepts=", ".join(concepts), sentence=sentence)
        prompt += self.template.evaluation_instruction
        return prompt

    def _parse_response(self, response: str) -> SentenceFeedback | None:
        pattern = re.compile(self.template.feedback_regex)
        match = pattern.search(response)
        if not match:
            return None
        feedback = SentenceFeedback(
            inclusion=match.group(1),
            inclusion_score=int(match.group(2)),
            logical=match.group(3),
            logical_score=int(match.group(4)),
            total_score=int(match.group(2)) + int(match.group(4)),
        )
        return feedback

    def _input(self) -> SentenceEvalInput:
        return SentenceEvalInput.model_validate(self._input_schema(schema=SentenceEvalInput.model_json_schema()))

    def _evaluate(self, _input: SentenceEvalInput, output_file, prompt_file) -> SentenceFeedback:
        prompt: str = self._build_prompt(concepts=_input.concepts, sentence=_input.sentence)
        if prompt_file:
            print(prompt, file=prompt_file, flush=True)
            print("=" * 200, file=prompt_file, flush=True)
        feedback: SentenceFeedback | None = None
        for i in range(self.config.max_calls):
            response = self.api(inputs=prompt)
            feedback = self._parse_response(response=response.outputs[0][0])
            if not feedback:
                print(f"ParseError: Evaluation format didn't match ({i + 1}). {response}", file=sys.stderr, flush=True)
                continue
            break
        assert feedback
        print(self.template.evaluation_output.format(concepts=", ".join(_input.concepts), sentence=_input.sentence, **feedback.model_dump()), file=output_file, flush=True)
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
                _input = SentenceGenOutput.model_validate_json(line)
                print(f">>> Evaluation #{i + 1} >>>", file=output_file, flush=True)
                if prompt_file:
                    print(f">>> Evaluation #{i + 1} >>>", file=prompt_file, flush=True)
                if _input.best.n == 1 or _input.iterations[0].sentence == _input.best.sentence:
                    input_a = SentenceEvalInput(concepts=_input.concepts, sentence=_input.iterations[0].sentence)
                    output_a: SentenceFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    output = SentenceEvalOutput(concepts=_input.concepts, sentence_a=input_a.sentence, feedback_a=output_a, sentence_b=input_a.sentence, feedback_b=output_a)
                else:
                    input_a = SentenceEvalInput(concepts=_input.concepts, sentence=_input.iterations[0].sentence)
                    output_a: SentenceFeedback = self._evaluate(_input=input_a, output_file=output_file, prompt_file=prompt_file)
                    input_b = SentenceEvalInput(concepts=_input.concepts, sentence=_input.best.sentence)
                    output_b: SentenceFeedback = self._evaluate(_input=input_b, output_file=output_file, prompt_file=prompt_file)
                    output = SentenceEvalOutput(concepts=_input.concepts, sentence_a=input_a.sentence, feedback_a=output_a, sentence_b=input_b.sentence, feedback_b=output_b)
                print(output.model_dump_json(), file=record_file, flush=True)
        else:
            self._evaluate(_input=self._input(), output_file=output_file, prompt_file=prompt_file)
