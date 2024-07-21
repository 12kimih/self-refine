from pydantic import BaseModel, Field

TASKS = ("acronym", "dialog", "math", "sentence", "sentiment")


class BaseConfig(BaseModel):
    task: str = Field(description="[task] task name", json_schema_extra={"required": True, "choices": list(TASKS)})
    generate: bool = Field(default=False, description="[task] whether or not to generate")
    evaluate: bool = Field(default=False, description="[task] whether or not to evaluate")
    file: bool = Field(default=False, description="[task] whether or not to take input from a file")
    generation_file: str = Field(default=str(), description="[task] input file for generation")
    evaluation_file: str = Field(default=str(), description="[task] input file for evaluation")
    prompt: bool = Field(default=False, description="[task] whether or not to print prompts")
    refine: int = Field(default=0, description="[task] refine level; 0: full refine, 1: generic feedback, 2: multiple sampling")
    auto: bool = Field(default=False, description="[task] whether or not to enable the model to autonomously determine the stopping criteria for generation")
    start: int | None = Field(default=None, description="[task] start index of input file")
    end: int | None = Field(default=None, description="[task] end index of input file")
    max_iter: int = Field(default=4, description="[task] maximum number of iterations when you generate")
    max_calls: int = Field(default=10, description="[task] maximum number of API calls when you fail to parse the response")
    example_file: str = Field(default=str(), description="[task] example file for generation")
    n_initial: int | None = Field(default=None, description="[task] number of examples for initial generation")
    n_feedback: int | None = Field(default=None, description="[task] number of examples for feedback")
    n_refine: int | None = Field(default=None, description="[task] number of examples for refine")


class AcronymConfig(BaseConfig):
    generation_file: str = "ml-acronyms-test.jsonl"
    example_file: str = "ml-acronyms-train-gpt-4.jsonl"
    n_feedback: int | None = 4
    n_refine: int | None = 4


class DialogConfig(BaseConfig):
    generation_file: str = "daily-dialog-test.jsonl"
    example_file: str = "daily-dialog-train-gpt-4.jsonl"
    n_feedback: int | None = 4
    n_refine: int | None = 4


class MathConfig(BaseConfig):
    generation_file: str = "gsm8k-test.jsonl"
    example_file: str = "gsm8k-train-gpt-4.jsonl"
    n_feedback: int | None = 4
    n_refine: int | None = 4


class SentenceConfig(BaseConfig):
    generation_file: str = "commongen-hard-test.jsonl"
    example_file: str = "commongen-hard-train-gpt-4.jsonl"
    n_feedback: int | None = 4
    n_refine: int | None = 4


class SentimentConfig(BaseConfig):
    generation_file: str = "yelp-review-full-test.jsonl"
    example_file: str = "yelp-review-full-train-gpt-4.jsonl"
    n_feedback: int | None = 4
    n_refine: int | None = 4


CONFIG = {
    "acronym": AcronymConfig,
    "dialog": DialogConfig,
    "math": MathConfig,
    "sentence": SentenceConfig,
    "sentiment": SentimentConfig,
}


def get_config(args):
    if args.task not in CONFIG:
        raise ValueError(f"{args.task} is not supported.")
    return CONFIG[args.task](**vars(args))
