from pydantic import BaseModel, Field

TASK = ("acronym", "dialog", "math", "sentence", "sentiment")


class BaseConfig(BaseModel):
    task: str = Field(description="[task] task name", json_schema_extra={"required": True, "choices": list(TASK)})
    train: bool = Field(default=False, description="[task] whether or not to generate train labels")
    test: bool = Field(default=False, description="[task] whether or not to generate test labels")
    file: bool = Field(default=False, description="[task] whether or not to take input from a file")
    train_file: str = Field(default=str(), description="[task] input file for train label generation")
    test_file: str = Field(default=str(), description="[task] input file for test label generation")
    prompt: bool = Field(default=False, description="[task] whether or not to print prompts")
    start: int | None = Field(default=None, description="[task] start index of input file")
    end: int | None = Field(default=None, description="[task] end index of input file")
    max_calls: int = Field(default=10, description="[task] maximum number of API calls when you fail to parse the response")


class AcronymConfig(BaseConfig):
    train_file: str = "ml-acronyms-train.jsonl"
    test_file: str = "ml-acronyms-test.jsonl"


class DialogConfig(BaseConfig):
    train_file: str = "daily-dialog-train.jsonl"
    test_file: str = "daily-dialog-test.jsonl"


class MathConfig(BaseConfig):
    train_file: str = "gsm8k-train.jsonl"
    test_file: str = "gsm8k-test.jsonl"


class SentenceConfig(BaseConfig):
    train_file: str = "commongen-hard-train.jsonl"
    test_file: str = "commongen-hard-test.jsonl"


class SentimentConfig(BaseConfig):
    train_file: str = "yelp-review-full-train.jsonl"
    test_file: str = "yelp-review-full-test.jsonl"


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
