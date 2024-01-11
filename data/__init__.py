from .math import MathTestGenerator, MathTrainGenerator
from .dialog import DialogTestGenerator, DialogTrainGenerator
from .acronym import AcronymTestGenerator, AcronymTrainGenerator
from .sentence import SentenceTestGenerator, SentenceTrainGenerator
from .sentiment import SentimentTestGenerator, SentimentTrainGenerator

TRAIN_GENERATOR = {
    "acronym": AcronymTrainGenerator,
    "dialog": DialogTrainGenerator,
    "math": MathTrainGenerator,
    "sentence": SentenceTrainGenerator,
    "sentiment": SentimentTrainGenerator,
}

TEST_GENERATOR = {
    "acronym": AcronymTestGenerator,
    "dialog": DialogTestGenerator,
    "math": MathTestGenerator,
    "sentence": SentenceTestGenerator,
    "sentiment": SentimentTestGenerator,
}


def get_train_generator(task: str):
    if task not in TRAIN_GENERATOR:
        raise ValueError(f"{task} is not supported.")
    return TRAIN_GENERATOR[task]


def get_test_generator(task: str):
    if task not in TEST_GENERATOR:
        raise ValueError(f"{task} is not supported.")
    return TEST_GENERATOR[task]
