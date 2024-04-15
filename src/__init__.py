from .tasks.acronym import AcronymEvaluator, AcronymGenerator
from .tasks.dialog import DialogEvaluator, DialogGenerator
from .tasks.math import MathEvaluator, MathGenerator
from .tasks.sentence import SentenceEvaluator, SentenceGenerator
from .tasks.sentiment import SentimentEvaluator, SentimentGenerator

GENERATOR = {
    "acronym": AcronymGenerator,
    "dialog": DialogGenerator,
    "math": MathGenerator,
    "sentence": SentenceGenerator,
    "sentiment": SentimentGenerator,
}

EVALUATOR = {
    "acronym": AcronymEvaluator,
    "dialog": DialogEvaluator,
    "math": MathEvaluator,
    "sentence": SentenceEvaluator,
    "sentiment": SentimentEvaluator,
}


def get_generator(task: str):
    if task not in GENERATOR:
        raise ValueError(f"{task} is not supported.")
    return GENERATOR[task]


def get_evaluator(task: str):
    if task not in EVALUATOR:
        raise ValueError(f"{task} is not supported.")
    return EVALUATOR[task]
