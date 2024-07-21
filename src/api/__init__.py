from .config import APIConfig
from .constants import MODELS
from .huggingface import HuggingFaceAPI
from .openai import OpenAIAPI

API = {
    "openai": OpenAIAPI,
    "huggingface": HuggingFaceAPI,
}


def get_api(args):
    config = APIConfig(**vars(args))
    if config.model not in MODELS:
        raise ValueError(f"{config.model} is not supported.")
    return API[MODELS[config.model][0]](config)


def get_model_name(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"{model} is not supported.")
    return MODELS[model][1]
