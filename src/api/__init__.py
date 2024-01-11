from .config import APIConfig
from .openai import OpenAIAPI
from .constants import MODEL
from .huggingface import HuggingFaceAPI

API = {
    "openai": OpenAIAPI,
    "huggingface": HuggingFaceAPI,
}


def get_api(args):
    config = APIConfig(**vars(args))
    if config.model not in MODEL:
        raise ValueError(f"{config.model} is not supported.")
    return API[MODEL[config.model][0]](config)


def get_model_name(model: str) -> str:
    if model not in MODEL:
        raise ValueError(f"{model} is not supported.")
    return MODEL[model][1]
