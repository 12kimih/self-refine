import torch

# July 21, 2024
MODELS = {
    # GPT-4o
    "gpt-4o": ("openai", "gpt-4o"),
    "gpt-4o-2024-05-13": ("openai", "gpt-4o"),
    # GPT-4o mini
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "gpt-4o-mini-2024-07-18": ("openai", "gpt-4o-mini"),
    # GPT-4 Turbo
    "gpt-4-turbo": ("openai", "gpt-4-turbo"),
    "gpt-4-turbo-2024-04-09": ("openai", "gpt-4-turbo"),
    "gpt-4-turbo-preview": ("openai", "gpt-4-turbo"),
    "gpt-4-0125-preview": ("openai", "gpt-4-turbo"),
    "gpt-4-1106-preview": ("openai", "gpt-4-turbo"),
    # GPT-4
    "gpt-4": ("openai", "gpt-4"),
    "gpt-4-0613": ("openai", "gpt-4"),
    "gpt-4-0314": ("openai", "gpt-4"),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ("openai", "gpt-3.5-turbo"),
    "gpt-3.5-turbo-0125": ("openai", "gpt-3.5-turbo"),
    "gpt-3.5-turbo-1106": ("openai", "gpt-3.5-turbo"),
    "gpt-3.5-turbo-instruct": ("openai", "gpt-3.5-turbo-instruct"),
    # Llama-2
    "meta-llama/Llama-2-7b-hf": ("huggingface", "llama-2-7b"),
    "meta-llama/Llama-2-7b-chat-hf": ("huggingface", "llama-2-7b-chat"),
    "meta-llama/Llama-2-13b-hf": ("huggingface", "llama-2-13b"),
    "meta-llama/Llama-2-13b-chat-hf": ("huggingface", "llama-2-13b-chat"),
    "meta-llama/Llama-2-70b-hf": ("huggingface", "llama-2-70b"),
    "meta-llama/Llama-2-70b-chat-hf": ("huggingface", "llama-2-70b-chat"),
    # Llama-3
    "meta-llama/Meta-Llama-3-8B": ("huggingface", "llama-3-8b"),
    "meta-llama/Meta-Llama-3-8B-Instruct": ("huggingface", "llama-3-8b-instruct"),
    "meta-llama/Meta-Llama-3-70B": ("huggingface", "llama-3-70b"),
    "meta-llama/Meta-Llama-3-70B-Instruct": ("huggingface", "llama-3-70b-instruct"),
    # Mistral
    "mistralai/Mistral-7B-v0.1": ("huggingface", "mistral-7b"),
    "mistralai/Mistral-7B-v0.3": ("huggingface", "mistral-7b"),
    "mistralai/Mistral-7B-Instruct-v0.1": ("huggingface", "mistral-7b-instruct"),
    "mistralai/Mistral-7B-Instruct-v0.2": ("huggingface", "mistral-7b-instruct"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("huggingface", "mistral-7b-instruct"),
    # Mixtral
    "mistralai/Mixtral-8x7B-v0.1": ("huggingface", "mixtral-8x7b"),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("huggingface", "mixtral-8x7b-instruct"),
    "mistralai/Mixtral-8x22B-v0.1": ("huggingface", "mixtral-8x22b"),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ("huggingface", "mixtral-8x22b-instruct"),
    # Gemma
    "google/gemma-2b": ("huggingface", "gemma-2b"),
    "google/gemma-2b-it": ("huggingface", "gemma-2b-instruct"),
    "google/gemma-7b": ("huggingface", "gemma-7b"),
    "google/gemma-7b-it": ("huggingface", "gemma-7b-instruct"),
    "google/gemma-1.1-2b-it": ("huggingface", "gemma-1.1-2b-instruct"),
    "google/gemma-1.1-7b-it": ("huggingface", "gemma-1.1-7b-instruct"),
    # Gemma-2
    "google/gemma-2-9b": ("huggingface", "gemma-2-9b"),
    "google/gemma-2-9b-it": ("huggingface", "gemma-2-9b-instruct"),
    "google/gemma-2-27b": ("huggingface", "gemma-2-27b"),
    "google/gemma-2-27b-it": ("huggingface", "gemma-2-27b-instruct"),
    # Phi
    "microsoft/phi-1": ("huggingface", "phi-1"),
    "microsoft/phi-1_5": ("huggingface", "phi-1.5"),
    "microsoft/phi-2": ("huggingface", "phi-2"),
    # Phi-3
    "microsoft/Phi-3-mini-4k-instruct": ("huggingface", "phi-3-mini-instruct"),
    "microsoft/Phi-3-mini-128k-instruct": ("huggingface", "phi-3-mini-instruct"),
    "microsoft/Phi-3-small-4k-instruct": ("huggingface", "phi-3-small-instruct"),
    "microsoft/Phi-3-small-128k-instruct": ("huggingface", "phi-3-small-instruct"),
    "microsoft/Phi-3-medium-4k-instruct": ("huggingface", "phi-3-medium-instruct"),
    "microsoft/Phi-3-medium-128k-instruct": ("huggingface", "phi-3-medium-instruct"),
    "microsoft/Phi-3-vision-128k-instruct": ("huggingface", "phi-3-vision-instruct"),
}

HF_DTYPE = {
    "default": dict(),
    "float16": {"torch_dtype": torch.float16},
    "bfloat16": {"torch_dtype": torch.bfloat16},
    "float32": {"torch_dtype": torch.float32},
}

HF_QUANT = {
    "default": dict(),
    "4bit": {"load_in_4bit": True},
    "8bit": {"load_in_8bit": True},
    "flash2": {"use_flash_attention_2": True},
}
