import torch

MODEL = {
    "gpt-3.5-turbo": ("openai", "gpt-3.5"),
    "gpt-3.5-turbo-instruct": ("openai", "gpt-3.5-instruct"),
    "gpt-3.5-turbo-0125": ("openai", "gpt-3.5"),
    "gpt-4": ("openai", "gpt-4"),
    "gpt-4-32k": ("openai", "gpt-4"),
    "gpt-4-turbo-preview": ("openai", "gpt-4"),
    "gpt-4-vision-preview": ("openai", "gpt-4-vision"),
    "gpt-4-0125-preview": ("openai", "gpt-4"),
    "meta-llama/Llama-2-7b-hf": ("huggingface", "llama-2-7b"),
    "meta-llama/Llama-2-7b-chat-hf": ("huggingface", "llama-2-7b-chat"),
    "meta-llama/Llama-2-13b-hf": ("huggingface", "llama-2-13b"),
    "meta-llama/Llama-2-13b-chat-hf": ("huggingface", "llama-2-13b-chat"),
    "meta-llama/Llama-2-70b-hf": ("huggingface", "llama-2-70b"),
    "meta-llama/Llama-2-70b-chat-hf": ("huggingface", "llama-2-70b-chat"),
    "codellama/CodeLlama-7b-hf": ("huggingface", "codellama-7b"),
    "codellama/CodeLlama-7b-Python-hf": ("huggingface", "codellama-7b-python"),
    "codellama/CodeLlama-7b-Instruct-hf": ("huggingface", "codellama-7b-instruct"),
    "codellama/CodeLlama-13b-hf": ("huggingface", "codellama-13b"),
    "codellama/CodeLlama-13b-Python-hf": ("huggingface", "codellama-13b-python"),
    "codellama/CodeLlama-13b-Instruct-hf": ("huggingface", "codellama-13b-instruct"),
    "codellama/CodeLlama-34b-hf": ("huggingface", "codellama-34b"),
    "codellama/CodeLlama-34b-Python-hf": ("huggingface", "codellama-34b-python"),
    "codellama/CodeLlama-34b-Instruct-hf": ("huggingface", "codellama-34b-instruct"),
    "mistralai/Mistral-7B-v0.1": ("huggingface", "mistral-7b-v0.1"),
    "mistralai/Mistral-7B-Instruct-v0.1": ("huggingface", "mistral-7b-instruct-v0.1"),
    "mistralai/Mistral-7B-Instruct-v0.2": ("huggingface", "mistral-7b-instruct-v0.2"),
    "mistralai/Mixtral-8x7B-v0.1": ("huggingface", "mixtral-8x7b-v0.1"),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ("huggingface", "mixtral-8x7b-instruct-v0.1"),
    "HuggingFaceH4/zephyr-7b-alpha": ("huggingface", "zephyr-7b-alpha"),
    "HuggingFaceH4/zephyr-7b-beta": ("huggingface", "zephyr-7b-beta"),
    "openchat/openchat_3.5": ("huggingface", "openchat-3.5"),
    "openchat/openchat-3.5-1210": ("huggingface", "openchat-3.5"),
    "microsoft/phi-1": ("huggingface", "phi-1"),
    "microsoft/phi-1_5": ("huggingface", "phi-1.5"),
    "microsoft/phi-2": ("huggingface", "phi-2"),
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
