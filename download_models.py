from dotenv import load_dotenv

load_dotenv()

import os

import huggingface_hub

huggingface_hub.login(token=os.environ["HUGGINGFACE_TOKEN"])

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# July 21, 2024
MODELS = [
    # Llama-2
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    # Llama-3
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # Mistral
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    # Gemma
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
    # Gemma-2
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    # Phi
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    # Phi-3
    "microsoft/Phi-3-mini-4k-instruct",
    # "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    # "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    # "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-vision-128k-instruct",
]

for i, m in enumerate(MODELS):
    print(f"({i + 1:02d}/{len(MODELS):02d}) Loading {m}...")
    tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16, trust_remote_code=True)
    del tokenizer, model
    gc.collect()
