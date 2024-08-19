import os

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

# August 18, 2024
MODELS = [
    # Llama
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    # Mistral
    "mistralai/Mistral-7B-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Nemo-Instruct-2407",
    # Gemma
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "google/gemma-1.1-2b-it",
    "google/gemma-1.1-7b-it",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    # Phi
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-small-128k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "microsoft/Phi-3-vision-128k-instruct",
]


def should_retry(exception):
    # OSError: [Errno 28] No space left on device
    if isinstance(exception, OSError) and exception.errno == 28:
        return False
    return True


def after_retry(retry_state):
    print(f"({retry_state.attempt_number:02d}/{retry_state.retry_object.stop.max_attempt_number:02d}) Exception: {retry_state.outcome.exception()}")


@retry(stop=stop_after_attempt(20), wait=wait_fixed(10), retry=retry_if_exception(should_retry), after=after_retry)
def download_with_backoff(*args, **kwargs):
    return snapshot_download(*args, **kwargs)


if __name__ == "__main__":
    load_dotenv()
    login(token=os.environ["HF_TOKEN"])

    for i, model_id in enumerate(MODELS):
        print(f"({i + 1:02d}/{len(MODELS):02d}) Downloading {model_id}...")
        download_with_backoff(repo_id=model_id, allow_patterns="model*.safetensors")
