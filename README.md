# Self-Refine

Re-implementation of Self-Refine

Paper: <https://arxiv.org/abs/2303.17651>  
Website: <https://selfrefine.info/>

## Usage

> [!NOTE]
> Make sure that you are in the project directory.

### Set up virtual environment

Choose either [`conda`](#conda) or [`venv`](#venv).

#### `conda`

```bash
conda create -n self-refine python=3.11
```

```bash
conda activate self-refine
```

```bash
conda install pip
```

```bash
pip install -r requirements.txt
```

#### `venv`

> [!CAUTION]
> Python version must be 3.11 or higher.

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Configure `dotenv`

```bash
cp dotenv .env
```

> [!IMPORTANT]
> Replace `YOUR_KEY` inside `.env` file with your own key.

### Run program

> [!NOTE]
> If `model` argument is not specified, the default model is `gpt-3.5-turbo-1106`.
>
> You can also use [Hugging Face](https://huggingface.co/) models. Run `python3 main.py -h` to find further details.

#### To take input from the command line

```bash
python3 main.py --task {acronym,dialog,sentence,sentiment} --generate [--model {gpt-3.5-turbo,gpt-3.5-turbo-16k,gpt-3.5-turbo-instruct,gpt-3.5-turbo-1106,gpt-4,gpt-4-32k,gpt-4-0613,gpt-4-32k-0613,gpt-4-1106-preview}]
```

#### To take input from a dataset file

```bash
python3 main.py --task {acronym,dialog,sentence,sentiment} --generate --file [--model {gpt-3.5-turbo,gpt-3.5-turbo-16k,gpt-3.5-turbo-instruct,gpt-3.5-turbo-1106,gpt-4,gpt-4-32k,gpt-4-0613,gpt-4-32k-0613,gpt-4-1106-preview}]
```

> [!NOTE]
> Outputs are saved in outputs/\<task\>/

### Examples

1. Acronym generation with `gpt-3.5-turbo-1106`, taking input from the command line

    ```bash
    python3 main.py --task acronym --generate
    ```

1. Acronym generation with `gpt-3.5-turbo-1106`, printing prompts

    ```bash
    python3 main.py --task acronym --generate --prompt
    ```

1. Acronym generation with `gpt-3.5-turbo-1106`, taking input from `data/acronym/ml-acronyms-test.jsonl`

    ```bash
    python3 main.py --task acronym --generate --file
    ```

1. Acronym generation with `gpt-4-1106-preview`

    ```bash
    python3 main.py --task acronym --generate --model gpt-4-1106-preview
    ```

1. Acronym generation with `mistralai/Mistral-7B-v0.1`, using `float16` data type

    ```bash
    python3 main.py --task acronym --generate --model mistralai/Mistral-7B-v0.1 --hf_dtype float16
    ```
