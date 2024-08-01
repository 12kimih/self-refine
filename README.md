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
conda update conda
```

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
pip install -U pip setuptools wheel
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
pip install -U pip setuptools wheel
```

```bash
pip install -r requirements.txt
```

### Configure `dotenv`

```bash
cp dotenv .env
```

> [!IMPORTANT]
> Fill in the blanks in the `.env` file according to your personal environment.

### Run program

> [!NOTE]
> If `model` argument is not specified, the default model is `gpt-4o-mini`.
>
> You can also use [Hugging Face](https://huggingface.co/) models. Run `python3 main.py -h` to find further details.

#### To take input from the command line

```bash
python3 main.py --task {acronym,dialog,sentence,sentiment} --generate [--model {gpt-4o,gpt-4o-mini,meta-llama/Meta-Llama-3.1-8B-Instruct,mistralai/Mistral-Nemo-Instruct-2407,google/gemma-2-9b-it,microsoft/Phi-3-medium-128k-instruct,...}]
```

#### To take input from a dataset file

```bash
python3 main.py --task {acronym,dialog,sentence,sentiment} --generate --file [--model {gpt-4o,gpt-4o-mini,meta-llama/Meta-Llama-3.1-8B-Instruct,mistralai/Mistral-Nemo-Instruct-2407,google/gemma-2-9b-it,microsoft/Phi-3-medium-128k-instruct,...}]
```

> [!NOTE]
> Outputs are saved in outputs/\<task\>/

### Examples

1. Acronym generation with `gpt-4o-mini`, taking input from the command line

    ```bash
    python3 main.py --task acronym --generate
    ```

1. Acronym generation with `gpt-4o-mini`, printing prompts

    ```bash
    python3 main.py --task acronym --generate --prompt
    ```

1. Acronym generation with `gpt-4o-mini`, taking input from `data/acronym/ml-acronyms-test.jsonl`

    ```bash
    python3 main.py --task acronym --generate --file
    ```

1. Acronym generation with `gpt-4o`

    ```bash
    python3 main.py --task acronym --generate --model gpt-4o
    ```

1. Acronym generation with `meta-llama/Meta-Llama-3.1-8B-Instruct`, using `float16` data type

    ```bash
    python3 main.py --task acronym --generate --model meta-llama/Meta-Llama-3.1-8B-Instruct --hf_dtype float16
    ```
