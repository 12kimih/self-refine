import argparse

from datasets import interleave_datasets, load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--all", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load all datasets")
parser.add_argument("--acronym", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load an acronym dataset")
parser.add_argument("--dialog", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load a dialog dataset")
parser.add_argument("--math", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load a math dataset")
parser.add_argument("--sentence", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load a sentence dataset")
parser.add_argument("--sentiment", default=False, type=bool, action=argparse.BooleanOptionalAction, help="load a sentiment dataset")
parser.add_argument("--total", default=200, type=int, help="number of total examples")
parser.add_argument("--train", default=10, type=int, help="number of train examples")
parser.add_argument("--seed", default=2024, type=int, help="seed for shuffling datasets")
args = parser.parse_args()


def load_acronym():
    dataset = load_dataset("json", data_files="data/acronym/ml-acronyms.jsonl", split="train")
    shuffled_dataset = dataset.shuffle(seed=args.seed)
    shuffled_dataset.select(range(args.train)).to_json("data/acronym/ml-acronyms-train.jsonl")
    shuffled_dataset.select(range(args.train, args.total)).to_json("data/acronym/ml-acronyms-test.jsonl")


def load_dialog():
    def replace_unicodes(text: str):
        unicodes = {
            "\u2013": "-",
            "\u2014": "-",
            " \u2018 ": "'",
            " \u2019 ": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u3002": ".",
        }
        for k, v in unicodes.items():
            text = text.replace(k, v)
        return text.strip()

    dataset = load_dataset("daily_dialog", split="test")
    dataset.to_json("data/dialog/daily-dialog.jsonl")
    shuffled_dataset = dataset.shuffle(seed=args.seed)
    filtered_dataset = shuffled_dataset.filter(lambda example: len(example["dialog"]) >= 8)
    updated_dataset = filtered_dataset.map(lambda example: {"dialog": [[replace_unicodes(s) for s in l[:5]] for l in example["dialog"]]}, batched=True)
    updated_dataset.select(range(args.train)).to_json("data/dialog/daily-dialog-train.jsonl")
    updated_dataset.select(range(args.train, args.total)).to_json("data/dialog/daily-dialog-test.jsonl")


def load_math():
    dataset = load_dataset("gsm8k", name="main", split="test")
    dataset.to_json("data/math/gsm8k.jsonl")
    shuffled_dataset = dataset.shuffle(seed=args.seed)
    shuffled_dataset.select(range(args.train)).to_json("data/math/gsm8k-train.jsonl")
    shuffled_dataset.select(range(args.train, args.total)).to_json("data/math/gsm8k-test.jsonl")


def load_sentence():
    dataset = load_dataset("json", data_files="data/sentence/commongen-hard.jsonl", split="train")
    shuffled_dataset = dataset.shuffle(seed=args.seed)
    shuffled_dataset.select(range(args.train)).to_json("data/sentence/commongen-hard-train.jsonl")
    shuffled_dataset.select(range(args.train, args.total)).to_json("data/sentence/commongen-hard-test.jsonl")


def load_sentiment():
    dataset = load_dataset("yelp_review_full", split="test")
    dataset.to_json("data/sentiment/yelp-review-full.jsonl")
    shuffled_dataset = dataset.shuffle(seed=args.seed)
    filtered_dataset_n = shuffled_dataset.filter(lambda example: example["label"] == 0 and len(example["text"]) >= 200 and len(example["text"]) <= 300)
    filtered_dataset_p = shuffled_dataset.filter(lambda example: example["label"] == 4 and len(example["text"]) >= 200 and len(example["text"]) <= 300)
    interleave_datasets([filtered_dataset_n.select(range(args.train // 2)), filtered_dataset_p.select(range(args.train // 2))]).to_json("data/sentiment/yelp-review-full-train.jsonl")
    interleave_datasets([filtered_dataset_n.select(range(args.train // 2, args.total // 2)), filtered_dataset_p.select(range(args.train // 2, args.total // 2))]).to_json("data/sentiment/yelp-review-full-test.jsonl")


if args.all:
    load_acronym()
    load_dialog()
    load_math()
    load_sentence()
    load_sentiment()
else:
    if args.acronym:
        load_acronym()
    if args.dialog:
        load_dialog()
    if args.math:
        load_math()
    if args.sentence:
        load_sentence()
    if args.sentiment:
        load_sentiment()
