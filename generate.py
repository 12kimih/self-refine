from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from data import get_test_generator, get_train_generator
from data.api import get_api, get_model_name
from data.config import get_config
from data.parser import create_parser
from data.template import get_template

parser = create_parser()
args = parser.parse_args()
print(args, flush=True)

api = get_api(args)
config = get_config(args)
template = get_template(args)

data_dir: Path = Path("data") / config.task

if config.train:
    name: str = "train"
    if config.file:
        name: str = Path(config.train_file).stem + "-" + get_model_name(model=api.config.model)

    input_file = open(data_dir / config.train_file, mode="r") if config.file else None
    output_file = open(data_dir / (name + ".txt"), mode="w")
    record_file = open(data_dir / (name + ".jsonl"), mode="w") if config.file else None
    config_file = open(data_dir / (name + "-c.json"), mode="w")
    prompt_file = open(data_dir / (name + "-p.txt"), mode="w") if config.prompt else None

    generator = get_train_generator(task=config.task)(api=api, config=config, template=template)
    generator(input_file=input_file, output_file=output_file, record_file=record_file, config_file=config_file, prompt_file=prompt_file)

    if input_file:
        input_file.close()
    output_file.close()
    if record_file:
        record_file.close()
    config_file.close()
    if prompt_file:
        prompt_file.close()

elif config.test:
    name: str = "test"
    if config.file:
        name: str = Path(config.test_file).stem + "-" + get_model_name(model=api.config.model)

    input_file = open(data_dir / config.test_file, mode="r") if config.file else None
    output_file = open(data_dir / (name + ".txt"), mode="w")
    record_file = open(data_dir / (name + ".jsonl"), mode="w") if config.file else None
    config_file = open(data_dir / (name + "-c.json"), mode="w")
    prompt_file = open(data_dir / (name + "-p.txt"), mode="w") if config.prompt else None

    generator = get_test_generator(task=config.task)(api=api, config=config, template=template)
    generator(input_file=input_file, output_file=output_file, record_file=record_file, config_file=config_file, prompt_file=prompt_file)

    if input_file:
        input_file.close()
    output_file.close()
    if record_file:
        record_file.close()
    config_file.close()
    if prompt_file:
        prompt_file.close()
