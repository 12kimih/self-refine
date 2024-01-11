from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from src import get_evaluator, get_generator
from src.api import get_api, get_model_name
from src.utils import get_file_number
from src.config import get_config
from src.parser import create_parser
from src.template import get_template

parser = create_parser()
args = parser.parse_args()
print(args, flush=True)

api = get_api(args)
config = get_config(args)
template = get_template(args)

data_dir: Path = Path("data") / config.task
output_dir: Path = Path("outputs") / config.task
output_dir.mkdir(parents=True, exist_ok=True)

if config.generate:
    name: str = "generation"
    if config.file:
        name = get_model_name(model=api.config.model)
        match (config.refine, config.auto):
            case (0, False):
                name += "-f"
            case (0, True):
                name += "-fa"
            case (1, False):
                name += "-g"
            case (1, True):
                name += "-ga"
            case (2, False):
                name += "-m"
            case (2, True):
                name += "-ma"
    n: int = get_file_number(dir=output_dir, name=name)
    name = name + f"-{n}"

    input_file = open(data_dir / config.generation_file, mode="r") if config.file else None
    output_file = open(output_dir / (name + ".txt"), mode="w")
    record_file = open(output_dir / (name + ".jsonl"), mode="w") if config.file else None
    config_file = open(output_dir / (name + "-c.json"), mode="w")
    prompt_file = open(output_dir / (name + "-p.txt"), mode="w") if config.prompt else None

    generator = get_generator(task=config.task)(api=api, config=config, template=template)
    generator(input_file=input_file, output_file=output_file, record_file=record_file, config_file=config_file, prompt_file=prompt_file)

    if input_file:
        input_file.close()
    output_file.close()
    if record_file:
        record_file.close()
    config_file.close()
    if prompt_file:
        prompt_file.close()

elif config.evaluate:
    name: str = "evaluation"
    if config.file:
        name = Path(config.evaluation_file).stem + "-e"
    n: int = get_file_number(dir=output_dir, name=name)
    name = name + f"-{n}"

    input_file = open(output_dir / config.evaluation_file, mode="r") if config.file else None
    output_file = open(output_dir / (name + ".txt"), mode="w")
    record_file = open(output_dir / (name + ".jsonl"), mode="w") if config.file else None
    config_file = open(output_dir / (name + "-c.json"), mode="w")
    prompt_file = open(output_dir / (name + "-p.txt"), mode="w") if config.prompt else None

    evaluator = get_evaluator(task=config.task)(api=api, config=config, template=template)
    evaluator(input_file=input_file, output_file=output_file, record_file=record_file, config_file=config_file, prompt_file=prompt_file)

    if input_file:
        input_file.close()
    output_file.close()
    if record_file:
        record_file.close()
    config_file.close()
    if prompt_file:
        prompt_file.close()
