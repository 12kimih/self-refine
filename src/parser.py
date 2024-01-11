import argparse

from .config import BaseConfig
from .template import BaseTemplate
from .api.config import APIConfig

TYPE = {
    "boolean": bool,
    "integer": int,
    "number": float,
    "string": str,
}

SCHEMA = (BaseConfig.model_json_schema(), APIConfig.model_json_schema(), BaseTemplate.model_json_schema())


def create_parser(schema=SCHEMA):
    parser = argparse.ArgumentParser()
    for s in schema:
        for arg, info in s["properties"].items():
            kwargs = dict()
            if "type" in info:
                if info["type"] == "array":
                    kwargs["type"] = TYPE[info["items"]["type"]]
                    kwargs["nargs"] = "+"
                else:
                    kwargs["type"] = TYPE[info["type"]]
                    if info["type"] == "boolean":
                        kwargs["action"] = argparse.BooleanOptionalAction
            elif "anyOf" in info:
                kwargs["type"] = TYPE[info["anyOf"][0]["type"]]
            if "required" in info:
                kwargs["required"] = info["required"]
            if "choices" in info:
                kwargs["choices"] = info["choices"]
            if "description" in info:
                kwargs["help"] = info["description"]
            parser.add_argument("--" + arg, default=argparse.SUPPRESS, **kwargs)
    return parser
