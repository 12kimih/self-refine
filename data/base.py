from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    def __init__(self, api, config, template) -> None:
        super().__init__()
        self.api = api
        self.config = config
        self.template = template

    def _input_schema(self, schema):
        inputs = dict()
        for key, info in schema["properties"].items():
            if "required" in info and not info["required"]:
                continue
            value = input(info["title"] + ": ")
            if info["type"] == "array":
                value = value.split(",")
            inputs[key] = value
        return inputs

    @abstractmethod
    def _build_prompt(self):
        pass

    @abstractmethod
    def _parse_response(self):
        pass

    @abstractmethod
    def _generate(self):
        pass

    @abstractmethod
    def __call__(self):
        pass
