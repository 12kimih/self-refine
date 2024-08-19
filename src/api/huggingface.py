import os

import huggingface_hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .base import BaseAPI, Response, Usage
from .constants import HF_DTYPE, HF_QUANT


class StopOnTokens(StoppingCriteria):
    def __init__(self, tokens) -> None:
        super().__init__()
        self.tokens = tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        for t in self.tokens:
            if torch.eq(input_ids[0][-len(t) :], t).all():
                return True
        return False


class HuggingFaceAPI(BaseAPI):
    def __init__(self, config) -> None:
        super().__init__(config=config)
        huggingface_hub.login(token=os.environ["HF_TOKEN"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model, device_map="auto", **HF_DTYPE[self.config.hf_dtype], **HF_QUANT[self.config.hf_quant])

    # take care of SPIECE_UNDERLINE token
    def _tokenize(self, sequences: list[str]):
        tokens = list()
        if "phi" in self.config.model:
            for s in sequences:
                tokens.append(self.tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(self.model.device))
                tokens.append(self.tokenizer(" " + s, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(self.model.device))
        else:
            for s in sequences:
                tokens.append(self.tokenizer(s, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0).to(self.model.device))
                tokens.append(self.tokenizer("\n" + s, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)[2:].to(self.model.device))
        return tokens

    def __call__(
        self,
        inputs: str | list[str] | None = None,
        messages: list[dict[str, str]] | None = None,
        stop: list[str] | None = None,
    ) -> Response:
        if inputs is None and messages is None:
            raise ValueError("API call with neither inputs nor messages.")
        if inputs is None:
            inputs = self.tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True, tokenize=False)  # type: ignore
        assert isinstance(inputs, str) or isinstance(inputs, list)
        tokens = self.tokenizer(inputs, padding=True, return_tensors="pt").to(device=self.model.device)
        input_tokens: int = tokens.input_ids.size(1)
        if self.config.verbose == 1:
            print(f"input_tokens={input_tokens}", flush=True)
        stopping_criteria = StoppingCriteriaList([StopOnTokens(self._tokenize(stop))]) if stop else None
        output_ids = self.model.generate(
            **tokens,
            max_length=self.config.max_length,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            num_beams=self.config.n_beams,
            penalty_alpha=self.config.penalty_alpha,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            num_return_sequences=self.config.n_sequences,
            stopping_criteria=stopping_criteria,
        )
        total_tokens: int = output_ids.size(1)
        output_tokens: int = total_tokens - input_tokens
        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)
        if self.config.verbose == 2:
            print(usage, flush=True)
        outputs = self.tokenizer.batch_decode(sequences=output_ids, skip_special_tokens=True)
        outputs = [outputs[i : i + self.config.n_sequences] for i in range(0, len(outputs), self.config.n_sequences)]
        if isinstance(inputs, str):
            inputs = [inputs]
        for i, p in enumerate(inputs):
            l: int = len(p)
            for j in range(self.config.n_sequences):
                outputs[i][j] = outputs[i][j][l:]
        response = Response(model=self.config.model, outputs=outputs, n=self.config.n_sequences, usage=usage)
        if self.config.verbose == 3:
            print(response, flush=True)
        return response
