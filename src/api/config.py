from pydantic import BaseModel, Field

from .constants import HF_DTYPE, HF_QUANT, MODEL


class APIConfig(BaseModel):
    model: str = Field(default="gpt-3.5-turbo-0125", description="[api] text generation model code", json_schema_extra={"choices": list(MODEL)})
    hf_dtype: str = Field(default="default", description="[api] HuggingFace model data type", json_schema_extra={"choices": list(HF_DTYPE)})
    hf_quant: str = Field(default="default", description="[api] HuggingFace model quantization method", json_schema_extra={"choices": list(HF_QUANT)})
    max_length: int = Field(default=4096, description="[api] maximum length of input prompt + max_new_tokens; overridden by max_new_tokens")
    max_new_tokens: int | None = Field(default=400, description="[api] maximum number of tokens to generate")
    do_sample: bool = Field(default=True, description="[api] whether or not to use sampling")
    n_beams: int = Field(default=1, description="[api] number of beams for beam search")
    penalty_alpha: float | None = Field(default=None, description="[api] value used to balance the model confidence and the degeneration penalty in contrastive search decoding")
    temperature: float = Field(default=0.7, description="[api] value used to modulate the next token probabilities")
    top_k: int = Field(default=50, description="[api] number of highest probability vocabulary tokens to keep for top-k-filtering")
    top_p: float = Field(default=0.8, description="[api] only the smallest set of most probable tokens with probabilities that add up to top_p or higher is kept for generation")
    frequency_penalty: float = Field(default=0.0, description="[api] number between -2.0 and 2.0; positive values penalize new tokens based on their existing frequency in the text so far")
    presence_penalty: float = Field(default=0.0, description="[api] number between -2.0 and 2.0; positive values penalize new tokens based on whether they appear in the text so far")
    n_sequences: int = Field(default=1, description="[api] number of independently computed returned sequences for each element in the batch")
    verbose: int = Field(default=0, description="[api] verbosity level; 0: no info, 1: model pre-usage, 2: model post-usage, 3: model response")
