"""Model configurations for text-to-gcode experiments."""

from dataclasses import dataclass
from enum import Enum


class ModelType(str, Enum):
    SEQ2SEQ = "seq2seq"  # T5, BART
    CAUSAL = "causal"  # GPT-2, Llama
    DIFFUSION = "diffusion"  # Experimental


@dataclass
class ModelConfig:
    name: str
    hf_id: str
    type: ModelType
    max_input_len: int = 128
    max_output_len: int = 2048
    recommended_batch: int = 4
    recommended_lr: float = 5e-5


# Models to experiment with
MODELS = {
    # Seq2Seq models (encoder-decoder)
    "flan-t5-small": ModelConfig(
        name="flan-t5-small",
        hf_id="google/flan-t5-small",
        type=ModelType.SEQ2SEQ,
        max_output_len=1024,
        recommended_batch=8,
    ),
    "flan-t5-base": ModelConfig(
        name="flan-t5-base",
        hf_id="google/flan-t5-base",
        type=ModelType.SEQ2SEQ,
        max_output_len=1024,
        recommended_batch=4,
    ),
    "flan-t5-large": ModelConfig(
        name="flan-t5-large",
        hf_id="google/flan-t5-large",
        type=ModelType.SEQ2SEQ,
        max_output_len=1024,
        recommended_batch=2,
    ),
    "bart-base": ModelConfig(
        name="bart-base",
        hf_id="facebook/bart-base",
        type=ModelType.SEQ2SEQ,
        max_output_len=1024,
        recommended_batch=4,
    ),
    # Causal LMs (decoder-only)
    "gpt2": ModelConfig(
        name="gpt2",
        hf_id="gpt2",
        type=ModelType.CAUSAL,
        max_output_len=1024,
        recommended_batch=4,
    ),
    "gpt2-medium": ModelConfig(
        name="gpt2-medium",
        hf_id="gpt2-medium",
        type=ModelType.CAUSAL,
        max_output_len=1024,
        recommended_batch=2,
    ),
    "codegen-350m": ModelConfig(
        name="codegen-350m",
        hf_id="Salesforce/codegen-350M-mono",
        type=ModelType.CAUSAL,
        max_output_len=1024,
        recommended_batch=4,
        recommended_lr=2e-5,
    ),
}


def get_model_config(name: str) -> ModelConfig:
    """Get model config by name or HF id."""
    if name in MODELS:
        return MODELS[name]
    # Assume it's a HF id
    for cfg in MODELS.values():
        if cfg.hf_id == name:
            return cfg
    # Return default config for unknown models
    return ModelConfig(
        name=name,
        hf_id=name,
        type=ModelType.SEQ2SEQ,  # Assume seq2seq
    )


def list_models() -> list[str]:
    """List available model names."""
    return list(MODELS.keys())

