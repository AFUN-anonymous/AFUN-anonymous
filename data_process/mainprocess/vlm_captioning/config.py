"""Typed configuration for the VLM captioning pipeline (see configs/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """Which model to load and how."""

    model_name_or_path: str = "Qwen/Qwen3.5-35B-A3B-FP8"  # HF id or local path
    quantization: Optional[str] = None                    # None = no extra quantization
    dtype: str = "auto"                                   # "auto", "float16", "bfloat16"
    trust_remote_code: bool = True


@dataclass
class EngineConfig:
    """vLLM engine settings; each maps to a ``vllm.LLM`` constructor argument."""

    tensor_parallel_size: int = 1                 # <= 0: use all visible GPUs
    pipeline_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None           # None: model default
    max_num_seqs: int = 64
    limit_mm_per_prompt: Optional[Dict[str, int]] = None  # e.g. {"image": 4}
    allowed_local_media_path: str = "/"
    enforce_eager: bool = False
    seed: int = 42
    # Extra keyword arguments forwarded verbatim to ``vllm.LLM()``.
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Sampling parameters."""

    max_new_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = -1               # -1 = disabled
    repetition_penalty: float = 1.0
    stop: Optional[list[str]] = None
    # Qwen3 thinking mode, passed as chat_template_kwargs; None = model default.
    enable_thinking: Optional[bool] = None


@dataclass
class VLMCaptioningConfig:
    """Root configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt_template: str = "sam3_query_gen"  # registered template name (see tasks/)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLMCaptioningConfig":
        """Build from a nested dictionary (e.g. parsed YAML)."""
        return cls(
            model=ModelConfig(**d.get("model", {})),
            engine=EngineConfig(**{
                k: v for k, v in d.get("engine", {}).items()
                if k != "extra_kwargs"
            }, extra_kwargs=d.get("engine", {}).get("extra_kwargs", {})),
            generation=GenerationConfig(**d.get("generation", {})),
            prompt_template=d.get("prompt_template", "sam3_query_gen"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VLMCaptioningConfig":
        """Load from a YAML file."""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
