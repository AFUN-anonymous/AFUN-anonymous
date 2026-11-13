"""
vlm_captioning - offline VLM batch inference used by stage 01 (query_gen_vllm).

- config   : dataclass config loaded from configs/*.yaml (model, vLLM engine, sampling).
- schema   : CaptionRequest / CaptionResult containers.
- prompt   : prompt-template registry (templates register by name).
- backend  : vLLM offline backend.
- pipeline : micro-batching orchestrator.
- tasks    : SAM3 query-generation prompt templates and request builders.
"""
