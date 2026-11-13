"""vLLM offline caption backend (``vllm.LLM.chat``) for Qwen-VL models."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

from ..config import VLMCaptioningConfig
from ..prompt import ChatMessages, PromptRegistry
from ..schema import CaptionRequest, CaptionResult
from .base import CaptionBackend, inline_images

logger = logging.getLogger(__name__)

class VLLMCaptionBackend(CaptionBackend):
    """Offline batch caption generation via vLLM."""

    def __init__(self) -> None:
        self._llm: Any = None               # vllm.LLM
        self._sampling_params: Any = None   # vllm.SamplingParams
        self._config: Optional[VLMCaptioningConfig] = None
        self._prompt_template: Optional[Any] = None

    def setup(self, config: VLMCaptioningConfig) -> None:
        """Load the model into vLLM."""
        # Avoid torch.compile/inductor errors in vLLM's model-inspection subprocess.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

        # Triton scratch-memory allocator, needed by Qwen3.5's custom Triton kernels.
        try:
            import torch
            import triton

            def _triton_alloc(size, alignment, stream):
                return torch.cuda.caching_allocator_alloc(size, stream=stream)

            triton.set_allocator(_triton_alloc)
        except Exception:
            pass

        from vllm import LLM, SamplingParams

        self._config = config
        mc = config.model
        ec = config.engine
        gc = config.generation

        if ec.tensor_parallel_size <= 0:
            import torch as _torch
            ec.tensor_parallel_size = max(1, _torch.cuda.device_count())
            logger.info("Auto-detected tensor_parallel_size = %d", ec.tensor_parallel_size)

        logger.info("Initialising vLLM engine …")
        logger.info("  model            : %s", mc.model_name_or_path)
        logger.info("  dtype            : %s", mc.dtype)
        logger.info("  tensor_parallel  : %d", ec.tensor_parallel_size)
        logger.info("  pipeline_parallel: %d", ec.pipeline_parallel_size)
        logger.info("  gpu_mem_util     : %.2f", ec.gpu_memory_utilization)
        logger.info("  max_model_len    : %s", ec.max_model_len)
        logger.info("  max_num_seqs     : %d", ec.max_num_seqs)

        llm_kwargs: Dict[str, Any] = dict(
            model=mc.model_name_or_path,
            dtype=mc.dtype,
            trust_remote_code=mc.trust_remote_code,
            tensor_parallel_size=ec.tensor_parallel_size,
            pipeline_parallel_size=ec.pipeline_parallel_size,
            gpu_memory_utilization=ec.gpu_memory_utilization,
            max_num_seqs=ec.max_num_seqs,
            enforce_eager=ec.enforce_eager,
            seed=ec.seed,
        )
        if mc.quantization:
            llm_kwargs["quantization"] = mc.quantization
        if ec.max_model_len is not None:
            llm_kwargs["max_model_len"] = ec.max_model_len
        if ec.limit_mm_per_prompt is not None:
            llm_kwargs["limit_mm_per_prompt"] = ec.limit_mm_per_prompt
        if ec.allowed_local_media_path:
            llm_kwargs["allowed_local_media_path"] = ec.allowed_local_media_path
        llm_kwargs.update(ec.extra_kwargs)

        t0 = time.time()
        self._llm = LLM(**llm_kwargs)
        logger.info("vLLM engine ready (%.1fs)", time.time() - t0)

        sp_kwargs: Dict[str, Any] = dict(
            max_tokens=gc.max_new_tokens,
            temperature=gc.temperature,
            top_p=gc.top_p,
            top_k=gc.top_k if gc.top_k > 0 else -1,
            repetition_penalty=gc.repetition_penalty,
        )
        if gc.stop:
            sp_kwargs["stop"] = gc.stop
        self._sampling_params = SamplingParams(**sp_kwargs)

        self._prompt_template = PromptRegistry.get(config.prompt_template)
        logger.info("Prompt template    : %s", config.prompt_template)

    def generate(self, requests: List[CaptionRequest]) -> List[CaptionResult]:
        """Batch-generate captions."""
        if self._llm is None or self._config is None:
            raise RuntimeError("Backend not initialised – call setup() first.")

        gc = self._config.generation

        chat_messages: List[ChatMessages] = []
        valid_indices: List[int] = []           # indices into *requests*
        results_map: Dict[int, CaptionResult] = {}

        for idx, req in enumerate(requests):
            try:
                messages = self._prompt_template(req)
                chat_messages.append(inline_images(messages, req.image_paths()))
                valid_indices.append(idx)
            except Exception as e:
                logger.error("Error building request %s: %s", req.id, e)
                results_map[idx] = CaptionResult(
                    id=req.id,
                    status="error",
                    error=f"Prompt build error: {type(e).__name__}: {e}",
                    metadata=req.metadata,
                )

        if chat_messages:
            logger.info("Running vLLM inference on %d requests …", len(chat_messages))
            t0 = time.time()

            chat_kwargs: Dict[str, Any] = {}
            if gc.enable_thinking is not None:
                chat_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(gc.enable_thinking)}

            try:
                outputs = self._llm.chat(
                    messages=chat_messages,
                    sampling_params=self._sampling_params,
                    **chat_kwargs,
                )
            except (ValueError, RuntimeError) as e:
                # One over-long prompt fails the whole call: mark the batch as errors.
                err_msg = f"Batch inference failed: {type(e).__name__}: {e}"
                logger.error(err_msg)
                for req_idx in valid_indices:
                    req = requests[req_idx]
                    results_map[req_idx] = CaptionResult(
                        id=req.id,
                        status="error",
                        error=err_msg,
                        metadata=req.metadata,
                    )
                return [results_map[i] for i in range(len(requests))]

            elapsed = time.time() - t0
            logger.info(
                "vLLM inference done: %d outputs in %.1fs (%.2f req/s)",
                len(outputs), elapsed, len(outputs) / max(elapsed, 0.01),
            )

            for out_idx, output in enumerate(outputs):
                req_idx = valid_indices[out_idx]
                req = requests[req_idx]
                try:
                    raw_text = output.outputs[0].text.strip()
                    results_map[req_idx] = CaptionResult(
                        id=req.id,
                        status="done",
                        raw_output=raw_text,
                        metadata=req.metadata,
                    )
                except Exception as e:
                    results_map[req_idx] = CaptionResult(
                        id=req.id,
                        status="error",
                        error=f"Output extraction error: {type(e).__name__}: {e}",
                        metadata=req.metadata,
                    )

        return [results_map[i] for i in range(len(requests))]

    def shutdown(self) -> None:
        """Release vLLM resources."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("vLLM engine shut down.")
