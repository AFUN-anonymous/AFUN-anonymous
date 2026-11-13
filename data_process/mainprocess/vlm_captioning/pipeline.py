"""Batch captioning pipeline: micro-batches requests through the vLLM backend."""

from __future__ import annotations

import logging
import math
import time
from typing import Callable, List, Optional

from .backend.base import CaptionBackend
from .backend.vllm_backend import VLLMCaptionBackend
from .config import VLMCaptioningConfig
from .schema import CaptionRequest, CaptionResult

logger = logging.getLogger(__name__)

PostProcessor = Callable[[CaptionResult], CaptionResult]


class CaptionPipeline:
    """Usage::

        pipeline = CaptionPipeline(cfg, post_processor=fn)
        pipeline.setup()
        results = pipeline.run_requests(requests, batch_size=512)
        pipeline.teardown()
    """

    def __init__(
        self,
        config: VLMCaptioningConfig,
        post_processor: Optional[PostProcessor] = None,
    ) -> None:
        self.config = config
        self.post_processor = post_processor
        self._backend: Optional[CaptionBackend] = None

    def setup(self) -> None:
        """Initialise the backend (load model)."""
        self._backend = VLLMCaptionBackend()
        self._backend.setup(self.config)

    def teardown(self) -> None:
        """Release resources."""
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None

    def run_requests(
        self,
        requests: List[CaptionRequest],
        batch_size: Optional[int] = None,
        on_batch_done: Optional[Callable[[List[CaptionResult]], None]] = None,
    ) -> List[CaptionResult]:
        """Run inference on *requests* in micro-batches of *batch_size*.

        batch_size None or <= 0 sends everything at once (vLLM schedules
        internally). *on_batch_done* is called with each batch's results.
        """
        if self._backend is None:
            raise RuntimeError("Pipeline not set up - call setup() first.")

        all_results: List[CaptionResult] = []

        if batch_size is None or batch_size <= 0:
            batch_size = len(requests)

        n_batches = math.ceil(len(requests) / batch_size)
        logger.info(
            "Processing %d requests in %d batch(es) (batch_size=%d)",
            len(requests), n_batches, batch_size,
        )

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(requests))
            batch = requests[start:end]

            logger.info(
                "Batch %d/%d  (%d requests, offset %d)",
                batch_idx + 1, n_batches, len(batch), start,
            )

            t0 = time.time()
            results = self._backend.generate(batch)
            elapsed = time.time() - t0

            if self.post_processor is not None:
                results = [self.post_processor(r) for r in results]

            all_results.extend(results)

            done_count = sum(1 for r in results if r.status == "done")
            err_count = len(results) - done_count
            logger.info(
                "  Batch %d done: %d ok, %d errors (%.1fs, %.2f req/s)",
                batch_idx + 1, done_count, err_count,
                elapsed, len(batch) / max(elapsed, 0.01),
            )

            if on_batch_done is not None:
                on_batch_done(results)

        return all_results
