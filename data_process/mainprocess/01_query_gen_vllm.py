#!/usr/bin/env python3
"""
01_query_gen_vllm.py - SAM3 text-query generation with a VLM (Qwen3.5 on vLLM).

For every interval with an instruction and the frames required by the prompt
template, the VLM returns {"rationale", "sam3_prompt"}; sam3_prompt is null when
the sample should be filtered out.

Reads:
    {output_root}/{dataset}/{ep_id}/interval_NNN/metadata.json   (interval_language)
    {output_root}/{dataset}/{ep_id}/interval_NNN/cam_0/<frames>  (see the task module)

Writes:
    query_gen manifest (vlm.manifest_path, default utils.QUERY_GEN_MANIFEST_PATH):
    {"ep_id/interval_NNN": {"status", "sam3_prompt", "rationale"} | {"status": "error", "error"}}

Run:
    python mainprocess.py stage=query_gen_vllm dataset_name=droid
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Set

from .utils import (
    QUERY_GEN_MANIFEST_PATH,
    done_keys,
    get_dataset_roots,
    load_manifest_at,
    save_manifest_at,
)
from .vlm_captioning.config import VLMCaptioningConfig
from .vlm_captioning.pipeline import CaptionPipeline
# Importing the tasks package registers every prompt template and multi-frame builder.
from .vlm_captioning import tasks as _vlm_tasks  # noqa: F401
from .vlm_captioning.tasks.multi_frame_builder import get_builder as _get_request_builder
from .vlm_captioning.tasks.sam3_query_gen import (
    build_requests_from_preprocess,
    parse_sam3_response,
    results_to_manifest,
)

_LOG_FMT = "[%(asctime)s] %(levelname)s %(name)s – %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(format=_LOG_FMT, datefmt=_LOG_DATEFMT, level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_VLM_CONFIG = Path(__file__).parent / "vlm_captioning" / "configs" / "sam3_query_gen.yaml"


def run_query_gen_vllm(cfg: Any) -> None:
    """Stage 01: generate SAM3 text queries with the configured VLM (reads cfg.vlm.*)."""
    output_root = Path(cfg.output_root)
    vlm_cfg = cfg.vlm

    config_path = vlm_cfg.get("config_path") or _DEFAULT_VLM_CONFIG
    vlm_config = VLMCaptioningConfig.from_yaml(config_path)

    dataset_roots = get_dataset_roots(cfg)
    manifest_path = Path(vlm_cfg.get("manifest_path") or QUERY_GEN_MANIFEST_PATH)

    logger.info("=" * 70)
    logger.info("SAM3 Query Generation (model=%s)", vlm_config.model.model_name_or_path)
    logger.info("  output_root    : %s", output_root)
    logger.info("  TP             : %d", vlm_config.engine.tensor_parallel_size)
    logger.info("  dataset dirs   : %s", [d.name for d in dataset_roots])
    logger.info("  manifest       : %s", manifest_path)
    logger.info("=" * 70)

    # --- existing results (skip_existing) ---
    existing_ids: Set[str] = set()
    if cfg.get("skip_existing", True):
        existing_ids = done_keys(load_manifest_at(manifest_path))
        if existing_ids:
            logger.info("Loaded %d existing done entries", len(existing_ids))

    # --- discover requests ---
    # A prompt template may register its own multi-frame builder (see
    # tasks/multi_frame_builder.py); otherwise use the 2-image builder.
    builder = _get_request_builder(vlm_config.prompt_template) or build_requests_from_preprocess
    if builder is not build_requests_from_preprocess:
        logger.info("Using custom request builder for prompt_template=%r", vlm_config.prompt_template)
    requests = builder(
        dataset_roots,
        existing_ids=existing_ids,
        limit=cfg.get("limit"),
        shuffle=cfg.get("shuffle", False),
        seed=cfg.get("seed", 42),
    )
    logger.info("Work items: %d", len(requests))
    if not requests:
        logger.info("Nothing to do.")
        return

    # --- run pipeline, saving the manifest after every micro-batch ---
    manifest = load_manifest_at(manifest_path)

    def _save_manifest_batch(batch_results):
        manifest.update(results_to_manifest(batch_results))
        save_manifest_at(manifest_path, manifest)
        logger.info("Manifest saved (%d total entries)", len(manifest))

    pipeline = CaptionPipeline(vlm_config, post_processor=parse_sam3_response)
    pipeline.setup()
    try:
        t0 = time.time()
        results = pipeline.run_requests(
            requests,
            batch_size=vlm_cfg.get("batch_size"),
            on_batch_done=_save_manifest_batch,
        )
        elapsed = time.time() - t0
    finally:
        pipeline.teardown()

    done_count = sum(1 for r in results if r.status == "done")
    err_count = len(results) - done_count
    logger.info("=" * 70)
    logger.info(
        "Done: %d items (%d ok, %d err) in %.1fs",
        len(results), done_count, err_count, elapsed,
    )
    logger.info("Manifest      → %s", manifest_path)
    logger.info("=" * 70)
