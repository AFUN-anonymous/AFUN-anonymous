"""
Multi-frame request builders for query generation.

``sam3_query_gen.build_requests_from_preprocess`` always sends
[obs_frame.png, contact_frame.png]. A task module whose prompt needs other
frames registers a builder under its prompt-template name::

    register_named_builder("my_template", frame_files=["obs_frame.png", ...])

Stage 01 then calls ``get_builder(prompt_template)`` and falls back to the
2-image builder when nothing is registered.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from tqdm import tqdm

from ...utils import collect_episode_dirs
from ..schema import CaptionRequest

logger = logging.getLogger(__name__)

# Same keyword arguments as ``build_requests_from_preprocess``.
RequestBuilder = Callable[..., List[CaptionRequest]]

_BUILDER_REGISTRY: Dict[str, RequestBuilder] = {}


def get_builder(name: Optional[str]) -> Optional[RequestBuilder]:
    """Return the builder registered under ``name``, or None if absent."""
    if not name:
        return None
    return _BUILDER_REGISTRY.get(name)


def build_multi_frame_requests(
    dataset_roots: List[Path],
    *,
    frame_files: Sequence[str],
    existing_ids: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = 42,
) -> List[CaptionRequest]:
    """Walk preprocess output and build requests with ``len(frame_files)`` images.

    Expected layout::

        {dataset_root}/{episode_id}/interval_NNN/metadata.json   (interval_language)
        {dataset_root}/{episode_id}/interval_NNN/cam_0/{frame_files[i]}

    Intervals in *existing_ids*, without an instruction, or missing any frame
    are skipped, as in the 2-image builder.
    """
    if not frame_files:
        raise ValueError("build_multi_frame_requests: 'frame_files' must be non-empty")
    frame_files = tuple(frame_files)

    if existing_ids is None:
        existing_ids = set()

    ep_dirs = collect_episode_dirs(dataset_roots, limit, shuffle, seed)

    def _scan_episode(ep_dir: Path) -> List[Dict[str, Any]]:
        ep_id = ep_dir.name
        items: List[Dict[str, Any]] = []
        try:
            entries = sorted(os.scandir(ep_dir), key=lambda e: e.name)
        except OSError:
            return items
        for entry in entries:
            if not (entry.is_dir() and entry.name.startswith("interval_")):
                continue
            iv_name = entry.name
            request_id = f"{ep_id}/{iv_name}"

            if request_id in existing_ids:
                continue

            iv_path = Path(entry.path)
            meta_path = iv_path / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            instruction = meta.get("interval_language")
            if not instruction:
                continue

            cam_dir = iv_path / "cam_0"
            image_paths = [cam_dir / fname for fname in frame_files]
            if not all(p.exists() for p in image_paths):
                continue

            items.append({
                "id": request_id,
                "images": [str(p) for p in image_paths],
                "instruction": instruction,
                "ep_id": ep_id,
                "iv_name": iv_name,
            })
        return items

    candidates: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_scan_episode, d): d for d in ep_dirs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            candidates.extend(future.result())

    # Threads finish in arbitrary order; sort for a deterministic request list.
    candidates.sort(key=lambda c: c["id"])

    if limit and len(candidates) > limit:
        candidates = candidates[:limit]

    requests = [
        CaptionRequest(
            id=c["id"],
            images=list(c["images"]),
            text_fields={"instruction": c["instruction"]},
            metadata={"episode_id": c["ep_id"], "interval_name": c["iv_name"]},
        )
        for c in candidates
    ]

    logger.info(
        "Discovered %d new multi-frame requests (%d frames each, %d existing skipped)",
        len(requests), len(frame_files), len(existing_ids),
    )
    return requests


def register_named_builder(name: str, *, frame_files: Sequence[str]) -> RequestBuilder:
    """Register a multi-frame builder for prompt template ``name`` and return it."""
    captured_frames = tuple(frame_files)

    def _builder(
        dataset_roots: List[Path],
        *,
        existing_ids: Optional[Set[str]] = None,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = 42,
    ) -> List[CaptionRequest]:
        return build_multi_frame_requests(
            dataset_roots,
            frame_files=captured_frames,
            existing_ids=existing_ids,
            limit=limit,
            shuffle=shuffle,
            seed=seed,
        )

    _builder.__name__ = f"build_multi_frame_requests__{name}"
    _BUILDER_REGISTRY[name] = _builder
    return _builder
