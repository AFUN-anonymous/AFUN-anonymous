"""
SAM3 query generation task (default 2-image prompt: obs_frame + contact_frame).

- ``sam3_query_gen`` prompt template (the prompt in paper Appendix C.2).
- ``parse_sam3_response``: post-processor that parses the model's JSON output.
- ``build_requests_from_preprocess``: preprocess output -> CaptionRequest list.
- ``results_to_manifest``: CaptionResult list -> query_gen manifest entries.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

from ...utils import collect_episode_dirs
from ..prompt import IMAGE_PLACEHOLDER, ChatMessages, PromptRegistry, PromptTemplate
from ..schema import CaptionRequest, CaptionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SAM3_SYSTEM_PROMPT = """\
You are an expert at converting robotics task language + visual evidence into a compact open-vocabulary segmentation query for SAM3.

Project context — Affordance Understanding:
This data is used to train an affordance prediction model. Given an RGB image and a task-level instruction (e.g., "Open the cabinet door"), the model must predict (1) the **affordance region** — the spatial area on the object where physical contact should occur, and (2) the **post-contact motion trajectory**. The task description intentionally specifies *what to do*, NOT *where to touch*; the model must learn the functional mapping from task intent to contact region. Therefore, only data samples with clear, physically grounded manipulation targets and meaningful task-level semantics are valuable for training.

Your goal is to produce a short noun phrase that uniquely identifies the **smallest manipulable target part** (e.g., "top-left drawer handle", "right knob", "front latch", "left hinge", "power button") that the robot will operate in the observation image.
You must be extremely concise and precise. You must not describe actions; only describe the target object/part to segment. Prefer concrete part names + discriminative attributes (location, row/column, color, shape, relative position).\
"""

SAM3_USER_PROMPT = """\
You are given:

* `instruction`: the task language instruction for this episode/action interval
* `obs_frame`: the observation frame image (the first frame of the action interval)
* `contact_frame`: the contact frame image (the frame at the gripper-close moment)

Task:
Generate a segmentation text query `sam3_prompt` for SAM3 that will produce a mask of the **minimal target part** the robot is about to manipulate **in `obs_frame`**.

Guidelines:

1. Output a **single short noun phrase** (1-10 words) suitable for open-vocabulary segmentation.
2. When a smaller part is clearly the interaction target, the phrase must refer to a **physical part** (handle/knob/lid/button/lever/edge/latch/hinge/tab/strap/rim) rather than a whole object.
3. Use `contact_frame` to infer the true contact target (where the gripper touches). Use `obs_frame` to phrase it in visible terms.
4. Preserve and include spatial qualifiers if present or inferable (e.g., "top-left", "second row right", "front", "leftmost", "upper", "nearest", "on the right side").
5. If the instruction is ambiguous, resolve it using visual evidence. If still ambiguous, choose the most likely minimal part and add one discriminative qualifier (e.g., color/position).
6. Avoid verbs and action words (open/pull/push/turn). Avoid pronouns ("it", "that"). Avoid long descriptions.
7. Do NOT mention "robot", "gripper", "contact", "frame", "image", "mask", "SAM", or "segmentation".
8. If the target is a drawer/door, prefer **handle** or **edge**. If it's a button/switch, prefer **button**/**switch**. If it's a lid, prefer **lid tab**/**lid edge**. If it's a black cup, prefer **black cup handle**.

Output format (strict):
Return ONLY a JSON object with two keys:
{{"rationale": "<your rationale>", "sam3_prompt": "<your noun phrase>"}}

Where:

* `rationale` is a very concise and very brief explanation of your reasoning.
* `sam3_prompt` is the best phrase.

Hard-Fail Policy (must follow):
- You output {{"rationale": "<your rationale>", "sam3_prompt": null}} ONLY when the instruction and the images are fundamentally incompatible such that selecting a manipulable target part would be guesswork.
- "Fundamentally incompatible" means: the instruction refers to an object/affordance category that is not present in obs_frame AND there is no clear gripper-contact target in contact_frame that could plausibly satisfy the instruction.
- Do NOT fail just because multiple candidates exist; only fail if it is genuinely impossible to identify any plausible target part.

Hard-Fail Affordance-Relevance Filter (must follow):
- You output {{"rationale": "<your rationale>", "sam3_prompt": null}} when the task is NOT useful for affordance model training. This includes:
  1. **No clear physical manipulation target**: the task does not involve contacting and manipulating a specific, localizable part (e.g., "move to the left", "wait", "look around", "navigate to the kitchen").
  2. **Ambiguous / unresolvable target**: even with both images, it is impossible to determine a single, well-defined contact region — e.g., the instruction is too vague ("do something with the stuff on the table") and the images provide no disambiguating evidence.
  3. **Non-rigid / deformable / soft-body object**: the target is a deformable, soft, or fabric-like object that lacks a well-defined rigid part structure. Our project focuses on rigid objects with strong part-level affordances (handles, knobs, lids, buttons, etc.). Filter out tasks involving cloth, fabric, towels, rope, dough, sponges, or similar soft bodies (e.g., "fold the towel", "hang the cloth", "flatten the dough", "squeeze the sponge", ""). Exception: if the task involves grasping a rigid part OF a soft object (e.g., a zipper pull on a jacket), keep it.
  4. **Trivial / non-functional interaction**: the task does not teach a meaningful affordance mapping — e.g., the instruction directly names the exact contact part rather than describing a functional task ("grasp the handle", "touch the knob"), or the task is purely a sensor/state check with no physical manipulation.
  5. **Pick-and-place / whole-object relocation**: the task is simply grasping an entire object and moving it to a different location. These do not teach meaningful part-level affordances because (a) the contact region is any graspable surface rather than a specific functional part, and (b) the post-contact trajectory is generic relocation (lift → translate → place) rather than a functionally determined motion (pull, rotate, press, flip, slide, etc.). Filter out instructions that describe picking up, moving, transferring, or placing an object from one location to another — e.g., "move the cup to the left", "pick up the apple and put it in the bowl", "place the block on the shelf", "put the bottle on the counter", "transfer A to B", "stack the cubes", "sort the objects into the bin". Exception: keep the task if it requires interacting with a specific functional part to achieve the relocation (e.g., "pick up the pot by its handle" names a functional grasp point; "slide the drawer out" involves a handle/edge affordance).
- You strictly filter. Leave out borderline samples. If a task even partially matches one of the above categories, output null. High-quality training data is far more valuable than quantity — a noisy sample hurts the model more than a missing one. Only output a valid sam3_prompt when you are confident the task has a clear, rigid, localizable manipulation target with meaningful task-level semantics.

instruction:
{instruction}\
"""


@PromptRegistry.register("sam3_query_gen")
class SAM3QueryGenPrompt(PromptTemplate):
    """Prompt template for SAM3 segmentation query generation.

    Expected ``CaptionRequest.text_fields``:
        - ``instruction``: the task language description.

    Expected ``CaptionRequest.images`` (order matters):
        - [0]: obs_frame path
        - [1]: contact_frame path
    """

    def build_messages(self, request: CaptionRequest) -> ChatMessages:
        instruction = request.text_fields.get("instruction", "")
        if not instruction:
            raise ValueError(
                f"Request {request.id} missing 'instruction' in text_fields"
            )

        user_text = SAM3_USER_PROMPT.format(instruction=instruction)

        return [
            {"role": "system", "content": SAM3_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    # obs_frame
                    {"type": "image_url", "image_url": {"url": IMAGE_PLACEHOLDER}},
                    # contact_frame
                    {"type": "image_url", "image_url": {"url": IMAGE_PLACEHOLDER}},
                ],
            },
        ]


# ---------------------------------------------------------------------------
# Response parser / post-processor
# ---------------------------------------------------------------------------

def parse_sam3_response(result: CaptionResult) -> CaptionResult:
    """Post-processor: parse the model's raw JSON output into ``result.parsed``.

    Accepts plain JSON, JSON in a markdown code fence, or the first {...}
    block in the text. Anything else becomes status "error" with
    sam3_prompt None.
    """
    raw = result.raw_output
    if not raw:
        result.parsed = {"rationale": "Empty model output", "sam3_prompt": None}
        return result

    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json ... ``` or ``` ... ```
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try to parse as JSON
    try:
        parsed = json.loads(cleaned)
        result.parsed = parsed
        return result
    except json.JSONDecodeError:
        pass

    # Fallback: extract first JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            result.parsed = parsed
            return result
        except json.JSONDecodeError:
            pass

    # Last resort — do NOT pass unparseable output as a prompt;
    # it would produce garbage masks downstream.
    result.parsed = {"rationale": "JSON parse error", "sam3_prompt": None}
    result.status = "error"
    result.error = "JSON parse error"
    return result


# ---------------------------------------------------------------------------
# Data discovery: convert preprocess directory layout → CaptionRequest list
# ---------------------------------------------------------------------------

def build_requests_from_preprocess(
    dataset_roots: List[Path],
    *,
    existing_ids: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = 42,
) -> List[CaptionRequest]:
    """Walk preprocess output directories and build 2-image CaptionRequests.

    Expected layout::

        {dataset_root}/{episode_id}/interval_NNN/metadata.json   (interval_language)
        {dataset_root}/{episode_id}/interval_NNN/cam_0/obs_frame.png
        {dataset_root}/{episode_id}/interval_NNN/cam_0/contact_frame.png

    Intervals in *existing_ids*, without an instruction, or missing a frame
    are skipped. *limit* caps both the episodes scanned and the requests.
    """
    if existing_ids is None:
        existing_ids = set()

    ep_dirs = collect_episode_dirs(dataset_roots, limit, shuffle, seed)

    def _scan_episode(ep_dir: Path) -> List[Dict[str, Any]]:
        """Scan one episode for valid requests. Thread-safe."""
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

            # Skip already-done items early (avoids metadata read + image stat)
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
            obs_path = cam_dir / "obs_frame.png"
            contact_path = cam_dir / "contact_frame.png"
            if not (obs_path.exists() and contact_path.exists()):
                continue

            items.append({
                "id": request_id,
                "obs_path": str(obs_path),
                "contact_path": str(contact_path),
                "instruction": instruction,
                "ep_id": ep_id,
                "iv_name": iv_name,
            })
        return items

    # Scan episodes in parallel (overlaps NFS I/O latency)
    candidates: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_scan_episode, d): d for d in ep_dirs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            candidates.extend(future.result())

    # Sort for deterministic ordering (threads finish in arbitrary order)
    candidates.sort(key=lambda c: c["id"])

    if limit and len(candidates) > limit:
        candidates = candidates[:limit]

    requests = [
        CaptionRequest(
            id=c["id"],
            images=[c["obs_path"], c["contact_path"]],
            text_fields={"instruction": c["instruction"]},
            metadata={"episode_id": c["ep_id"], "interval_name": c["iv_name"]},
        )
        for c in candidates
    ]

    logger.info(
        "Discovered %d new requests (%d existing skipped)",
        len(requests), len(existing_ids),
    )
    return requests


# ---------------------------------------------------------------------------
# Results → manifest conversion
# ---------------------------------------------------------------------------

def results_to_manifest(results: List[CaptionResult]) -> Dict[str, Any]:
    """Convert CaptionResults to query_gen manifest entries keyed by request id."""
    manifest: Dict[str, Any] = {}
    for r in results:
        entry: Dict[str, Any] = {"status": r.status}
        if r.status == "done":
            entry["sam3_prompt"] = r.parsed.get("sam3_prompt", "")
            if "rationale" in r.parsed:
                entry["rationale"] = r.parsed["rationale"]
        else:
            entry["error"] = r.error or "unknown error"
        manifest[r.id] = entry
    return manifest
