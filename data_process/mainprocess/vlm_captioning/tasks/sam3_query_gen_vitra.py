"""
SAM3 query generation task, VITRA variant (vitra_epic, vitra_ego4d).

Same 2-image request (obs_frame + contact_frame) and parsing as
``sam3_query_gen``; only the prompt text is adapted to egocentric human video.
"""

from __future__ import annotations

from ..prompt import IMAGE_PLACEHOLDER, ChatMessages, PromptRegistry, PromptTemplate
from ..schema import CaptionRequest


# ---------------------------------------------------------------------------
# Prompt template  (VITRA-specific)
# ---------------------------------------------------------------------------

SAM3_SYSTEM_PROMPT_VITRA = """\
You are an expert at converting task language + visual evidence from egocentric human manipulation videos into a compact open-vocabulary segmentation query for SAM3.

Project context — Affordance Understanding:
This data is used to train an affordance prediction model. Given an RGB image and a task-level instruction (e.g., "Open the cabinet door"), the model must predict (1) the **affordance region** — the spatial area on the object where physical contact should occur, and (2) the **post-contact motion trajectory**. The task description intentionally specifies *what to do*, NOT *where to touch*; the model must learn the functional mapping from task intent to contact region. Therefore, only data samples with clear, physically grounded manipulation targets and meaningful task-level semantics are valuable for training.

Your goal is to produce a short noun phrase that uniquely identifies the **smallest manipulable target part** (e.g., "top-left drawer handle", "right knob", "front latch", "left hinge", "power button") that the person will operate in the observation image.
You must be extremely concise and precise. You must not describe actions; only describe the target object/part to segment. Prefer concrete part names + discriminative attributes (location, row/column, color, shape, relative position).\
"""

SAM3_USER_PROMPT_VITRA = """\
You are given:

* `instruction`: the task language instruction for this episode/action interval
* `obs_frame`: the observation frame image (the first frame of the action interval, from an egocentric / head-mounted camera)
* `contact_frame`: the contact frame image (the frame at the moment the hand contacts the object)

Task:
Generate a segmentation text query `sam3_prompt` for SAM3 that will produce a mask of the **minimal target part** the person is about to manipulate **in `obs_frame`**.

Guidelines:

1. Output a **single short noun phrase** (1-10 words) suitable for open-vocabulary segmentation.
2. When a smaller part is clearly the interaction target, the phrase must refer to a **physical part** (handle/knob/lid/button/lever/edge/latch/hinge/tab/strap/rim) rather than a whole object.
3. Focus on the object/part that the `instruction` and `contact_frame` jointly suggest is being manipulated, NOT other arbitrary objects in the scene.
4. Use `contact_frame` to infer the true contact target (where the hand touches). Use `obs_frame` to phrase it in visible terms.
4. Preserve and include spatial qualifiers if present or inferable (e.g., "top-left", "second row right", "front", "leftmost", "upper", "nearest", "on the right side").
5. Avoid verbs and action words (open/pull/push/turn). Avoid pronouns ("it", "that"). Avoid long descriptions.
6. Do NOT mention "hand", "finger", "person", "contact", "frame", "image", "mask", "SAM", or "segmentation".
7. If the target is a drawer/door, prefer **handle** or **edge**. If it's a button/switch, prefer **button**/**switch**. If it's a lid, prefer **lid tab**/**lid edge**. If it's a black cup, prefer **black cup handle**.
8. **Tool use**: if the instruction mentions a tool the hand operates (e.g., "... with a paintbrush", "... using a knife", "position the power tool onto ..."), the target is the **tool** (specifically its grip/handle — the part the hand holds), NOT the workpiece or the surface being acted upon. The hand touches the tool, not the workpiece; `contact_frame` should confirm this. Examples:
   - "Brush the wooden beam with a paintbrush" → `paintbrush handle`
   - "Position the power tool onto the wheel's lug nut" → `power tool handle`
   - "Cut the bread with a knife" → `knife handle`
   - "Stir the soup with a spoon" → `spoon handle`
   - "Put pencil to pencil sharpener" → `pencil`
   - "Grate the potatoes using a box grater" → `grater handle`
   Direct-hand actions (no intermediate tool mentioned) are unchanged and still follow Guideline 8:
   - "Open the drawer with hands" → `drawer handle` (NOT "hand")
   - "Open the door" → `door handle / knob` (NOT "hand")
   - "Close the pot lid" → `pot lid tab` (NOT "hand")
   

Output format (strict):
Return ONLY a JSON object with two keys:
{{"rationale": "<your rationale>", "sam3_prompt": "<your noun phrase>"}}

Where:

* `rationale` is a very concise and very brief explanation of your reasoning.
* `sam3_prompt` is the best phrase.

Hard-Fail Policy (must follow):
- You output {{"rationale": "<your rationale>", "sam3_prompt": null}} ONLY when the instruction and the images are fundamentally incompatible such that selecting a manipulable target part would be guesswork.
- "Fundamentally incompatible" means: the instruction refers to an object/affordance category that is not present in obs_frame AND there is no clear hand-contact target in contact_frame that could plausibly satisfy the instruction.
- Do NOT fail just because multiple candidates exist; only fail if it is genuinely impossible to identify any plausible target part.

Hard-Fail Affordance-Relevance Filter (must follow):
- You output {{"rationale": "<your rationale>", "sam3_prompt": null}} when the task is NOT useful for affordance model training. This includes:
  1. **No clear physical manipulation target**: the task does not involve contacting and manipulating a specific, localizable part (e.g., "move to the left", "wait", "look around", "navigate to the kitchen").
  2. **Ambiguous / unresolvable target**: even with both images, it is impossible to determine a single, well-defined contact region — e.g., the instruction is too vague ("do something with the stuff on the table") and the images provide no disambiguating evidence.
  3. **Non-rigid / deformable / soft-body object**: the target is a deformable, soft, or fabric-like object that lacks a well-defined rigid part structure. Our project focuses on rigid objects with strong part-level affordances (handles, knobs, lids, buttons, etc.). Filter out tasks involving cloth, fabric, towels, rope, dough, sponges, or similar soft bodies (e.g., "fold the towel", "hang the cloth", "flatten the dough", "squeeze the sponge"). Exception: if the task involves grasping a rigid part OF a soft object (e.g., a zipper pull on a jacket), keep it.
  4. **Trivial / non-functional interaction**: the task does not teach a meaningful affordance mapping — e.g., the instruction directly names the exact contact part rather than describing a functional task ("touch the knob"), or the task is purely a sensor/state check with no physical manipulation.
  5. **Pick-and-place / whole-object relocation**: the task is simply grasping an entire object and moving it to a different location. These do not teach meaningful part-level affordances because (a) the contact region is any graspable surface rather than a specific functional part, and (b) the post-contact trajectory is generic relocation (lift → translate → place) rather than a functionally determined motion (pull, rotate, press, flip, slide, tool use, etc.). Filter out instructions that describe picking up, moving, transferring, or placing an object from one location to another — e.g., "move the cup to the left", "pick up the apple and put it in the bowl", "place the block on the shelf", "put the bottle on the counter", "transfer A to B", "stack the cubes", "sort the objects into the bin". Exception1: keep the task if it requires interacting with a specific functional part to achieve the relocation (e.g., "pick up the pot by its handle" names a functional grasp point; "slide the drawer out" involves a handle/edge affordance). Exception2: if the instruction describes operating a held tool on a workpiece ('cut/slice/stir/wipe/scrub/peel X with a Y'), this is NOT pick-and-place — keep it and output the tool's grip. 
  6. **Affordance mismatch**: the action described does NOT match the designed affordance or intended function of the target object/part. Only keep tasks where the interaction is consistent with what the object part was designed for (e.g., a handle is for pulling/turning, a button is for pressing, a lid is for opening/closing, a knob is for rotating). Filter out tasks where the object is used in an unintended or ad-hoc way that does not reflect its inherent affordance.
  7. **No clear part-level structure**: prefer objects that have well-defined, distinct parts (handle, knob, lid, button, lever, switch, latch, hinge, tab, dial, trigger, cap, nozzle, spout, etc.). Filter out tasks where the target is a featureless whole object with no identifiable functional part — e.g., a plain ball, a flat plate, a solid block. Exception: keep the task if the whole object IS the functional part (e.g., a button, a coin to insert, a key).
  8. **Reaching / pre-contact motion only**: the task describes only reaching toward, approaching, or moving the hand toward an object without actually manipulating it. These do not teach affordance because no functional interaction occurs — e.g., "reach for the cup", "move hand toward the bottle", "extend arm to the shelf". Only keep tasks where active manipulation (push, pull, rotate, press, twist, slide, flip, squeeze, turn, open, close, etc.) is happening.
  9. **Static holding only**: the task describes only holding, gripping, or grasping an object without performing any further functional manipulation. Passive holding does not teach meaningful post-contact trajectory — e.g., "hold the scissors", "grip the mug", "keep holding the bottle". Only keep tasks where the person is actively operating on the object (twist, rotate, press, slide, open, tear, pour, etc.), not merely maintaining a static grasp.
  10. **Overly complex / noisy task or scene**: the task is likely to produce a very noisy training sample because the action or visual scene is too complex to pin down a single clean contact region. Filter out: (a) multi-step or compound actions that involve several distinct contacts in one instruction; (b) cluttered scenes where many visually similar objects make the target ambiguous; (c) unusual or rare tool use whose intended affordance is not standard or widely recognizable. We want only simple, unambiguous, single-contact manipulations.
- If a task even partially matches one of the above categories, output null. You strictly filter. Leave out borderline samples. High-quality training data is far more valuable than quantity — a noisy sample hurts the model more than a missing one. Only output a valid sam3_prompt when you are confident the task has a clear, rigid, localizable manipulation target with meaningful task-level semantics.

instruction:
{instruction}\
"""


@PromptRegistry.register("sam3_query_gen_vitra")
class SAM3QueryGenVitraPrompt(PromptTemplate):
    """Prompt template for SAM3 query generation — VITRA variant.

    Same interface as SAM3QueryGenPrompt; only the system/user prompts differ.
    """

    def build_messages(self, request: CaptionRequest) -> ChatMessages:
        instruction = request.text_fields.get("instruction", "")
        if not instruction:
            raise ValueError(
                f"Request {request.id} missing 'instruction' in text_fields"
            )

        user_text = SAM3_USER_PROMPT_VITRA.format(instruction=instruction)

        return [
            {"role": "system", "content": SAM3_SYSTEM_PROMPT_VITRA},
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
