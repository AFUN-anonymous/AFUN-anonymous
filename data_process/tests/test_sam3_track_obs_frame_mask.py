"""track_interval_video must return masks for the frames before the prompt frame (the observation frame is frame 0).

The fake predictor behaves like SAM3: in one session, the backward pass starts from the tracker state
left by the forward pass, where the selected object no longer exists.
"""
import importlib
from pathlib import Path

import numpy as np

M = importlib.import_module("mainprocess.02_5_sam3_best_frame_tracking_concurrent")
N_FRAMES, OBJ = 10, 7


def _outputs(obj_ids):
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    return {"out_obj_ids": np.array(obj_ids), "out_binary_masks": [mask for _ in obj_ids],
            "out_probs": np.array([0.9] * len(obj_ids))}


class FakeSam3Predictor:
    def __init__(self):
        self.forward_done = {}

    def handle_request(self, request):
        if request["type"] == "start_session":
            sid = str(len(self.forward_done))
            self.forward_done[sid] = False
            return {"session_id": sid}
        if request["type"] == "add_prompt":
            return {"outputs": _outputs([OBJ])}
        return {}

    def handle_stream_request(self, request):
        sid, direction, start = request["session_id"], request["propagation_direction"], request["start_frame_index"]
        if direction in ("both", "forward"):
            for i in range(start, N_FRAMES):
                yield {"frame_index": i, "outputs": _outputs([OBJ])}
            self.forward_done[sid] = True
        if direction in ("both", "backward"):
            for i in range(start - 1, -1, -1):
                yield {"frame_index": i, "outputs": _outputs([] if self.forward_done[sid] else [OBJ])}


def test_masks_reach_the_observation_frame():
    result = M.track_interval_video(FakeSam3Predictor(), Path("unused"), "bin lid", None, prompt_frame_idx=6)
    assert result["has_track"]
    assert sorted(result["masks_by_frame"]) == list(range(N_FRAMES))


def test_prompt_on_first_frame_needs_no_backward_pass():
    predictor = FakeSam3Predictor()
    result = M.track_interval_video(predictor, Path("unused"), "bin lid", None, prompt_frame_idx=0)
    assert sorted(result["masks_by_frame"]) == list(range(N_FRAMES))
    assert len(predictor.forward_done) == 1
