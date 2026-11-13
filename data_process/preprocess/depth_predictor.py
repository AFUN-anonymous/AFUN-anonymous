"""DA3-Streaming video depth, used by the mainprocess stage sam3_mask_traj_depth.

Requires Depth-Anything-3 (with its da3_streaming/ directory) at
extern/Depth-Anything-3 in the repository root. Depth outputs are float32 millimeters.
"""

from __future__ import annotations

import os
import sys
import glob
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers to ensure external packages are importable
# ---------------------------------------------------------------------------

def _ensure_da3_streaming_on_path() -> None:
    """Add the Depth-Anything-3 parent dir and da3_streaming subdir to sys.path.

    - parent dir  (extern/Depth-Anything-3/)       : makes `import da3_streaming.da3_streaming` work
    - da3_streaming dir (extern/.../da3_streaming/) : lets internal imports like
                                                      `from loop_utils.xxx` resolve correctly
    """
    da3_parent = str(Path(__file__).parent.parent / "extern" / "Depth-Anything-3")
    da3_dir = str(Path(da3_parent) / "da3_streaming")
    # Insert da3_dir first so loop_utils/fastloop are importable inside da3_streaming.py.
    # Then insert da3_parent at index 0 so `import da3_streaming` resolves to the
    # package directory (da3_streaming/__init__.py) rather than da3_streaming.py.
    if da3_dir not in sys.path:
        sys.path.insert(0, da3_dir)
    if da3_parent not in sys.path:
        sys.path.insert(0, da3_parent)


# ---------------------------------------------------------------------------
# DepthPredictor
# ---------------------------------------------------------------------------

class DepthPredictor:
    """Wrapper around DA3_Streaming that keeps one streamer per process."""

    # Cached DA3_Streaming for predict_video. Reusing it across cam_dirs saves
    # ~4-6 s of model reconstruction per call.
    _video_streamer = None
    _video_weights_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Video inference
    # ------------------------------------------------------------------

    @staticmethod
    def _rebind_video_streamer(
        streamer,
        image_dir: str,
        save_dir: str,
        chunk_size: int,
        overlap: int,
    ) -> None:
        """Rebind a cached DA3_Streaming instance to a new work_dir.

        Clears all per-run state populated by DA3_Streaming.__init__ and
        DA3_Streaming.run(), redirects I/O paths, and syncs chunk_size /
        overlap (which sam3_mask_traj_depth clamps dynamically per cam_dir).
        """
        streamer.img_dir = image_dir
        streamer.output_dir = save_dir
        # Per-run state cleared at the top of run().
        streamer.img_list = None
        streamer.chunk_indices = None
        streamer.all_camera_poses = []
        streamer.all_camera_intrinsics = []
        streamer.sim3_list = []
        streamer.loop_sim3_list = []
        streamer.loop_predict_list = []
        streamer.loop_list = []
        # Re-create the tmp result subdirectories under the new save_dir.
        streamer.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        streamer.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        streamer.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        streamer.result_output_dir = os.path.join(save_dir, "results_output")
        streamer.pcd_dir = os.path.join(save_dir, "pcd")
        os.makedirs(streamer.result_unaligned_dir, exist_ok=True)
        os.makedirs(streamer.result_aligned_dir, exist_ok=True)
        os.makedirs(streamer.result_loop_dir, exist_ok=True)
        os.makedirs(streamer.pcd_dir, exist_ok=True)
        # sam3_mask_traj_depth clamps chunk_size to max(n+1, default) and sets overlap=0
        # for short sequences. Propagate both the attributes and the config
        # dict (DA3_Streaming reads both in different code paths).
        streamer.config["Model"]["chunk_size"] = chunk_size
        streamer.config["Model"]["overlap"] = overlap
        streamer.chunk_size = chunk_size
        streamer.overlap = overlap
        streamer.overlap_e = streamer.overlap - streamer.overlap_s

    @classmethod
    def predict_video(
        cls,
        frames: Dict[int, np.ndarray],
        work_dir: Path,
        cfg,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
        """Predict temporally-consistent depth for a sequence of frames.

        Parameters
        ----------
        frames:
            Mapping from original frame index → RGB (H, W, 3) uint8 array.
        work_dir:
            Temporary directory for DA3-Streaming I/O.  Sub-directories
            ``input/`` and ``output/`` are created automatically.
        cfg:
            Hydra OmegaConf config.  Must contain::

                cfg.depth_prediction.video.model        # local dir with model.safetensors + config.json
                cfg.depth_prediction.video.chunk_size   # int, frames per chunk
                cfg.depth_prediction.video.overlap      # int, overlap frames between chunks

        Returns
        -------
        (depth_map, intrinsics_map)
            ``depth_map``: {orig_idx → (H, W) float32 depth in mm}, resized to input RGB
            resolution. Frames for which depth could not be computed are omitted.
            ``intrinsics_map``: {orig_idx → {"fx","fy","cx","cy"}} — DA3's predicted
            intrinsics, scaled from DA3's internal processed resolution to the input
            RGB resolution so they match ``depth_map``.
        """
        _ensure_da3_streaming_on_path()
        from da3_streaming.da3_streaming import DA3_Streaming  # type: ignore

        work_dir = Path(work_dir)
        input_dir = work_dir / "input"
        output_dir = work_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Write frames as sequential PNGs ---
        sorted_frame_idxs = sorted(frames.keys())
        n_frames = len(sorted_frame_idxs)

        # seq_pos → original frame_idx
        seq_to_orig: Dict[int, int] = {}
        for seq_pos, orig_idx in enumerate(sorted_frame_idxs):
            seq_to_orig[seq_pos] = orig_idx
            png_path = input_dir / f"{seq_pos:06d}.png"
            cv2.imwrite(str(png_path), frames[orig_idx])

        # --- Build streaming config ---
        video_cfg = cfg.depth_prediction.video
        chunk_size: int = int(video_cfg.chunk_size)
        overlap: int = int(video_cfg.overlap)
        # model_weights_dir: local directory containing model.safetensors + config.json
        model_weights_dir = Path(str(video_cfg.model))
        da3_safetensors = str(model_weights_dir / "model.safetensors")
        da3_config_json = str(model_weights_dir / "config.json")

        streaming_config = {
            "Weights": {
                "DA3": da3_safetensors,     # path to model.safetensors
                "DA3_CONFIG": da3_config_json,  # path to config.json
                "SALAD": "",                # not used when loop_enable=False
            },
            "Model": {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "loop_chunk_size": 20,
                "loop_enable": False,         # disable loop closure (requires faiss)
                "useDBoW": False,
                "delete_temp_files": False,
                "align_lib": "torch",
                "align_method": "sim3",
                "scale_compute_method": "auto",
                "align_type": "dense",
                "ref_view_strategy": "saddle_balanced",
                "ref_view_strategy_loop": "saddle_balanced",
                "depth_threshold": 15.0,
                "save_depth_conf_result": True,
                "save_debug_info": False,
                "Sparse_Align": {
                    "keypoint_select": "orb",
                    "keypoint_num": 5000,
                },
                "IRLS": {
                    "delta": 0.1,
                    "max_iters": 5,
                    "tol": "1e-9",   # must be a string; da3_streaming uses eval() on this
                },
                "Pointcloud_Save": {
                    "sample_ratio": 0.015,
                    "conf_threshold_coef": 0.75,
                },
            },
            "Loop": {
                "SALAD": {
                    "image_size": [336, 336],
                    "batch_size": 32,
                    "similarity_threshold": 0.85,
                    "top_k": 5,
                    "use_nms": True,
                    "nms_threshold": 25,
                },
                "SIM3_Optimizer": {
                    "lang_version": "python",
                    "max_iterations": 30,
                    "lambda_init": "1e-6",  # must be a string; da3_streaming uses eval() on this
                },
            },
        }

        print(
            f"[DepthPredictor] Starting video prediction: {n_frames} frames, "
            f"chunk_size={chunk_size}, overlap={overlap}"
        )

        # Lazy-load singleton: reuse the streamer across cam_dirs to avoid
        # the ~4-6s per-call DepthAnything3 reconstruction.
        # Invalidate if the weights directory changes.
        weights_key = str(model_weights_dir)
        if cls._video_streamer is None or cls._video_weights_dir != weights_key:
            streamer = DA3_Streaming(
                image_dir=str(input_dir),
                save_dir=str(output_dir),
                config=streaming_config,
            )
            cls._video_streamer = streamer
            cls._video_weights_dir = weights_key
        else:
            streamer = cls._video_streamer
            cls._rebind_video_streamer(
                streamer,
                image_dir=str(input_dir),
                save_dir=str(output_dir),
                chunk_size=chunk_size,
                overlap=overlap,
            )
        streamer.run()

        # --- Read back per-chunk Prediction pickles ---
        # DA3-Streaming writes each chunk's Prediction object as a pickled
        # numpy array at _tmp_results_unaligned/chunk_{idx}.npy. For
        # multi-chunk sequences it ALSO produces _tmp_results_aligned/ with
        # the SIM3-aligned versions. We prefer aligned if present.
        aligned_dir = output_dir / "_tmp_results_aligned"
        unaligned_dir = output_dir / "_tmp_results_unaligned"

        read_dir = aligned_dir if any(aligned_dir.glob("chunk_*.npy")) else unaligned_dir
        chunk_files = sorted(
            glob.glob(str(read_dir / "chunk_*.npy")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )

        depth_map: Dict[int, np.ndarray] = {}
        intrinsics_map: Dict[int, Dict[str, float]] = {}
        seq_pos = 0
        for chunk_path in chunk_files:
            pred = np.load(chunk_path, allow_pickle=True).item()
            # pred is a depth_anything_3.specs.Prediction object with a
            # .depth attribute of shape (N_chunk_frames, H', W') in METERS.
            chunk_depth = np.asarray(pred.depth, dtype=np.float32)
            if chunk_depth.ndim == 2:
                chunk_depth = chunk_depth[None, :, :]
            # pred.intrinsics is (N_chunk_frames, 3, 3) at DA3's internal
            # processed resolution (same (H', W') as chunk_depth before resize).
            chunk_intrinsics = np.asarray(pred.intrinsics, dtype=np.float64)
            if chunk_intrinsics.ndim == 2:
                chunk_intrinsics = chunk_intrinsics[None, :, :]

            # Each chunk covers frames [seq_pos, seq_pos + N_chunk_frames).
            # For multi-chunk sequences with overlap, later chunks contain
            # overlapping frames — we only keep the first occurrence of
            # each seq_pos (first chunk wins). This is safe for single-
            # chunk (no overlap) and conservative for multi-chunk.
            for i in range(chunk_depth.shape[0]):
                if seq_pos not in seq_to_orig:
                    seq_pos += 1
                    continue
                orig_idx = seq_to_orig[seq_pos]
                if orig_idx in depth_map:
                    seq_pos += 1
                    continue

                depth_m = chunk_depth[i]  # (H', W') meters
                K_da3 = chunk_intrinsics[i]  # (3, 3) at DA3 resolution

                ref_frame = frames[orig_idx]
                h_in, w_in = ref_frame.shape[:2]
                h_da3, w_da3 = depth_m.shape
                if (h_da3, w_da3) != (h_in, w_in):
                    depth_m = cv2.resize(depth_m, (w_in, h_in),
                                         interpolation=cv2.INTER_LINEAR)

                # Scale DA3's intrinsics from (h_da3, w_da3) → (h_in, w_in) so
                # they match the resized depth map (and any mask defined at
                # input resolution).
                sx = w_in / float(w_da3)
                sy = h_in / float(h_da3)
                intrinsics_map[orig_idx] = {
                    "fx": float(K_da3[0, 0] * sx),
                    "fy": float(K_da3[1, 1] * sy),
                    "cx": float(K_da3[0, 2] * sx),
                    "cy": float(K_da3[1, 2] * sy),
                }
                depth_map[orig_idx] = (depth_m * 1000.0).astype(np.float32)
                seq_pos += 1

        n_done = len(depth_map)
        print(
            f"[DepthPredictor] Video prediction done: {n_done}/{n_frames} frames"
            f" (read from {read_dir.name})"
        )

        return depth_map, intrinsics_map
