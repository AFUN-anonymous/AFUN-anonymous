"""Minimal ZED SVO reader (left RGB + depth). Requires the ZED SDK (pyzed.sl)."""

from typing import Optional, Tuple

import numpy as np

try:
    import pyzed.sl as sl
except (ModuleNotFoundError, ImportError):
    # Raise ImportError so importers can handle the missing SDK.
    raise ImportError(
        "ZED SDK (pyzed.sl) not available. Install ZED SDK: "
        "https://www.stereolabs.com/developers/release"
    )


class SVOReader:
    """Random-access reader for ZED SVO files."""

    def __init__(self, svo_path: str, depth_mode: str = "NEURAL"):
        """
        Args:
            svo_path: Path to the .svo file
            depth_mode: "NEURAL", or "NONE" (no depth, no GPU allocation)
        """
        self.svo_path = svo_path
        self._cam = sl.Camera()
        self._current_frame = -1

        init_params = sl.InitParameters()
        init_params.set_from_svo_file(str(svo_path))
        init_params.svo_real_time_mode = False
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.camera_image_flip = sl.FLIP_MODE.OFF

        depth_modes = {
            "NONE": sl.DEPTH_MODE.NONE,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        init_params.depth_mode = depth_modes.get(depth_mode, sl.DEPTH_MODE.NEURAL)

        status = self._cam.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open SVO file: {status}")

        self._left_img = sl.Mat()
        self._depth_mat = sl.Mat()

        self._camera_info = self._cam.get_camera_information()
        self._frame_count = self._cam.get_svo_number_of_frames()

    def get_fps(self) -> float:
        return self._camera_info.camera_configuration.fps

    def set_frame_position(self, frame_idx: int):
        if frame_idx < 0 or frame_idx >= self._frame_count:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self._frame_count})")
        self._cam.set_svo_position(frame_idx)
        self._current_frame = frame_idx - 1  # incremented by grab()

    def read_frame(
        self,
        frame_idx: Optional[int] = None,
        get_depth: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Read the left view at frame_idx (next frame if None).

        Returns (RGB uint8 HxWx3, depth float32 HxW in mm or None).
        Invalid depth (inf/nan) is set to 0.
        """
        if frame_idx is not None and frame_idx != self._current_frame + 1:
            self.set_frame_position(frame_idx)

        status = self._cam.grab(sl.RuntimeParameters())
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab frame: {status}")
        self._current_frame += 1

        zed_resolution = sl.Resolution(0, 0)  # native resolution
        self._cam.retrieve_image(self._left_img, sl.VIEW.LEFT, resolution=zed_resolution)
        rgb = self._left_img.get_data()[:, :, :3].copy()  # drop alpha
        rgb = rgb[:, :, ::-1]  # BGR -> RGB

        depth = None
        if get_depth:
            self._cam.retrieve_measure(self._depth_mat, sl.MEASURE.DEPTH, resolution=zed_resolution)
            depth = self._depth_mat.get_data().copy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        return rgb, depth

    def close(self):
        if hasattr(self, '_cam'):
            self._cam.close()
