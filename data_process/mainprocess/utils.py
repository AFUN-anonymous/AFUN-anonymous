import fcntl
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# Dataset roots
# =============================================================================

def get_dataset_roots(cfg: DictConfig) -> List[Path]:
    """Return dataset directories under output_root."""
    if cfg.dataset_name is None:
        dataset_root = Path(cfg.output_root)
        return sorted(d for d in dataset_root.iterdir() if d.is_dir())
    data_path = Path(cfg.output_root) / cfg.dataset_name
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")
    return [data_path]


# =============================================================================
# Manifest helpers
# =============================================================================
# Stage manifests live at {output_root}/{stage}_manifest.json with the format
#   { "unit_key": {"status": "done"|"error"|..., ...stage-specific fields} }

STAGE_SAM3_VIDEO_TRACK = "sam3_video_track"
STAGE_SAM3_MASK_TRAJ = "sam3_mask_traj"

# Default query_gen manifest. It is kept outside output_root so queries can be
# reused across runs; override with vlm.manifest_path (stage 01) and
# sam3_best_frame_track.qg_manifest_path (stage 02.5).
QUERY_GEN_MANIFEST_PATH = _PROJECT_ROOT / "query_gen_cache" / "query_gen_manifest.json"


def manifest_path(output_root: Path, stage: str) -> Path:
    return output_root / f"{stage}_manifest.json"


def load_manifest(output_root: Path, stage: str) -> Dict[str, Any]:
    """Load a stage manifest (returns empty dict if absent)."""
    return load_manifest_at(manifest_path(output_root, stage))


def save_manifest(output_root: Path, stage: str, data: Dict[str, Any]) -> None:
    """Merge *data* into the on-disk stage manifest under a file lock."""
    _locked_merge_json(manifest_path(output_root, stage), data)


def load_manifest_at(path: Path) -> Dict[str, Any]:
    """Load a manifest from an explicit path (returns empty dict if absent)."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest_at(path: Path, data: Dict[str, Any]) -> None:
    """Merge *data* into the manifest at an explicit path under a file lock."""
    _locked_merge_json(path, data)


def _locked_merge_json(path: Path, new_entries: dict) -> None:
    """Read-merge-write JSON under an exclusive file lock (new entries win).

    Safe when several processes save to the same manifest concurrently.
    """
    lock_path = path.with_suffix(".json.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing: dict = {}
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = {}
        existing.update(new_entries)
        _atomic_json_write(path, existing)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _atomic_json_write(path: Path, data: dict) -> None:
    """Write JSON atomically using a unique temp file + rename."""
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp", prefix=path.stem + ".", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def done_keys(manifest: dict) -> Set[str]:
    """Set of all keys whose status is not 'error'."""
    return {k for k, v in manifest.items() if v.get("status") != "error"}


def extract_prompts(qg_manifest: dict) -> Dict[str, str]:
    """Extract {ep_id/iv_name: prompt_string} from the query_gen manifest."""
    prompts: Dict[str, str] = {}
    for key, entry in qg_manifest.items():
        if entry.get("status") == "done":
            p = entry.get("sam3_prompt")
            if isinstance(p, str) and p.strip():
                prompts[key] = p
    return prompts


# =============================================================================
# Dataset preprocessor (frame / depth loading for stages 02.5 and 04)
# =============================================================================

_PREPROCESSOR_CACHE: Dict[Any, Any] = {}

# JSON of the composed `paths` config, set by mainprocess.py, so loaders built
# here (also in Ray / worker processes) see command-line paths.* overrides.
PATHS_ENV = "DATA_PROCESS_PATHS"


def get_dataset_preprocessor(dataset_name: str, output_root: Path, *, extract_depth: Optional[bool] = None):
    """Instantiate and cache the dataset-specific preprocess loader.

    Builds the class from config/dataset/<name>.yaml and config/paths/default.yaml.
    The loader provides load_episode, extract_frame_and_depth, close_episode, etc.

    extract_depth overrides the dataset config's flag; False lets backends that
    allocate GPU memory for depth (e.g. ZED NEURAL mode) skip it.
    """
    from hydra.utils import instantiate

    # Make the local 'datasets' package importable inside Ray workers.
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    cache_key = (dataset_name, extract_depth)
    preprocessor = _PREPROCESSOR_CACHE.get(cache_key)
    if preprocessor is None:
        config_root = _PROJECT_ROOT / "config"
        dataset_cfg_path = config_root / "dataset" / f"{dataset_name}.yaml"
        paths_cfg_path = config_root / "paths" / "default.yaml"

        if not dataset_cfg_path.exists():
            raise FileNotFoundError(f"Dataset config not found for '{dataset_name}': {dataset_cfg_path}")

        dataset_cfg = OmegaConf.load(dataset_cfg_path)
        if extract_depth is not None:
            dataset_cfg.extract_depth = extract_depth
        paths_cfg = OmegaConf.load(paths_cfg_path) if paths_cfg_path.exists() else OmegaConf.create({})
        if os.environ.get(PATHS_ENV):
            paths_cfg = OmegaConf.merge(paths_cfg, json.loads(os.environ[PATHS_ENV]))

        cfg = OmegaConf.create({
            "paths": paths_cfg,
            "output_root": str(output_root),
            "multi_processing": False,
            "num_processes": 1,
            "limit": None,
            "dataset": dataset_cfg,
        })
        OmegaConf.resolve(cfg)
        preprocessor = instantiate({"_target_": cfg.dataset._target_}, cfg)
        _PREPROCESSOR_CACHE[cache_key] = preprocessor
    return preprocessor


# =============================================================================
# Episode discovery
# =============================================================================

def collect_episode_dirs(
    dataset_roots: List[Path], limit: Optional[int] = None, shuffle: bool = False, seed: Optional[int] = 42
) -> List[Path]:
    """Collect episode directories (containing episode_summary.json)."""
    ep_dirs: List[Path] = []
    for root in dataset_roots:
        root_dirs = sorted(root.iterdir())
        if shuffle:
            random.seed(seed)
            random.shuffle(root_dirs)
        for d in tqdm(root_dirs):
            if d.is_dir() and (d / "episode_summary.json").exists():
                ep_dirs.append(d)
                if limit and len(ep_dirs) >= limit:
                    break
        if limit and len(ep_dirs) >= limit:
            break
    return ep_dirs[:limit] if limit else ep_dirs
