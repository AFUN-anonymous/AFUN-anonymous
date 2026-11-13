"""Re-render CALVIN dataset at 640x640 using PyBullet state restoration.

For each original NPZ frame, restores the full scene state (robot joints,
block poses, table articulations, lights) and renders the static camera
at 640x640. Outputs new NPZ files with high-res rgb_static + depth_static
and all non-image fields copied verbatim. Gripper/tactile cameras are dropped.

Optional task filtering (--task-filter config/dataset/calvin_task_filter.yaml):
only frames of kept tasks (plus a context margin) are re-rendered.

Needs pybullet and the CALVIN env assets (calvin_env/data with the URDFs).

Usage:
    python tools/calvin_rerender_640.py \\
        --src /path/to/Calvin/task_ABCD_D \\
        --dst /path/to/Calvin/calvin_ABCD_D_640 \\
        --assets /path/to/Calvin/calvin_env_assets/data \\
        [--splits training validation] \\
        [--width 640] [--height 640] \\
        [--margin-seconds 2.0] \\
        [--task-filter config/dataset/calvin_task_filter.yaml]
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Camera config from CALVIN env D (identical across A-D)
# ---------------------------------------------------------------------------
STATIC_CAM = {
    'fov': 10,
    'aspect': 1,
    'nearval': 0.01,
    'farval': 10,
    'look_at': [-0.026242351159453392, -0.0302329882979393, 0.3920000493526459],
    'look_from': [2.871459009488717, -2.166602199425597, 2.555159848480571],
    'up_vector': [0.4041403970338857, 0.22629790978217404, 0.8862616969685161],
}

ROBOT_CFG = {
    'filename': 'franka_panda/panda_longer_finger.urdf',
    'base_position': [-0.34, -0.46, 0.24],
    'base_orientation': [0, 0, 0],
    'arm_joint_ids': [0, 1, 2, 3, 4, 5, 6],
    'gripper_joint_ids': [9, 11],
    'initial_joint_positions': [
        -1.21779206, 1.03987646, 2.11978261,
        -2.34205014, -0.87015947, 1.64119353, 0.55344866,
    ],
}

SCENE_CFG = {
    'plane_urdf': 'plane/plane.urdf',
    'global_scaling': 0.8,
    'block_urdfs': {
        'block_red': 'blocks/block_red_middle.urdf',
        'block_blue': 'blocks/block_blue_small.urdf',
        'block_pink': 'blocks/block_pink_big.urdf',
    },
}

# Per-environment table URDFs
ENV_TABLE_URDFS = {
    'calvin_scene_A': 'calvin_table_A/urdf/calvin_table_A.urdf',
    'calvin_scene_B': 'calvin_table_B/urdf/calvin_table_B.urdf',
    'calvin_scene_C': 'calvin_table_C/urdf/calvin_table_C.urdf',
    'calvin_scene_D': 'calvin_table_D/urdf/calvin_table_D.urdf',
}
DEFAULT_TABLE_URDF = 'calvin_table_D/urdf/calvin_table_D.urdf'

# scene_obs layout for env D (from PlayTableScene.parse_scene_obs):
#   Ordered as: doors(2), buttons(1), switches(1), lights(2), objects(18)
#   [0] = base__slide (door), [1] = base__drawer (door),
#   [2] = base__button (button), [3] = base__switch (switch),
#   [4] = lightbulb state (0/1), [5] = led state (0/1),
#   [6:12] = block_red (pos3 + euler3),
#   [12:18] = block_blue (pos3 + euler3),
#   [18:24] = block_pink (pos3 + euler3)

# scene_obs index → table joint name mapping
SCENE_OBS_JOINT_MAP = {
    0: 'base__slide',
    1: 'base__drawer',
    2: 'base__button',
    3: 'base__switch',
}
# Light link names
LIGHT_LINKS = {'lightbulb': 'light_link', 'led': 'led_link'}
LIGHT_COLORS = {
    'lightbulb': {'on': [1, 1, 0, 1], 'off': [1, 1, 1, 1]},
    'led': {'on': [0, 1, 0, 1], 'off': [1, 1, 1, 1]},
}

FPS = 30.0


# ---------------------------------------------------------------------------
# PyBullet scene setup
# ---------------------------------------------------------------------------
class CalvinScene:
    """Minimal CALVIN scene for state restoration and rendering."""

    def __init__(self, asset_dir: str, width: int = 640, height: int = 640,
                 table_urdf: str = None):
        self.asset_dir = Path(asset_dir)
        self.width = width
        self.height = height
        self.table_urdf = table_urdf or DEFAULT_TABLE_URDF

        # Start PyBullet in DIRECT mode (headless)
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(str(self.asset_dir), physicsClientId=self.cid)

        # Load scene objects
        self.plane_id = p.loadURDF(
            SCENE_CFG['plane_urdf'],
            physicsClientId=self.cid,
        )
        self.table_id = p.loadURDF(
            self.table_urdf,
            globalScaling=SCENE_CFG['global_scaling'],
            useFixedBase=True,
            physicsClientId=self.cid,
        )
        self.robot_id = p.loadURDF(
            ROBOT_CFG['filename'],
            basePosition=ROBOT_CFG['base_position'],
            baseOrientation=p.getQuaternionFromEuler(ROBOT_CFG['base_orientation']),
            useFixedBase=True,
            physicsClientId=self.cid,
        )

        # Load blocks
        self.block_ids = {}
        for name, urdf in SCENE_CFG['block_urdfs'].items():
            self.block_ids[name] = p.loadURDF(
                urdf,
                globalScaling=SCENE_CFG['global_scaling'],
                physicsClientId=self.cid,
            )

        # Discover table joint indices
        self.table_joint_indices = {}
        for i in range(p.getNumJoints(self.table_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.table_id, i, physicsClientId=self.cid)
            name = info[1].decode('utf-8')
            if name in SCENE_OBS_JOINT_MAP.values():
                self.table_joint_indices[name] = i

        # Discover light link indices
        self.light_link_ids = {}
        for i in range(p.getNumJoints(self.table_id, physicsClientId=self.cid)):
            info = p.getJointInfo(self.table_id, i, physicsClientId=self.cid)
            link_name = info[12].decode('utf-8')
            for light_name, cfg_link_name in LIGHT_LINKS.items():
                if link_name == cfg_link_name:
                    self.light_link_ids[light_name] = i

        # Compute camera matrices for the target resolution
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=STATIC_CAM['look_from'],
            cameraTargetPosition=STATIC_CAM['look_at'],
            cameraUpVector=STATIC_CAM['up_vector'],
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=STATIC_CAM['fov'],
            aspect=width / height,
            nearVal=STATIC_CAM['nearval'],
            farVal=STATIC_CAM['farval'],
        )

        print(f"[CalvinScene] Loaded: table={self.table_id}, robot={self.robot_id}, "
              f"blocks={self.block_ids}")
        print(f"[CalvinScene] Table joints: {self.table_joint_indices}")
        print(f"[CalvinScene] Light links: {self.light_link_ids}")

    def restore_state(self, robot_obs: np.ndarray, scene_obs: np.ndarray):
        """Restore full scene state from a single frame's observations."""
        # --- Robot arm joints ---
        joint_positions = robot_obs[7:14]  # 7 arm joints
        for i, jid in enumerate(ROBOT_CFG['arm_joint_ids']):
            p.resetJointState(self.robot_id, jid, joint_positions[i],
                              physicsClientId=self.cid)

        # --- Robot gripper ---
        gripper_width = robot_obs[6]  # total opening width
        finger_pos = gripper_width / 2.0
        for jid in ROBOT_CFG['gripper_joint_ids']:
            p.resetJointState(self.robot_id, jid, finger_pos,
                              physicsClientId=self.cid)

        # --- Table joints (slide, drawer, button, switch) ---
        for obs_idx, joint_name in SCENE_OBS_JOINT_MAP.items():
            if joint_name in self.table_joint_indices:
                p.resetJointState(self.table_id, self.table_joint_indices[joint_name],
                                  scene_obs[obs_idx], physicsClientId=self.cid)

        # --- Lights (visual only — change link color) ---
        light_states = {'lightbulb': scene_obs[4], 'led': scene_obs[5]}
        for light_name, state in light_states.items():
            if light_name in self.light_link_ids:
                link_id = self.light_link_ids[light_name]
                if state >= 0.5:  # ON
                    color = LIGHT_COLORS[light_name]['on']
                else:
                    color = LIGHT_COLORS[light_name]['off']
                p.changeVisualShape(self.table_id, link_id,
                                    rgbaColor=color,
                                    physicsClientId=self.cid)

        # --- Movable objects (blocks) ---
        block_names = ['block_red', 'block_blue', 'block_pink']
        for i, name in enumerate(block_names):
            offset = 6 + i * 6
            pos = scene_obs[offset:offset + 3]
            euler = scene_obs[offset + 3:offset + 6]
            orn = p.getQuaternionFromEuler(euler.tolist())
            p.resetBasePositionAndOrientation(
                self.block_ids[name], pos.tolist(), orn,
                physicsClientId=self.cid,
            )

    def render(self):
        """Render static camera at target resolution. Returns (rgb, depth_meters)."""
        _, _, rgba, depth_buffer, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )

        # RGBA → RGB uint8
        rgb = np.reshape(rgba, (self.height, self.width, 4))[:, :, :3].astype(np.uint8)

        # Z-buffer → metric depth (meters)
        depth_buf = np.reshape(depth_buffer, (self.height, self.width))
        near = STATIC_CAM['nearval']
        far = STATIC_CAM['farval']
        depth_m = (far * near / (far - (far - near) * depth_buf)).astype(np.float32)

        return rgb, depth_m

    def close(self):
        p.disconnect(self.cid)


# ---------------------------------------------------------------------------
# Task filtering
# ---------------------------------------------------------------------------
def load_task_filter(filter_path: str):
    """Load kept task names from YAML filter file."""
    if filter_path is None:
        return None
    with open(filter_path) as f:
        cfg = yaml.safe_load(f)
    return set(cfg['keep_tasks'])


def get_kept_frame_ranges(split_dir: Path, keep_tasks: set, margin_frames: int):
    """Identify frame ranges to re-render based on task filter.

    Returns a set of absolute frame indices that should be re-rendered.
    If keep_tasks is None, returns None (render all frames).
    """
    ann_path = split_dir / 'lang_annotations' / 'auto_lang_ann.npy'
    if not ann_path.exists():
        return set()

    ann = np.load(str(ann_path), allow_pickle=True).item()
    tasks = ann['language'].get('task', None)
    frame_ranges = ann['info']['indx']

    if tasks is None:
        # No task labels — keep all annotated frames
        kept = set()
        for start, end in frame_ranges:
            for f in range(max(0, int(start) - margin_frames), int(end) + margin_frames + 1):
                kept.add(f)
        return kept

    kept = set()
    for task, (start, end) in zip(tasks, frame_ranges):
        if task in keep_tasks:
            for f in range(max(0, int(start) - margin_frames), int(end) + margin_frames + 1):
                kept.add(f)
    return kept


def filter_annotations(split_dir: Path, keep_tasks: set):
    """Return filtered annotation dict keeping only annotations for kept tasks."""
    ann_path = split_dir / 'lang_annotations' / 'auto_lang_ann.npy'
    if not ann_path.exists():
        return None

    ann = np.load(str(ann_path), allow_pickle=True).item()
    tasks = ann['language'].get('task', None)
    if tasks is None or keep_tasks is None:
        return ann

    # Filter
    texts = ann['language']['ann']
    frame_ranges = ann['info']['indx']

    keep_idx = [i for i, t in enumerate(tasks) if t in keep_tasks]

    n = len(texts)
    filtered = {
        'language': {
            'ann': [texts[i] for i in keep_idx],
            'task': [tasks[i] for i in keep_idx],
        },
        'info': {
            'indx': [frame_ranges[i] for i in keep_idx],
        },
    }
    # Preserve any other keys (handle ndarrays and mismatched lengths)
    for k in ann['language']:
        if k not in filtered['language']:
            v = ann['language'][k]
            if isinstance(v, np.ndarray) and len(v) == n:
                filtered['language'][k] = v[keep_idx]
            elif hasattr(v, '__len__') and len(v) == n:
                filtered['language'][k] = [v[i] for i in keep_idx]
    for k in ann['info']:
        if k not in filtered['info']:
            v = ann['info'][k]
            if isinstance(v, np.ndarray) and len(v) == n:
                filtered['info'][k] = v[keep_idx]
            elif hasattr(v, '__len__') and len(v) == n:
                filtered['info'][k] = [v[i] for i in keep_idx]
            # Skip keys with mismatched length (e.g. empty 'episodes')

    return filtered


# ---------------------------------------------------------------------------
# Main re-render loop
# ---------------------------------------------------------------------------
def _load_env_frame_ranges(src_split: Path):
    """Read scene_info.npy to get per-environment frame ranges.

    Returns dict: env_name → (start, end) covering that environment's frames.
    Falls back to a single 'calvin_scene_D' entry if scene_info is missing.
    """
    si_path = src_split / 'scene_info.npy'
    if si_path.exists():
        si = np.load(str(si_path), allow_pickle=True).item()
        return {env: (int(r[0]), int(r[1])) for env, r in si.items()}
    return {'calvin_scene_D': (0, 999_999_999)}


def rerender_split(asset_dir: str, width: int, height: int,
                   src_split: Path, dst_split: Path,
                   keep_tasks, margin_frames: int,
                   shard: int = None, num_shards: int = None):
    """Re-render all kept frames in a single split, handling multiple envs."""

    # Load episode boundaries
    ep_ids_path = src_split / 'ep_start_end_ids.npy'
    if not ep_ids_path.exists():
        print(f"  [SKIP] No ep_start_end_ids.npy in {src_split}")
        return

    ep_ids = np.load(str(ep_ids_path))

    # Determine which frames to render
    if keep_tasks is not None:
        kept_frames = get_kept_frame_ranges(src_split, keep_tasks, margin_frames)
        print(f"  Task filter: {len(kept_frames)} frames to render "
              f"(out of total episode range)")
    else:
        kept_frames = None

    # Create output directory
    dst_split.mkdir(parents=True, exist_ok=True)

    # Only shard 0 (or non-sharded runs) copies metadata to avoid race conditions
    if shard is None or shard == 0:
        # Copy metadata files
        for meta_file in ['ep_start_end_ids.npy', 'ep_lens.npy', 'scene_info.npy',
                          'statistics.yaml']:
            src_meta = src_split / meta_file
            if src_meta.exists():
                shutil.copy2(str(src_meta), str(dst_split / meta_file))

        # Copy .hydra config
        src_hydra = src_split / '.hydra'
        dst_hydra = dst_split / '.hydra'
        if src_hydra.exists() and not dst_hydra.exists():
            shutil.copytree(str(src_hydra), str(dst_hydra))

        # Write filtered annotations
        if keep_tasks is not None:
            filtered_ann = filter_annotations(src_split, keep_tasks)
            if filtered_ann is not None:
                ann_dir = dst_split / 'lang_annotations'
                ann_dir.mkdir(exist_ok=True)
                np.save(str(ann_dir / 'auto_lang_ann.npy'), filtered_ann)
        else:
            src_ann = src_split / 'lang_annotations'
            dst_ann = dst_split / 'lang_annotations'
            if src_ann.exists() and not dst_ann.exists():
                shutil.copytree(str(src_ann), str(dst_ann))

    # Collect all frame IDs across episodes
    all_frame_ids = []
    for start, end in ep_ids:
        for fid in range(int(start), int(end) + 1):
            if kept_frames is None or fid in kept_frames:
                all_frame_ids.append(fid)

    # Apply sharding
    if shard is not None and num_shards is not None:
        all_frame_ids = all_frame_ids[shard::num_shards]

    # Detect per-environment frame ranges
    env_ranges = _load_env_frame_ranges(src_split)
    envs_present = sorted(env_ranges.keys())
    print(f"  Environments: {envs_present}")

    # Build frame→env lookup
    def _get_env(fid):
        for env, (s, e) in env_ranges.items():
            if s <= fid <= e:
                return env
        return envs_present[0]

    # Group frames by environment for efficient scene reuse
    from collections import OrderedDict
    env_frames = OrderedDict()
    for fid in all_frame_ids:
        env = _get_env(fid)
        env_frames.setdefault(env, []).append(fid)

    total = len(all_frame_ids)
    shard_info = f" (shard {shard}/{num_shards})" if shard is not None else ""
    print(f"  Rendering {total} frames at {width}x{height}{shard_info}")

    rendered = 0
    for env_name, fids in env_frames.items():
        table_urdf = ENV_TABLE_URDFS.get(env_name, DEFAULT_TABLE_URDF)
        print(f"    [{env_name}] {len(fids)} frames, table: {table_urdf}")
        scene = CalvinScene(asset_dir, width=width, height=height,
                            table_urdf=table_urdf)
        try:
            for fid in tqdm(fids, desc=f"  {src_split.name}/{env_name}"):
                src_npz = src_split / f'episode_{fid:07d}.npz'
                dst_npz = dst_split / f'episode_{fid:07d}.npz'

                if dst_npz.exists():
                    continue

                if not src_npz.exists():
                    continue

                data = np.load(str(src_npz))
                scene.restore_state(data['robot_obs'], data['scene_obs'])
                rgb, depth_m = scene.render()

                np.savez_compressed(
                    str(dst_npz),
                    rgb_static=rgb,
                    depth_static=depth_m,
                    robot_obs=data['robot_obs'],
                    actions=data['actions'],
                    rel_actions=data['rel_actions'],
                    scene_obs=data['scene_obs'],
                )
                data.close()
                rendered += 1
        finally:
            scene.close()

    print(f"  Done: {rendered} new frames rendered{shard_info}")


def launch_workers(num_workers: int, args):
    """Launch num_workers parallel shard processes."""
    procs = []
    base_cmd = [
        sys.executable, __file__,
        '--src', str(args.src),
        '--dst', str(args.dst),
        '--assets', args.assets,
        '--splits', *args.splits,
        '--width', str(args.width),
        '--height', str(args.height),
        '--margin-seconds', str(args.margin_seconds),
    ]
    if args.task_filter:
        base_cmd += ['--task-filter', args.task_filter]

    print(f"[Launcher] Starting {num_workers} worker processes...")
    for i in range(num_workers):
        cmd = base_cmd + ['--shard', str(i), '--num-shards', str(num_workers)]
        log_path = Path(args.dst) / f'worker_{i}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(str(log_path), 'w')
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_file))
        print(f"  Worker {i}/{num_workers} started (pid={proc.pid}, log={log_path})")

    # Wait for all workers
    print(f"[Launcher] All {num_workers} workers running. Waiting for completion...")
    failed = []
    for i, proc, log_file in procs:
        rc = proc.wait()
        log_file.close()
        if rc != 0:
            failed.append(i)
            print(f"  Worker {i} FAILED (rc={rc})")
        else:
            print(f"  Worker {i} done")

    if failed:
        print(f"\n[Launcher] {len(failed)} workers failed: {failed}")
        print("Check logs at:", Path(args.dst) / 'worker_*.log')
        sys.exit(1)
    else:
        # Clean up log files on success
        for i in range(num_workers):
            log_path = Path(args.dst) / f'worker_{i}.log'
            log_path.unlink(missing_ok=True)
        print(f"\n[Launcher] All {num_workers} workers completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Re-render CALVIN at higher resolution')
    parser.add_argument('--src', required=True, help='Source dataset dir (e.g. Calvin/task_D_D)')
    parser.add_argument('--dst', required=True, help='Output dir (e.g. Calvin/calvin_D_D_640)')
    parser.add_argument('--assets', required=True,
                        help='Path to calvin_env_assets/data with URDFs')
    parser.add_argument('--splits', nargs='+', default=['training', 'validation'])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--margin-seconds', type=float, default=2.0,
                        help='Context margin before/after task intervals (seconds)')
    parser.add_argument('--task-filter', type=str, default=None,
                        help='Path to task filter YAML (e.g. config/dataset/calvin_task_filter.yaml)')
    # Multi-process options
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Launch N parallel worker processes (convenience wrapper)')
    parser.add_argument('--shard', type=int, default=None,
                        help='This worker processes frame_ids[shard::num_shards]')
    parser.add_argument('--num-shards', type=int, default=None,
                        help='Total number of shards (used with --shard)')
    args = parser.parse_args()

    # Launcher mode: spawn N workers and wait
    if args.num_workers is not None:
        launch_workers(args.num_workers, args)
        return

    src = Path(args.src)
    dst = Path(args.dst)
    margin_frames = int(args.margin_seconds * FPS)

    shard_info = ""
    if args.shard is not None:
        shard_info = f" (shard {args.shard}/{args.num_shards})"

    # Load task filter
    keep_tasks = load_task_filter(args.task_filter)
    if keep_tasks is not None:
        print(f"[TaskFilter]{shard_info} Keeping {len(keep_tasks)} tasks")
    else:
        print(f"[TaskFilter]{shard_info} No filter — rendering all frames")

    for split in args.splits:
        src_split = src / split
        dst_split = dst / split
        if not src_split.exists():
            print(f"[SKIP] Split dir not found: {src_split}")
            continue
        print(f"\n=== Processing {split}{shard_info} ===")
        rerender_split(args.assets, args.width, args.height,
                       src_split, dst_split, keep_tasks, margin_frames,
                       shard=args.shard, num_shards=args.num_shards)

    print(f"\nDone!{shard_info}")


if __name__ == '__main__':
    main()
