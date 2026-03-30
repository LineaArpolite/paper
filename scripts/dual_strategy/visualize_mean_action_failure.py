import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import cv2
import h5py
import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.dual_strategy.collect_dual_strategy_dataset import _build_env


def _load_pair(a_path: Path, b_path: Path) -> Tuple[Dict, Dict]:
    def _load(path: Path) -> Dict:
        with h5py.File(path, "r") as f:
            ep = f["data/demo_1"]
            obs = ep["obs"]
            return {
                "path": str(path),
                "actions": ep["actions"][()],
                "imgs": obs["agentview_rgb"][()],
                "init_state": ep["init_state"][()],
                "success": int(ep.attrs.get("success", 0)),
                "collision": int(ep.attrs.get("collision", 0)),
                "bddl_file": f["data"].attrs["bddl_file_name"],
                "seed": int(ep.attrs.get("random_seed", 0)),
            }

    return _load(a_path), _load(b_path)


def _collision_sets(env):
    geom_name_to_id = {
        env.sim.model.geom_id2name(i): i
        for i in range(env.sim.model.ngeom)
        if env.sim.model.geom_id2name(i)
    }
    obstacle = {gid for name, gid in geom_name_to_id.items() if "obstacle_1" in name}
    robot = {
        gid
        for name, gid in geom_name_to_id.items()
        if name.startswith("robot0_link") or name.startswith("gripper0_")
    }
    target = {gid for name, gid in geom_name_to_id.items() if name.startswith("milk_1_")}
    return obstacle, robot, target


def _has_collision(env, obstacle: set, robot: set, target: set) -> bool:
    for ci in range(env.sim.data.ncon):
        contact = env.sim.data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in obstacle and (g2 in robot or g2 in target)) or (
            g2 in obstacle and (g1 in robot or g1 in target)
        ):
            return True
    return False


def _save_mp4(frames, out_path: Path, fps: int) -> None:
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0], dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected RGB frames with shape (H, W, 3)")
    h, w = first.shape[:2]
    if h % 2 != 0:
        h -= 1
    if w % 2 != 0:
        w -= 1

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {out_path}")

    for frame in frames:
        rgb = np.asarray(frame, dtype=np.uint8)
        rgb = rgb[:h, :w, :3]
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data_collection_outputs")
    parser.add_argument("--pairs", default="1,2,3,4,5")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--output-dir",
        default="data_collection_outputs/visualizations/collision_debug",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for token in args.pairs.split(","):
        token = token.strip()
        if not token:
            continue
        gid = int(token)

        a_path = root / "trajectories" / f"group_{gid:02d}_A.hdf5"
        b_path = root / "trajectories" / f"group_{gid:02d}_B.hdf5"
        if not a_path.exists() or not b_path.exists():
            print(f"[warn] skip missing pair {gid:02d}")
            continue

        a, b = _load_pair(a_path, b_path)
        n = min(len(a["actions"]), len(b["actions"]))
        mean_actions = (a["actions"][:n] + b["actions"][:n]) / 2.0

        env, _ = _build_env(Path(a["bddl_file"]))
        env.seed(a["seed"])
        env.reset()
        env.sim.set_state_from_flattened(a["init_state"])
        env.sim.forward()
        obs = env._get_observations()

        obstacle_ids, robot_ids, target_ids = _collision_sets(env)

        mean_frames = []
        collision_steps = []
        for i, action in enumerate(mean_actions):
            obs, _, _, _ = env.step(action)
            mean_frames.append(obs["agentview_image"].copy())
            if _has_collision(env, obstacle_ids, robot_ids, target_ids):
                collision_steps.append(i)

        mean_success = bool(env._check_success())
        mean_collision = len(collision_steps) > 0
        env.close()

        # Triplet MP4: A | B | mean
        m = min(len(a["imgs"]), len(b["imgs"]), len(mean_frames))
        stride = max(1, m // 450)
        triplet_frames = []
        for i in range(0, m, stride):
            triplet_frames.append(
                np.concatenate(
                    [
                        np.flipud(a["imgs"][i]),
                        np.flipud(b["imgs"][i]),
                        np.flipud(mean_frames[i]),
                    ],
                    axis=1,
                )
            )

        triplet_path = out_dir / f"group_{gid:02d}_mean_action_triplet.mp4"
        _save_mp4(triplet_frames, triplet_path, args.fps)

        payload = {
            "group_id": gid,
            "A_success": int(a["success"]),
            "B_success": int(b["success"]),
            "mean_success": int(mean_success),
            "mean_collision": int(mean_collision),
            "mean_collision_steps": collision_steps,
            "num_steps_mean": int(n),
            "triplet_mp4": str(triplet_path),
        }
        json_path = out_dir / f"group_{gid:02d}_mean_action_failure.json"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(
            f"[group {gid:02d}] A={a['success']} B={b['success']} "
            f"mean_success={mean_success} mean_collision={mean_collision}"
        )


if __name__ == "__main__":
    main()
