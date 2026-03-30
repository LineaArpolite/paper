import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from robosuite import load_controller_config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

TARGET_OBJECT_NAME = "milk_1"
TARGET_CONTAINER_NAME = "basket_1"
OBSTACLE_NAME = "obstacle_1"


@dataclass
class RolloutResult:
    strategy_label: str
    success: bool
    raw_success: bool
    collision: bool
    collision_steps: List[int]
    branch_step: int
    grasp_step: int
    place_step: int
    init_state: np.ndarray
    actions: np.ndarray
    states: np.ndarray
    agentview_rgb: np.ndarray
    eye_in_hand_rgb: np.ndarray
    ee_pos: np.ndarray
    object_pos: np.ndarray
    obstacle_pos: np.ndarray
    obstacle_bbox_xy: np.ndarray


class Recorder:
    def __init__(self) -> None:
        self.actions: List[np.ndarray] = []
        self.states: List[np.ndarray] = []
        self.agentview: List[np.ndarray] = []
        self.eye: List[np.ndarray] = []
        self.ee: List[np.ndarray] = []
        self.obj: List[np.ndarray] = []
        self.obstacle: List[np.ndarray] = []

    def add(self, action: np.ndarray, env, obs: Dict, obstacle_pos: np.ndarray) -> None:
        self.actions.append(action.astype(np.float32).copy())
        self.states.append(env.sim.get_state().flatten().copy())
        self.agentview.append(obs["agentview_image"].copy())
        self.eye.append(obs["robot0_eye_in_hand_image"].copy())
        self.ee.append(obs["robot0_eef_pos"].copy())
        self.obj.append(obs[f"{TARGET_OBJECT_NAME}_pos"].copy())
        self.obstacle.append(obstacle_pos.copy())

    def to_arrays(self) -> Tuple[np.ndarray, ...]:
        return (
            np.asarray(self.actions, dtype=np.float32),
            np.asarray(self.states, dtype=np.float64),
            np.asarray(self.agentview, dtype=np.uint8),
            np.asarray(self.eye, dtype=np.uint8),
            np.asarray(self.ee, dtype=np.float32),
            np.asarray(self.obj, dtype=np.float32),
            np.asarray(self.obstacle, dtype=np.float32),
        )


def _build_env(bddl_file: Path):
    problem_info = BDDLUtils.get_problem_info(str(bddl_file))
    controller = load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_info["problem_name"]](
        bddl_file_name=str(bddl_file),
        robots=["Panda"],
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=128,
        camera_widths=128,
        ignore_done=True,
        reward_shaping=True,
        control_freq=20,
    )
    return env, problem_info


def _reset_to_snapshot(env, snapshot: np.ndarray) -> Dict:
    # Reset first so controllers / env internal caches are reinitialized.
    env.reset()
    env.sim.set_state_from_flattened(snapshot)
    env.sim.forward()
    return env._get_observations()


def _collision_id_sets(env) -> Tuple[set, set, set]:
    geom_name_to_id = {
        env.sim.model.geom_id2name(i): i
        for i in range(env.sim.model.ngeom)
        if env.sim.model.geom_id2name(i)
    }
    obstacle_ids = {gid for name, gid in geom_name_to_id.items() if OBSTACLE_NAME in name}
    robot_ids = {
        gid
        for name, gid in geom_name_to_id.items()
        if name.startswith("robot0_link") or name.startswith("gripper0_")
    }
    object_ids = {
        gid for name, gid in geom_name_to_id.items() if name.startswith(f"{TARGET_OBJECT_NAME}_")
    }
    return obstacle_ids, robot_ids, object_ids


def _has_collision_with_obstacle(
    env,
    obstacle_ids: set,
    robot_ids: set,
    object_ids: set,
) -> bool:
    for ci in range(env.sim.data.ncon):
        contact = env.sim.data.contact[ci]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if (g1 in obstacle_ids and (g2 in robot_ids or g2 in object_ids)) or (
            g2 in obstacle_ids and (g1 in robot_ids or g1 in object_ids)
        ):
            return True
    return False


def _obstacle_position(env) -> np.ndarray:
    return env.sim.data.body_xpos[env.sim.model.body_name2id(f"{OBSTACLE_NAME}_main")].copy()


def _obstacle_bbox_xy(env) -> np.ndarray:
    xs: List[float] = []
    ys: List[float] = []
    for i in range(env.sim.model.ngeom):
        name = env.sim.model.geom_id2name(i)
        if not name or OBSTACLE_NAME not in name:
            continue
        center = env.sim.data.geom_xpos[i]
        size = env.sim.model.geom_size[i]
        xs.extend([float(center[0] - size[0]), float(center[0] + size[0])])
        ys.extend([float(center[1] - size[1]), float(center[1] + size[1])])
    return np.asarray([min(xs), max(xs), min(ys), max(ys)], dtype=np.float32)


def _layout_feature(obs: Dict, obstacle_pos: np.ndarray) -> np.ndarray:
    obj_xy = obs[f"{TARGET_OBJECT_NAME}_pos"][:2]
    tgt_xy = obs[f"{TARGET_CONTAINER_NAME}_pos"][:2]
    obs_xy = obstacle_pos[:2]
    return np.concatenate([obj_xy, tgt_xy, obs_xy]).astype(np.float32)


def _plan_shared_prefix(env, snapshot: np.ndarray) -> Tuple[np.ndarray, bool]:
    obs = _reset_to_snapshot(env, snapshot)
    actions: List[np.ndarray] = []

    target_obj = [o for o in env.objects if o.name == TARGET_OBJECT_NAME][0]
    carry_z = 0.235

    def step_plan(action: np.ndarray) -> None:
        nonlocal obs
        obs, _, _, _ = env.step(action)
        actions.append(action.astype(np.float32).copy())

    def goto(
        target_pos: np.ndarray,
        gripper_cmd: float,
        tol: float = 0.012,
        max_steps: int = 140,
        gain: float = 10.0,
    ) -> None:
        prev_dist = 1e9
        near_stable_steps = 0
        for _ in range(max_steps):
            delta = target_pos - obs["robot0_eef_pos"]
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                break
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[:3] = np.clip(gain * delta, -1.0, 1.0)
            action[-1] = gripper_cmd
            step_plan(action)

            if abs(prev_dist - dist) < 3e-4 and dist < (tol * 1.8):
                near_stable_steps += 1
                if near_stable_steps >= 4:
                    break
            else:
                near_stable_steps = 0
            prev_dist = dist

    object_pos = obs[f"{TARGET_OBJECT_NAME}_pos"].copy()
    obstacle_pos = _obstacle_position(env)

    goto(object_pos + np.array([0.0, 0.0, 0.13]), -1.0)
    goto(object_pos + np.array([0.0, 0.0, 0.02]), -1.0, tol=0.008, max_steps=120, gain=8.0)

    grasped = False
    for _ in range(24):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = 1.0
        step_plan(action)
        if env._check_grasp(env.robots[0].gripper, target_obj.contact_geoms):
            grasped = True
            for _ in range(4):
                action = np.zeros(env.action_dim, dtype=np.float32)
                action[-1] = 1.0
                step_plan(action)
            break

    if not grasped:
        return np.asarray(actions, dtype=np.float32), False

    current_obj_pos = obs[f"{TARGET_OBJECT_NAME}_pos"].copy()
    goto(np.array([current_obj_pos[0], current_obj_pos[1], carry_z]), 1.0, tol=0.011, max_steps=140, gain=9.0)
    fork = np.array([obstacle_pos[0] - 0.02, obstacle_pos[1] - 0.14, carry_z], dtype=np.float32)
    goto(fork, 1.0, tol=0.011, max_steps=150, gain=9.0)

    return np.asarray(actions, dtype=np.float32), True


def _rollout_strategy(
    env,
    snapshot: np.ndarray,
    strategy_label: str,
    prefix_actions: np.ndarray,
) -> RolloutResult:
    assert strategy_label in ["A", "B"]

    obstacle_ids, robot_ids, object_ids = _collision_id_sets(env)
    obstacle_bbox_xy = _obstacle_bbox_xy(env)
    obs = _reset_to_snapshot(env, snapshot)

    recorder = Recorder()
    collision_steps: List[int] = []
    collision_flag = False

    target_obj = [o for o in env.objects if o.name == TARGET_OBJECT_NAME][0]

    def step_once(action: np.ndarray) -> None:
        nonlocal obs, collision_flag
        obs, _, _, _ = env.step(action)
        obstacle_pos = _obstacle_position(env)
        recorder.add(action, env, obs, obstacle_pos)
        if _has_collision_with_obstacle(env, obstacle_ids, robot_ids, object_ids):
            collision_flag = True
            collision_steps.append(len(recorder.actions) - 1)

    def goto(
        target_pos: np.ndarray,
        gripper_cmd: float,
        tol: float = 0.011,
        max_steps: int = 140,
        gain: float = 9.0,
    ) -> None:
        prev_dist = 1e9
        near_stable_steps = 0
        for _ in range(max_steps):
            delta = target_pos - obs["robot0_eef_pos"]
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                break
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[:3] = np.clip(gain * delta, -1.0, 1.0)
            action[-1] = gripper_cmd
            step_once(action)

            if abs(prev_dist - dist) < 3e-4 and dist < (tol * 1.8):
                near_stable_steps += 1
                if near_stable_steps >= 4:
                    break
            else:
                near_stable_steps = 0
            prev_dist = dist

    carry_z = float(obs["robot0_eef_pos"][2])
    side_x = 0.115
    side_y_pre = -0.02 if strategy_label == "A" else -0.10
    side_y_post = 0.155

    target_pos = obs[f"{TARGET_CONTAINER_NAME}_pos"].copy()
    obstacle_pos = _obstacle_position(env)

    grasp_step = -1
    # Replay exactly the same pre-branch actions for A and B to minimize visible divergence.
    for action in prefix_actions:
        step_once(action)
        if grasp_step < 0 and env._check_grasp(env.robots[0].gripper, target_obj.contact_geoms):
            grasp_step = len(recorder.actions) - 1

    branch_step = max(0, len(recorder.actions) - 1)

    # Divergence: left vs right route around the obstacle.
    side_sign = -1.0 if strategy_label == "A" else 1.0
    waypoints = [
        np.array([obstacle_pos[0] + side_sign * side_x, obstacle_pos[1] + side_y_pre, carry_z]),
        np.array([obstacle_pos[0] + side_sign * side_x, obstacle_pos[1] + side_y_post, carry_z]),
    ]
    for waypoint in waypoints:
        goto(waypoint, 1.0)

    # Raise before final horizontal approach to reduce rim rubbing on basket.
    transit_z = max(carry_z + 0.05, float(target_pos[2] + 0.23))
    current_ee = obs["robot0_eef_pos"].copy()
    goto(np.array([current_ee[0], current_ee[1], transit_z]), 1.0, tol=0.010, max_steps=120, gain=8.0)
    goto(np.array([target_pos[0], target_pos[1], transit_z]), 1.0, tol=0.010, max_steps=130, gain=8.0)
    # Use lower, gentler insertion to reduce object-basket rim hits near placement.
    goto(np.array([target_pos[0], target_pos[1], target_pos[2] + 0.055]), 1.0, tol=0.007, max_steps=180, gain=7.0)
    place_step = len(recorder.actions) - 1

    for i in range(12):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = -1.0
        if i >= 6:
            action[2] = 0.18
        step_once(action)

    goto(np.array([target_pos[0], target_pos[1], carry_z + 0.015]), -1.0, tol=0.012, max_steps=90, gain=8.0)

    raw_success = bool(env._check_success())
    success = raw_success and (not collision_flag)

    (
        actions,
        states,
        agentview_rgb,
        eye_in_hand_rgb,
        ee_pos,
        object_pos_seq,
        obstacle_pos_seq,
    ) = recorder.to_arrays()

    return RolloutResult(
        strategy_label=strategy_label,
        success=success,
        raw_success=raw_success,
        collision=collision_flag,
        collision_steps=collision_steps,
        branch_step=int(branch_step),
        grasp_step=int(grasp_step),
        place_step=int(place_step),
        init_state=snapshot.copy(),
        actions=actions,
        states=states,
        agentview_rgb=agentview_rgb,
        eye_in_hand_rgb=eye_in_hand_rgb,
        ee_pos=ee_pos,
        object_pos=object_pos_seq,
        obstacle_pos=obstacle_pos_seq,
        obstacle_bbox_xy=obstacle_bbox_xy,
    )


def _save_rollout_hdf5(
    out_path: Path,
    result: RolloutResult,
    group_id: int,
    random_seed: int,
    bddl_file: Path,
    instruction: str,
    init_object_pos: np.ndarray,
    init_target_pos: np.ndarray,
    init_obstacle_pos: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        grp = f.create_group("data")
        ep = grp.create_group("demo_1")
        obs_grp = ep.create_group("obs")

        obs_grp.create_dataset("agentview_rgb", data=result.agentview_rgb, compression="gzip", compression_opts=2)
        obs_grp.create_dataset("eye_in_hand_rgb", data=result.eye_in_hand_rgb, compression="gzip", compression_opts=2)
        obs_grp.create_dataset("ee_pos", data=result.ee_pos)
        obs_grp.create_dataset("object_pos", data=result.object_pos)
        obs_grp.create_dataset("obstacle_pos", data=result.obstacle_pos)

        ep.create_dataset("actions", data=result.actions)
        ep.create_dataset("states", data=result.states)
        ep.create_dataset("init_state", data=result.init_state)

        dones = np.zeros(len(result.actions), dtype=np.uint8)
        rewards = np.zeros(len(result.actions), dtype=np.uint8)
        if len(dones) > 0:
            dones[-1] = 1
            rewards[-1] = 1 if result.raw_success else 0
        ep.create_dataset("dones", data=dones)
        ep.create_dataset("rewards", data=rewards)

        ep.attrs["num_samples"] = int(len(result.actions))
        ep.attrs["success"] = int(result.success)
        ep.attrs["raw_success"] = int(result.raw_success)
        ep.attrs["collision"] = int(result.collision)
        ep.attrs["branch_step"] = int(result.branch_step)
        ep.attrs["grasp_step"] = int(result.grasp_step)
        ep.attrs["place_step"] = int(result.place_step)
        ep.attrs["collision_steps_json"] = json.dumps(result.collision_steps)
        ep.attrs["strategy_label"] = result.strategy_label
        ep.attrs["group_id"] = int(group_id)
        ep.attrs["random_seed"] = int(random_seed)
        ep.attrs["target_object_name"] = TARGET_OBJECT_NAME
        ep.attrs["target_container_name"] = TARGET_CONTAINER_NAME
        ep.attrs["obstacle_name"] = OBSTACLE_NAME
        ep.attrs["obstacle_bbox_xy_json"] = json.dumps(result.obstacle_bbox_xy.tolist())
        ep.attrs["init_object_pos_xyz_json"] = json.dumps(init_object_pos.tolist())
        ep.attrs["init_target_pos_xyz_json"] = json.dumps(init_target_pos.tolist())
        ep.attrs["init_obstacle_pos_xyz_json"] = json.dumps(init_obstacle_pos.tolist())

        grp.attrs["instruction"] = instruction
        grp.attrs["group_id"] = int(group_id)
        grp.attrs["strategy_label"] = result.strategy_label
        grp.attrs["random_seed"] = int(random_seed)
        grp.attrs["bddl_file_name"] = str(bddl_file)
        grp.attrs["env_name"] = "libero_floor_manipulation"
        grp.attrs["env_args"] = json.dumps(
            {
                "problem_name": "libero_floor_manipulation",
                "bddl_file": str(bddl_file),
                "controller": "OSC_POSE",
                "camera_names": ["agentview", "robot0_eye_in_hand"],
            }
        )


def _save_metadata_json(
    out_path: Path,
    result: RolloutResult,
    group_id: int,
    random_seed: int,
    bddl_file: Path,
    instruction: str,
    init_object_pos: np.ndarray,
    init_target_pos: np.ndarray,
    init_obstacle_pos: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "group_id": group_id,
        "trajectory_id": f"group_{group_id:02d}_{result.strategy_label}",
        "strategy_label": result.strategy_label,
        "instruction": instruction,
        "random_seed": random_seed,
        "environment_config": {
            "bddl_file": str(bddl_file),
            "target_object_name": TARGET_OBJECT_NAME,
            "target_container_name": TARGET_CONTAINER_NAME,
            "obstacle_name": OBSTACLE_NAME,
            "obstacle_bbox_xy": result.obstacle_bbox_xy.tolist(),
            "init_object_pos_xyz": init_object_pos.tolist(),
            "init_target_pos_xyz": init_target_pos.tolist(),
            "init_obstacle_pos_xyz": init_obstacle_pos.tolist(),
        },
        "trajectory_length": int(len(result.actions)),
        "key_steps": {
            "branch_step": int(result.branch_step),
            "grasp_step": int(result.grasp_step),
            "place_step": int(result.place_step),
        },
        "status": {
            "success": bool(result.success),
            "raw_success": bool(result.raw_success),
            "collision": bool(result.collision),
            "collision_steps": [int(x) for x in result.collision_steps],
        },
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_seeds(text: str) -> List[int]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bddl-file",
        type=Path,
        default=Path("data_collection_outputs/bddl/dual_strategy_milk_obstacle.bddl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data_collection_outputs"),
    )
    parser.add_argument("--num-groups", type=int, default=5)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1,
        help="Used when --seed-candidates is empty.",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=240,
        help="Used when --seed-candidates is empty.",
    )
    parser.add_argument(
        "--seed-candidates",
        type=str,
        default="",
        help="Comma-separated seeds. Leave empty to use [seed-start, seed-end].",
    )
    parser.add_argument(
        "--min-layout-distance",
        type=float,
        default=0.022,
        help="Minimum L2 distance between accepted initial layouts (obj_xy, target_xy, obstacle_xy).",
    )
    args = parser.parse_args()

    if not args.bddl_file.exists():
        raise FileNotFoundError(f"Missing BDDL file: {args.bddl_file}")

    if args.seed_candidates.strip():
        seeds = _parse_seeds(args.seed_candidates)
    else:
        seeds = list(range(args.seed_start, args.seed_end + 1))

    trajectories_dir = args.output_root / "trajectories"
    metadata_dir = args.output_root / "metadata"
    analysis_dir = args.output_root / "analysis"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    problem_info = BDDLUtils.get_problem_info(str(args.bddl_file))
    instruction = problem_info["language_instruction"]

    accepted: List[Dict] = []
    accepted_layouts: List[np.ndarray] = []

    for seed in seeds:
        if len(accepted) >= args.num_groups:
            break

        env, _ = _build_env(args.bddl_file)
        env.seed(seed)
        init_obs = env.reset()
        snapshot = env.sim.get_state().flatten().copy()
        init_object_pos = init_obs[f"{TARGET_OBJECT_NAME}_pos"].copy()
        init_target_pos = init_obs[f"{TARGET_CONTAINER_NAME}_pos"].copy()
        init_obstacle_pos = _obstacle_position(env).copy()
        layout_feat = _layout_feature(init_obs, init_obstacle_pos)

        min_layout_dist = float("inf")
        if accepted_layouts:
            min_layout_dist = min(float(np.linalg.norm(layout_feat - prev)) for prev in accepted_layouts)
            if min_layout_dist < args.min_layout_distance:
                print(
                    f"[skip-diversity] seed={seed} min_layout_dist={min_layout_dist:.4f} "
                    f"< threshold={args.min_layout_distance:.4f}"
                )
                env.close()
                continue

        prefix_actions, prefix_ok = _plan_shared_prefix(env, snapshot)
        if not prefix_ok:
            print(f"[skip-prefix] seed={seed} failed to build stable shared prefix")
            env.close()
            continue

        result_a = _rollout_strategy(env, snapshot, "A", prefix_actions)
        result_b = _rollout_strategy(env, snapshot, "B", prefix_actions)
        env.close()

        if not (result_a.success and result_b.success):
            print(
                f"[skip] seed={seed} "
                f"A(success={result_a.success}, collision={result_a.collision}) "
                f"B(success={result_b.success}, collision={result_b.collision})"
            )
            continue

        group_id = len(accepted) + 1
        a_h5 = trajectories_dir / f"group_{group_id:02d}_A.hdf5"
        b_h5 = trajectories_dir / f"group_{group_id:02d}_B.hdf5"
        a_json = metadata_dir / f"group_{group_id:02d}_A.json"
        b_json = metadata_dir / f"group_{group_id:02d}_B.json"

        _save_rollout_hdf5(
            a_h5,
            result_a,
            group_id,
            seed,
            args.bddl_file,
            instruction,
            init_object_pos,
            init_target_pos,
            init_obstacle_pos,
        )
        _save_rollout_hdf5(
            b_h5,
            result_b,
            group_id,
            seed,
            args.bddl_file,
            instruction,
            init_object_pos,
            init_target_pos,
            init_obstacle_pos,
        )
        _save_metadata_json(
            a_json,
            result_a,
            group_id,
            seed,
            args.bddl_file,
            instruction,
            init_object_pos,
            init_target_pos,
            init_obstacle_pos,
        )
        _save_metadata_json(
            b_json,
            result_b,
            group_id,
            seed,
            args.bddl_file,
            instruction,
            init_object_pos,
            init_target_pos,
            init_obstacle_pos,
        )
        accepted_layouts.append(layout_feat.copy())

        accepted.append(
            {
                "group_id": group_id,
                "seed": seed,
                "min_layout_dist_to_prev": None if min_layout_dist == float("inf") else float(min_layout_dist),
                "init_layout": {
                    "object_xyz": init_object_pos.tolist(),
                    "target_xyz": init_target_pos.tolist(),
                    "obstacle_xyz": init_obstacle_pos.tolist(),
                },
                "A": {
                    "path": str(a_h5),
                    "num_steps": int(len(result_a.actions)),
                    "branch_step": int(result_a.branch_step),
                    "collision": bool(result_a.collision),
                },
                "B": {
                    "path": str(b_h5),
                    "num_steps": int(len(result_b.actions)),
                    "branch_step": int(result_b.branch_step),
                    "collision": bool(result_b.collision),
                },
            }
        )

        print(f"[ok] group={group_id:02d} seed={seed} saved A/B trajectories")

    if len(accepted) < args.num_groups:
        raise RuntimeError(
            f"Collected {len(accepted)} valid groups, less than requested {args.num_groups}. "
            f"Try expanding seed range or lowering --min-layout-distance."
        )

    summary_path = analysis_dir / "collection_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "num_groups": len(accepted),
                "instruction": instruction,
                "bddl_file": str(args.bddl_file),
                "groups": accepted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved collection summary: {summary_path}")


if __name__ == "__main__":
    main()
