import argparse
import json
import os
import re
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
TARGET_CONTAINER_NAME = os.environ.get("TARGET_CONTAINER_NAME", "")
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


def _infer_target_container_name(bddl_file: Path) -> str:
    text = bddl_file.read_text(encoding="utf-8")
    m = re.search(r"\(:obj_of_interest\s+([^\s()]+)\s+([^\s()]+)\)", text, flags=re.MULTILINE)
    if m:
        return m.group(2)
    m = re.search(r"\(contain_region\s+\(:target\s+([^\s()]+)\)\)", text, flags=re.MULTILINE)
    if m:
        return m.group(1)
    raise RuntimeError(f"Could not infer target container name from BDDL: {bddl_file}")


def _rollout_strategy(env, snapshot: np.ndarray, strategy_label: str, args: argparse.Namespace) -> RolloutResult:
    assert strategy_label in ["A", "B"]

    obstacle_ids, robot_ids, object_ids = _collision_id_sets(env)
    obstacle_bbox_xy = _obstacle_bbox_xy(env)
    obs = _reset_to_snapshot(env, snapshot)

    recorder = Recorder()
    collision_steps: List[int] = []
    collision_flag = False
    last_xyz_cmd = np.zeros(3, dtype=np.float32)

    target_obj = [o for o in env.objects if o.name == TARGET_OBJECT_NAME][0]

    def step_once(action: np.ndarray) -> None:
        nonlocal obs, collision_flag, last_xyz_cmd
        obs, _, _, _ = env.step(action)
        last_xyz_cmd = action[:3].astype(np.float32).copy()
        obstacle_pos = _obstacle_position(env)
        recorder.add(action, env, obs, obstacle_pos)
        if _has_collision_with_obstacle(env, obstacle_ids, robot_ids, object_ids):
            collision_flag = True
            collision_steps.append(len(recorder.actions) - 1)

    def goto(
        target_pos: np.ndarray,
        gripper_cmd: float,
        tol: float = -1.0,
        max_steps: int = -1,
        max_speed: float = -1.0,
        max_axis_speed: float = -1.0,
        max_accel: float = -1.0,
    ) -> None:
        if tol <= 0.0:
            tol = float(args.goto_tol)
        if max_steps <= 0:
            max_steps = int(args.goto_max_steps)
        if max_speed <= 0.0:
            max_speed = float(args.goto_max_speed)
        if max_axis_speed <= 0.0:
            max_axis_speed = float(args.goto_max_axis_speed)
        if max_accel <= 0.0:
            max_accel = float(args.goto_max_accel)
        prev_dist = 1e9
        near_stable_steps = 0
        for _ in range(max_steps):
            delta = target_pos - obs["robot0_eef_pos"]
            dist = float(np.linalg.norm(delta))
            if dist < tol:
                break
            action = np.zeros(env.action_dim, dtype=np.float32)
            cmd = np.clip(float(args.goto_gain) * delta, -max_axis_speed, max_axis_speed)
            cmd_norm = float(np.linalg.norm(cmd))
            if cmd_norm > max_speed and cmd_norm > 1e-8:
                cmd = cmd * (max_speed / cmd_norm)
            cmd_delta = np.clip(cmd - last_xyz_cmd, -max_accel, max_accel)
            action[:3] = last_xyz_cmd + cmd_delta
            action[-1] = gripper_cmd
            step_once(action)
            if abs(prev_dist - dist) < float(args.goto_stable_delta) and dist < (tol * float(args.goto_stable_factor)):
                near_stable_steps += 1
                if near_stable_steps >= int(args.goto_stable_steps):
                    break
            else:
                near_stable_steps = 0
            prev_dist = dist

    def hold(gripper_cmd: float, steps: int) -> None:
        for _ in range(steps):
            action = np.zeros(env.action_dim, dtype=np.float32)
            action[-1] = gripper_cmd
            step_once(action)

    carry_z = float(args.carry_z)
    side_x = float(args.side_x)
    side_y_pre = float(args.side_y_pre_a) if strategy_label == "A" else float(args.side_y_pre_b)
    side_y_post = float(args.side_y_post)

    object_pos = obs[f"{TARGET_OBJECT_NAME}_pos"].copy()
    target_pos = obs[f"{TARGET_CONTAINER_NAME}_pos"].copy()
    obstacle_pos = _obstacle_position(env)

    # Shared pre-branch prefix: reach, grasp, lift, move to fork.
    goto(
        object_pos + np.array([0.0, 0.0, float(args.pregrasp_hover_z)]),
        -1.0,
        max_speed=float(args.approach_max_speed),
        max_accel=float(args.approach_max_accel),
    )
    goto(
        object_pos + np.array([0.0, 0.0, float(args.pregrasp_touch_z)]),
        -1.0,
        tol=float(args.touch_goto_tol),
        max_steps=int(args.touch_goto_max_steps),
        max_speed=float(args.touch_max_speed),
        max_accel=float(args.touch_max_accel),
    )

    grasp_step = -1
    for i in range(int(args.grasp_close_steps)):
        action = np.zeros(env.action_dim, dtype=np.float32)
        live_obj = obs[f"{TARGET_OBJECT_NAME}_pos"].copy()
        live_ee = obs["robot0_eef_pos"].copy()
        grasp_delta = live_obj - live_ee
        xy_cmd = np.clip(
            float(args.grasp_track_gain) * grasp_delta[:2],
            -float(args.grasp_track_max_xy),
            float(args.grasp_track_max_xy),
        ) * float(args.grasp_close_xy_scale)
        z_cmd = np.clip(
            float(args.grasp_track_gain) * grasp_delta[2],
            -float(args.grasp_close_max_down_z),
            float(args.grasp_track_max_z),
        )
        z_cmd = np.clip(
            z_cmd + float(args.grasp_close_upward_bias),
            -float(args.grasp_close_max_down_z),
            float(args.grasp_track_max_z),
        )
        action[:2] = xy_cmd
        action[2] = z_cmd
        action[-1] = 1.0
        step_once(action)
        if env._check_grasp(env.robots[0].gripper, target_obj.contact_geoms):
            if grasp_step < 0:
                grasp_step = len(recorder.actions) - 1
            if i >= int(args.grasp_min_settle_steps):
                if int(args.grasp_hold_steps) > 0:
                    hold(1.0, int(args.grasp_hold_steps))
                break

    goto(
        np.array([object_pos[0], object_pos[1], carry_z]),
        1.0,
        max_speed=float(args.lift_max_speed),
        max_accel=float(args.lift_max_accel),
    )
    fork = np.array([obstacle_pos[0] - 0.02, obstacle_pos[1] + float(args.fork_y_offset), carry_z])
    goto(
        fork,
        1.0,
        max_speed=float(args.lift_max_speed),
        max_accel=float(args.lift_max_accel),
    )
    branch_step = len(recorder.actions) - 1

    # Divergence: left vs right route around the obstacle.
    side_sign = -1.0 if strategy_label == "A" else 1.0
    side_y_mid = 0.5 * (side_y_pre + side_y_post)
    waypoints = [
        np.array([obstacle_pos[0] + side_sign * side_x, obstacle_pos[1] + side_y_pre, carry_z]),
        np.array([obstacle_pos[0] + side_sign * side_x, obstacle_pos[1] + side_y_mid, carry_z]),
        np.array([obstacle_pos[0] + side_sign * side_x, obstacle_pos[1] + side_y_post, carry_z]),
    ]
    for waypoint in waypoints:
        goto(
            waypoint,
            1.0,
            max_speed=float(args.detour_max_speed),
            max_accel=float(args.detour_max_accel),
        )

    goto(
        np.array([target_pos[0], target_pos[1], carry_z]),
        1.0,
        max_speed=float(args.place_max_speed),
        max_accel=float(args.place_max_accel),
    )
    goto(
        np.array([target_pos[0], target_pos[1], target_pos[2] + 0.10]),
        1.0,
        max_speed=float(args.place_max_speed),
        max_accel=float(args.place_max_accel),
    )
    place_step = len(recorder.actions) - 1

    for i in range(int(args.release_steps)):
        action = np.zeros(env.action_dim, dtype=np.float32)
        action[-1] = -1.0
        if i >= int(args.release_lift_start):
            action[2] = float(args.release_lift_z)
        step_once(action)
    goto(
        np.array([target_pos[0], target_pos[1], carry_z]),
        -1.0,
        max_speed=float(args.retreat_max_speed),
        max_accel=float(args.retreat_max_accel),
    )
    if int(args.final_open_hold_steps) > 0:
        hold(-1.0, int(args.final_open_hold_steps))

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


def _pair_quality_metrics(a: RolloutResult, b: RolloutResult) -> Dict[str, float]:
    bstep = int(min(a.branch_step, b.branch_step, len(a.actions) - 1, len(b.actions) - 1))
    pre_n = int(min(max(2, bstep), len(a.actions), len(b.actions)))
    if pre_n <= 0:
        pre_n = int(min(len(a.actions), len(b.actions)))

    pre_action_l2 = float(np.linalg.norm(a.actions[:pre_n, :3] - b.actions[:pre_n, :3], axis=1).mean())
    pre_ee_l2 = float(np.linalg.norm(a.ee_pos[:pre_n] - b.ee_pos[:pre_n], axis=1).mean())
    pre_object_l2 = float(np.linalg.norm(a.object_pos[:pre_n] - b.object_pos[:pre_n], axis=1).mean())

    ia = int(np.clip(bstep, 0, len(a.agentview_rgb) - 1))
    ib = int(np.clip(bstep, 0, len(b.agentview_rgb) - 1))
    img_a = a.agentview_rgb[ia].astype(np.float32)
    img_b = b.agentview_rgb[ib].astype(np.float32)
    branch_image_mse = float(np.mean((img_a - img_b) ** 2))
    branch_ee_l2 = float(np.linalg.norm(a.ee_pos[ia] - b.ee_pos[ib]))
    branch_object_l2 = float(np.linalg.norm(a.object_pos[ia] - b.object_pos[ib]))

    ga = int(np.clip(a.grasp_step, 0, len(a.ee_pos) - 1))
    gb = int(np.clip(b.grasp_step, 0, len(b.ee_pos) - 1))
    grasp_step_diff = float(abs(int(a.grasp_step) - int(b.grasp_step)))
    a_grasp_obj_dist = float(np.linalg.norm(a.ee_pos[ga] - a.object_pos[ga]))
    b_grasp_obj_dist = float(np.linalg.norm(b.ee_pos[gb] - b.object_pos[gb]))
    grasp_obj_dist_max = float(max(a_grasp_obj_dist, b_grasp_obj_dist))
    grasp_object_gap = float(np.linalg.norm(a.object_pos[ga] - b.object_pos[gb]))

    return {
        "pre_action_l2": pre_action_l2,
        "pre_ee_l2": pre_ee_l2,
        "pre_object_l2": pre_object_l2,
        "branch_image_mse": branch_image_mse,
        "branch_ee_l2": branch_ee_l2,
        "branch_object_l2": branch_object_l2,
        "grasp_step_diff": grasp_step_diff,
        "grasp_obj_dist_max": grasp_obj_dist_max,
        "grasp_object_gap": grasp_object_gap,
        "grasp_step_a": float(a.grasp_step),
        "grasp_step_b": float(b.grasp_step),
    }


def _pair_quality_ok(metrics: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, str]:
    checks = [
        ("pre_action_l2", float(args.max_pre_action_l2)),
        ("pre_ee_l2", float(args.max_pre_ee_l2)),
        ("pre_object_l2", float(args.max_pre_object_l2)),
        ("branch_image_mse", float(args.max_branch_image_mse)),
        ("branch_ee_l2", float(args.max_branch_ee_l2)),
        ("branch_object_l2", float(args.max_branch_object_l2)),
        ("grasp_step_diff", float(args.max_grasp_step_diff)),
        ("grasp_obj_dist_max", float(args.max_grasp_obj_dist)),
        ("grasp_object_gap", float(args.max_grasp_object_gap)),
    ]
    for key, threshold in checks:
        value = float(metrics[key])
        if value > threshold:
            return False, f"{key}={value:.6f}>{threshold:.6f}"
    if int(metrics["grasp_step_a"]) < 0 or int(metrics["grasp_step_b"]) < 0:
        return False, "missing grasp_step"
    return True, ""


def main() -> None:
    global TARGET_CONTAINER_NAME
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
        "--seed-candidates",
        type=str,
        default="6,7,8,9,16,21,23,25,33",
    )
    parser.add_argument("--carry-z", type=float, default=0.218)
    parser.add_argument("--fork-y-offset", type=float, default=-0.145)
    parser.add_argument("--side-x", type=float, default=0.125)
    parser.add_argument("--side-y-pre-a", type=float, default=-0.05)
    parser.add_argument("--side-y-pre-b", type=float, default=-0.10)
    parser.add_argument("--side-y-post", type=float, default=0.16)
    parser.add_argument("--goto-gain", type=float, default=9.0)
    parser.add_argument("--goto-tol", type=float, default=0.01)
    parser.add_argument("--goto-max-steps", type=int, default=160)
    parser.add_argument("--goto-max-speed", type=float, default=0.85)
    parser.add_argument("--goto-max-axis-speed", type=float, default=0.85)
    parser.add_argument("--goto-max-accel", type=float, default=0.22)
    parser.add_argument("--goto-stable-delta", type=float, default=3e-4)
    parser.add_argument("--goto-stable-factor", type=float, default=1.7)
    parser.add_argument("--goto-stable-steps", type=int, default=4)
    parser.add_argument("--approach-max-speed", type=float, default=0.70)
    parser.add_argument("--approach-max-accel", type=float, default=0.18)
    parser.add_argument("--touch-max-speed", type=float, default=0.42)
    parser.add_argument("--touch-max-accel", type=float, default=0.12)
    parser.add_argument("--lift-max-speed", type=float, default=0.58)
    parser.add_argument("--lift-max-accel", type=float, default=0.16)
    parser.add_argument("--detour-max-speed", type=float, default=0.38)
    parser.add_argument("--detour-max-accel", type=float, default=0.08)
    parser.add_argument("--place-max-speed", type=float, default=0.48)
    parser.add_argument("--place-max-accel", type=float, default=0.12)
    parser.add_argument("--retreat-max-speed", type=float, default=0.38)
    parser.add_argument("--retreat-max-accel", type=float, default=0.08)
    parser.add_argument("--pregrasp-hover-z", type=float, default=0.14)
    parser.add_argument("--pregrasp-touch-z", type=float, default=0.03)
    parser.add_argument("--touch-goto-tol", type=float, default=0.028)
    parser.add_argument("--touch-goto-max-steps", type=int, default=60)
    parser.add_argument("--grasp-close-steps", type=int, default=16)
    parser.add_argument("--grasp-min-settle-steps", type=int, default=2)
    parser.add_argument("--grasp-hold-steps", type=int, default=0)
    parser.add_argument("--grasp-track-gain", type=float, default=6.0)
    parser.add_argument("--grasp-track-max-xy", type=float, default=0.05)
    parser.add_argument("--grasp-track-max-z", type=float, default=0.02)
    parser.add_argument("--grasp-close-xy-scale", type=float, default=0.18)
    parser.add_argument("--grasp-close-max-down-z", type=float, default=0.0)
    parser.add_argument("--grasp-close-upward-bias", type=float, default=0.010)
    parser.add_argument("--release-steps", type=int, default=8)
    parser.add_argument("--release-lift-start", type=int, default=4)
    parser.add_argument("--release-lift-z", type=float, default=0.16)
    parser.add_argument("--final-open-hold-steps", type=int, default=0)
    parser.add_argument(
        "--enforce-pair-quality",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out seeds that fail pre-branch consistency / grasp quality checks.",
    )
    parser.add_argument("--max-pre-action-l2", type=float, default=0.090)
    parser.add_argument("--max-pre-ee-l2", type=float, default=0.014)
    parser.add_argument("--max-pre-object-l2", type=float, default=0.015)
    parser.add_argument("--max-branch-image-mse", type=float, default=360.0)
    parser.add_argument("--max-branch-ee-l2", type=float, default=0.024)
    parser.add_argument("--max-branch-object-l2", type=float, default=0.015)
    parser.add_argument("--max-grasp-step-diff", type=float, default=2.0)
    parser.add_argument("--max-grasp-obj-dist", type=float, default=0.042)
    parser.add_argument("--max-grasp-object-gap", type=float, default=0.015)
    args = parser.parse_args()

    if not args.bddl_file.exists():
        raise FileNotFoundError(f"Missing BDDL file: {args.bddl_file}")

    if not TARGET_CONTAINER_NAME:
        TARGET_CONTAINER_NAME = _infer_target_container_name(args.bddl_file)
    print(f"Using target container name: {TARGET_CONTAINER_NAME}")

    seeds = _parse_seeds(args.seed_candidates)
    trajectories_dir = args.output_root / "trajectories"
    metadata_dir = args.output_root / "metadata"
    analysis_dir = args.output_root / "analysis"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    problem_info = BDDLUtils.get_problem_info(str(args.bddl_file))
    instruction = problem_info["language_instruction"]

    accepted: List[Dict] = []

    for seed in seeds:
        if len(accepted) >= args.num_groups:
            break

        env, _ = _build_env(args.bddl_file)
        env.seed(seed)
        env.reset()
        snapshot = env.sim.get_state().flatten().copy()

        result_a = _rollout_strategy(env, snapshot, "A", args)
        result_b = _rollout_strategy(env, snapshot, "B", args)
        env.close()

        if not (result_a.success and result_b.success):
            print(
                f"[skip] seed={seed} "
                f"A(success={result_a.success}, collision={result_a.collision}) "
                f"B(success={result_b.success}, collision={result_b.collision})"
            )
            continue
        quality_metrics = _pair_quality_metrics(result_a, result_b)
        if bool(args.enforce_pair_quality):
            ok, reason = _pair_quality_ok(quality_metrics, args)
            if not ok:
                print(f"[skip] seed={seed} quality={reason}")
                continue

        group_id = len(accepted) + 1
        a_h5 = trajectories_dir / f"group_{group_id:02d}_A.hdf5"
        b_h5 = trajectories_dir / f"group_{group_id:02d}_B.hdf5"
        a_json = metadata_dir / f"group_{group_id:02d}_A.json"
        b_json = metadata_dir / f"group_{group_id:02d}_B.json"

        _save_rollout_hdf5(a_h5, result_a, group_id, seed, args.bddl_file, instruction)
        _save_rollout_hdf5(b_h5, result_b, group_id, seed, args.bddl_file, instruction)
        _save_metadata_json(a_json, result_a, group_id, seed, args.bddl_file, instruction)
        _save_metadata_json(b_json, result_b, group_id, seed, args.bddl_file, instruction)

        accepted.append(
            {
                "group_id": group_id,
                "seed": seed,
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
                "quality": {k: float(v) for k, v in quality_metrics.items()},
            }
        )

        print(f"[ok] group={group_id:02d} seed={seed} saved A/B trajectories")

    if len(accepted) < args.num_groups:
        raise RuntimeError(
            f"Collected {len(accepted)} valid groups, less than requested {args.num_groups}. "
            f"Try adding more seed candidates."
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
