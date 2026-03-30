import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np


def load_traj(path: Path) -> Dict:
    with h5py.File(path, "r") as f:
        ep = f["data/demo_1"]
        obs = ep["obs"]
        return {
            "path": str(path),
            "actions": ep["actions"][()],
            "states": ep["states"][()],
            "imgs": obs["agentview_rgb"][()],
            "eye": obs["eye_in_hand_rgb"][()],
            "ee": obs["ee_pos"][()],
            "obj": obs["object_pos"][()],
            "obstacle": obs["obstacle_pos"][()],
            "success": int(ep.attrs.get("success", 0)),
            "collision": int(ep.attrs.get("collision", 0)),
            "branch_step": int(ep.attrs.get("branch_step", max(1, len(ep["actions"]) // 3))),
            "grasp_step": int(ep.attrs.get("grasp_step", -1)),
            "place_step": int(ep.attrs.get("place_step", -1)),
            "obstacle_bbox_xy": np.asarray(
                json.loads(ep.attrs.get("obstacle_bbox_xy_json", "[0,0,0,0]")), dtype=np.float32
            ),
        }


def choose_key_indices(length: int, grasp: int, branch: int, place: int) -> List[int]:
    idx = [0]
    if grasp >= 0:
        idx += [max(0, grasp - 12), grasp]
    idx += [max(0, branch - 12), branch, min(length - 1, branch + 24)]
    if place >= 0:
        idx += [place]
    idx += [length - 1]
    idx = sorted(set(int(np.clip(i, 0, length - 1)) for i in idx))
    return idx


def _maybe_flip(img: np.ndarray, flip_vertical: bool) -> np.ndarray:
    return np.flipud(img) if flip_vertical else img


def _save_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
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


def _effective_fps(
    kept_frames: int,
    total_steps: int,
    user_fps: int,
    source_step_fps: float,
    preserve_native_duration: bool,
) -> float:
    if not preserve_native_duration:
        return float(max(1, user_fps))
    if kept_frames <= 0 or total_steps <= 0:
        return float(max(1, user_fps))
    return float(max(1.0, kept_frames * float(source_step_fps) / float(total_steps)))


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    da = np.asarray(a, dtype=np.int16)
    db = np.asarray(b, dtype=np.int16)
    return float(np.mean(np.abs(da - db)))


def _frame_at(imgs: np.ndarray, idx: int) -> np.ndarray:
    if len(imgs) == 0:
        raise ValueError("Empty frame sequence")
    i = int(np.clip(idx, 0, len(imgs) - 1))
    return imgs[i]


def _downsample_indices(indices: List[int], max_frames: int) -> List[int]:
    if len(indices) <= max_frames:
        return indices
    picks = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
    return [indices[i] for i in picks]


def _adaptive_single_indices(
    imgs: np.ndarray,
    max_frames: int,
    min_frame_diff: float,
    must_keep: List[int],
    max_static_skip: int,
) -> List[int]:
    n = len(imgs)
    if n <= 2:
        return list(range(n))

    keep = [0]
    static_run = 0
    for i in range(1, n - 1):
        is_static = _frame_diff(imgs[i], imgs[i - 1]) < min_frame_diff
        if not is_static:
            keep.append(i)
            static_run = 0
        else:
            static_run += 1
            if static_run >= max_static_skip:
                keep.append(i)
                static_run = 0
    keep.append(n - 1)

    for x in must_keep:
        if 0 <= x < n:
            keep.append(int(x))
    keep = sorted(set(keep))

    key_set = set(int(x) for x in must_keep if 0 <= x < n)
    key_set.update({0, n - 1})
    regular = [i for i in keep if i not in key_set]
    max_regular = max(0, max_frames - len(key_set))
    regular = _downsample_indices(regular, max_regular)
    merged = sorted(set(regular).union(key_set))
    return merged


def _adaptive_pair_indices(
    a_imgs: np.ndarray,
    b_imgs: np.ndarray,
    max_frames: int,
    min_frame_diff: float,
    must_keep: List[int],
    max_static_skip: int,
) -> List[int]:
    n = max(len(a_imgs), len(b_imgs))
    if n <= 2:
        return list(range(n))

    keep = [0]
    static_run = 0
    for i in range(1, n - 1):
        diff = 0.5 * (
            _frame_diff(_frame_at(a_imgs, i), _frame_at(a_imgs, i - 1))
            + _frame_diff(_frame_at(b_imgs, i), _frame_at(b_imgs, i - 1))
        )
        if diff >= min_frame_diff:
            keep.append(i)
            static_run = 0
        else:
            static_run += 1
            if static_run >= max_static_skip:
                keep.append(i)
                static_run = 0
    keep.append(n - 1)

    for x in must_keep:
        if 0 <= x < n:
            keep.append(int(x))
    keep = sorted(set(keep))

    key_set = set(int(x) for x in must_keep if 0 <= x < n)
    key_set.update({0, n - 1})
    regular = [i for i in keep if i not in key_set]
    max_regular = max(0, max_frames - len(key_set))
    regular = _downsample_indices(regular, max_regular)
    merged = sorted(set(regular).union(key_set))
    return merged


def save_side_by_side_mp4(
    a_imgs: np.ndarray,
    b_imgs: np.ndarray,
    out_path: Path,
    fps: int,
    flip_vertical: bool,
    max_frames: int,
    min_frame_diff: float,
    must_keep: List[int],
    max_static_skip: int,
    source_step_fps: float,
    preserve_native_duration: bool,
) -> None:
    n = max(len(a_imgs), len(b_imgs))
    idx = _adaptive_pair_indices(
        a_imgs,
        b_imgs,
        max_frames=max_frames,
        min_frame_diff=min_frame_diff,
        must_keep=must_keep,
        max_static_skip=max_static_skip,
    )
    frames = []
    for i in idx:
        a = _maybe_flip(_frame_at(a_imgs, i), flip_vertical)
        b = _maybe_flip(_frame_at(b_imgs, i), flip_vertical)
        frames.append(np.concatenate([a, b], axis=1))
    out_fps = _effective_fps(
        kept_frames=len(idx),
        total_steps=n,
        user_fps=fps,
        source_step_fps=source_step_fps,
        preserve_native_duration=preserve_native_duration,
    )
    _save_mp4(frames, out_path, fps=out_fps)


def visualize_pair(
    dataset_root: Path,
    pair_idx: int,
    output_dir: Path,
    make_mp4: bool,
    flip_vertical: bool,
    max_video_frames: int,
    min_frame_diff: float,
    max_static_skip: int,
    fps: int,
    source_step_fps: float,
    preserve_native_duration: bool,
) -> bool:
    a_path = dataset_root / "trajectories" / f"group_{pair_idx:02d}_A.hdf5"
    b_path = dataset_root / "trajectories" / f"group_{pair_idx:02d}_B.hdf5"
    if not a_path.exists() or not b_path.exists():
        print(f"[warn] missing pair {pair_idx:02d}")
        return False

    a = load_traj(a_path)
    b = load_traj(b_path)

    rollouts_dir = output_dir / "rollout_videos"
    paired_dir = output_dir / "paired_comparisons"
    topdown_dir = output_dir / "topdown_paths"
    fork_dir = output_dir / "fork_alignment"
    collision_dir = output_dir / "collision_debug"

    if make_mp4:
        for label, data in [("A", a), ("B", b)]:
            key = [int(data["grasp_step"]), int(data["branch_step"]), int(data["place_step"])]
            idx = _adaptive_single_indices(
                data["imgs"],
                max_frames=max_video_frames,
                min_frame_diff=min_frame_diff,
                must_keep=key,
                max_static_skip=max_static_skip,
            )
            frames = [_maybe_flip(data["imgs"][i], flip_vertical) for i in idx]
            out_fps = _effective_fps(
                kept_frames=len(idx),
                total_steps=len(data["imgs"]),
                user_fps=fps,
                source_step_fps=source_step_fps,
                preserve_native_duration=preserve_native_duration,
            )
            _save_mp4(frames, rollouts_dir / f"group_{pair_idx:02d}_{label}.mp4", fps=out_fps)

        pair_key = [
            int(a["branch_step"]),
            int(b["branch_step"]),
            int(a["grasp_step"]),
            int(b["grasp_step"]),
            int(a["place_step"]),
            int(b["place_step"]),
        ]
        save_side_by_side_mp4(
            a["imgs"],
            b["imgs"],
            paired_dir / f"group_{pair_idx:02d}_A_vs_B.mp4",
            fps=fps,
            flip_vertical=flip_vertical,
            max_frames=max_video_frames,
            min_frame_diff=min_frame_diff,
            must_keep=pair_key,
            max_static_skip=max_static_skip,
            source_step_fps=source_step_fps,
            preserve_native_duration=preserve_native_duration,
        )

    # Topdown path overlay
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(a["ee"][:, 0], a["ee"][:, 1], color="tab:blue", lw=2.0, label="A ee path")
    ax.plot(b["ee"][:, 0], b["ee"][:, 1], color="tab:orange", lw=2.0, label="B ee path")
    ax.plot(a["obj"][:, 0], a["obj"][:, 1], color="tab:blue", ls="--", alpha=0.8, label="A object path")
    ax.plot(b["obj"][:, 0], b["obj"][:, 1], color="tab:orange", ls="--", alpha=0.8, label="B object path")

    bbox = a["obstacle_bbox_xy"]
    rect = plt.Rectangle(
        (bbox[0], bbox[2]),
        bbox[1] - bbox[0],
        bbox[3] - bbox[2],
        fill=False,
        edgecolor="red",
        linewidth=2.0,
        linestyle="-",
        label="obstacle footprint",
    )
    ax.add_patch(rect)

    ba = max(0, min(a["branch_step"], len(a["ee"]) - 1))
    bb = max(0, min(b["branch_step"], len(b["ee"]) - 1))
    ax.scatter([a["ee"][ba, 0]], [a["ee"][ba, 1]], c="tab:blue", s=36, marker="o", label="A branch")
    ax.scatter([b["ee"][bb, 0]], [b["ee"][bb, 1]], c="tab:orange", s=36, marker="o", label="B branch")

    ax.set_title(f"Group {pair_idx:02d} Topdown Paths")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    topdown_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(topdown_dir / f"group_{pair_idx:02d}_topdown.png", dpi=180)
    plt.close(fig)

    # Fork alignment keyframes + pre-branch distance curve
    pre_n = max(5, min(a["branch_step"], b["branch_step"]))
    pre_dist = np.linalg.norm(a["ee"][:pre_n] - b["ee"][:pre_n], axis=1)

    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(2, 4)
    ax_curve = fig.add_subplot(gs[0, :])
    ax_curve.plot(np.arange(pre_n), pre_dist, color="black", lw=1.8)
    ax_curve.set_title("Pre-Branch EE Distance (A vs B)")
    ax_curve.set_xlabel("step")
    ax_curve.set_ylabel("L2 distance")
    ax_curve.grid(alpha=0.3)

    a_keys = choose_key_indices(len(a["imgs"]), a["grasp_step"], a["branch_step"], a["place_step"])
    b_keys = choose_key_indices(len(b["imgs"]), b["grasp_step"], b["branch_step"], b["place_step"])
    a_sel = [a_keys[0], a_keys[len(a_keys) // 2], a_keys[-1]]
    b_sel = [b_keys[0], b_keys[len(b_keys) // 2], b_keys[-1]]

    for j, k in enumerate(a_sel):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(_maybe_flip(a["imgs"][k], flip_vertical))
        ax.axis("off")
        ax.set_title(f"A t={k}")

    for j, k in enumerate(b_sel):
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(_maybe_flip(b["imgs"][k], flip_vertical))
        ax.axis("off")
        ax.set_title(f"B t={k}")

    fork_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fork_dir / f"group_{pair_idx:02d}_fork_alignment.png", dpi=180)
    plt.close(fig)

    # Collision debug card
    collision_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 2.2))
    ax.axis("off")
    text = (
        f"Group {pair_idx:02d}\n"
        f"A success={a['success']} collision={a['collision']}\n"
        f"B success={b['success']} collision={b['collision']}"
    )
    ax.text(0.02, 0.55, text, fontsize=11, va="center", ha="left")
    fig.tight_layout()
    fig.savefig(collision_dir / f"group_{pair_idx:02d}_status.png", dpi=180)
    plt.close(fig)

    # Overview figure
    fig = plt.figure(figsize=(12.8, 8.0))
    gs = fig.add_gridspec(2, 3)

    axx = fig.add_subplot(gs[0, :])
    n = min(len(a["actions"]), len(b["actions"]))
    t = np.arange(n)
    axx.plot(t, a["actions"][:n, 0], label="A action_x", lw=1.0)
    axx.plot(t, b["actions"][:n, 0], label="B action_x", lw=1.0)
    axx.plot(t, a["actions"][:n, 1], label="A action_y", lw=1.0, alpha=0.7)
    axx.plot(t, b["actions"][:n, 1], label="B action_y", lw=1.0, alpha=0.7)
    branch = min(a["branch_step"], b["branch_step"], n - 1)
    axx.axvline(branch, color="red", linestyle="--", linewidth=1.2, label=f"branch={branch}")
    axx.set_title("Action Curves (x/y) with Branch Marker")
    axx.set_xlabel("time step")
    axx.set_ylabel("action value")
    axx.legend(ncol=5, fontsize=8)
    axx.grid(alpha=0.25)

    a_xy = np.cumsum(a["actions"][:n, :2], axis=0)
    b_xy = np.cumsum(b["actions"][:n, :2], axis=0)
    a_xy = a_xy - a_xy[0]
    b_xy = b_xy - b_xy[0]
    a_pre = a_xy[: branch + 1]
    b_pre = b_xy[: branch + 1]
    a_post = a_xy[branch:]
    b_post = b_xy[branch:]

    axy = fig.add_subplot(gs[1, :])
    axy.plot(a_pre[:, 0], a_pre[:, 1], color="tab:blue", linewidth=2.0, label="A pre-branch")
    axy.plot(b_pre[:, 0], b_pre[:, 1], color="tab:orange", linewidth=2.0, label="B pre-branch")
    axy.plot(a_post[:, 0], a_post[:, 1], color="tab:blue", linestyle="--", linewidth=2.0, label="A post-branch")
    axy.plot(b_post[:, 0], b_post[:, 1], color="tab:orange", linestyle="--", linewidth=2.0, label="B post-branch")
    axy.scatter([a_xy[branch, 0]], [a_xy[branch, 1]], c="red", s=30, label="branch point")
    axy.set_title("Integrated XY Action Path (qualitative trajectory contrast)")
    axy.set_xlabel("integrated action x")
    axy.set_ylabel("integrated action y")
    axy.legend(ncol=5, fontsize=8)
    axy.grid(alpha=0.25)

    fig.suptitle(
        f"Paired Trajectory {pair_idx:04d} | A success={a['success']} B success={b['success']} | branch={branch}",
        fontsize=13,
    )
    fig.tight_layout()
    paired_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paired_dir / f"group_{pair_idx:02d}_overview.png", dpi=180)
    plt.close(fig)

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data_collection_outputs")
    parser.add_argument("--output-dir", default="data_collection_outputs/visualizations")
    parser.add_argument("--pairs", default="1,2,3,4,5")
    parser.add_argument(
        "--make-mp4",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write rollout and paired comparison videos as .mp4 files.",
    )
    parser.add_argument(
        "--flip-vertical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip image vertically for human-friendly viewing. Disable with --no-flip-vertical.",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=420,
        help="Upper bound of frames kept in each generated mp4.",
    )
    parser.add_argument(
        "--min-frame-diff",
        type=float,
        default=0.55,
        help="Mean absolute RGB diff threshold used to drop near-static frames.",
    )
    parser.add_argument(
        "--max-static-skip",
        type=int,
        default=4,
        help="In near-static segments, keep one frame every N steps to avoid jumpy playback.",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--source-step-fps", type=float, default=20.0)
    parser.add_argument(
        "--preserve-native-duration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adjust output fps so video duration matches native rollout duration.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)

    pair_ids: List[int] = []
    for x in args.pairs.split(","):
        x = x.strip()
        if x:
            pair_ids.append(int(x))

    created = 0
    for pid in pair_ids:
        ok = visualize_pair(
            root,
            pid,
            out,
            make_mp4=args.make_mp4,
            flip_vertical=args.flip_vertical,
            max_video_frames=int(args.max_video_frames),
            min_frame_diff=float(args.min_frame_diff),
            max_static_skip=int(args.max_static_skip),
            fps=int(args.fps),
            source_step_fps=float(args.source_step_fps),
            preserve_native_duration=bool(args.preserve_native_duration),
        )
        if ok:
            created += 1

    print(f"Created visualizations for {created} pair(s) in: {out}")


if __name__ == "__main__":
    main()
