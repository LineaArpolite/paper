import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


def _load_episode(path: Path) -> Dict:
    with h5py.File(path, "r") as f:
        ep = f["data/demo_1"]
        obs = ep["obs"]
        return {
            "path": str(path),
            "actions": ep["actions"][()],
            "states": ep["states"][()],
            "dones": ep["dones"][()],
            "success": int(ep.attrs.get("success", 0)),
            "collision": int(ep.attrs.get("collision", 0)),
            "branch_step": int(ep.attrs.get("branch_step", max(1, ep["actions"].shape[0] // 3))),
            "agentview": obs["agentview_rgb"][()],
            "ee": obs["ee_pos"][()],
            "obj": obs["object_pos"][()],
        }


def _get_pairs(root: Path) -> List[Tuple[Path, Path]]:
    a_files = sorted((root / "trajectories").glob("group_*_A.hdf5"))
    pairs = []
    for a in a_files:
        b = Path(str(a).replace("_A.hdf5", "_B.hdf5"))
        if b.exists():
            pairs.append((a, b))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data_collection_outputs")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    pairs = _get_pairs(root)
    if not pairs:
        raise RuntimeError(
            f"No paired trajectories found under: {root}/trajectories with suffixes _A and _B"
        )

    pre_dists = []
    pre_ee_dists = []
    branch_img_mse = []
    post_divergence = []
    post_path_divergence = []

    details = []
    success_a = 0
    success_b = 0
    collision_free_a = 0
    collision_free_b = 0

    for a_path, b_path in pairs:
        a = _load_episode(a_path)
        b = _load_episode(b_path)

        success_a += int(a["success"] == 1)
        success_b += int(b["success"] == 1)
        collision_free_a += int(a["collision"] == 0)
        collision_free_b += int(b["collision"] == 0)

        ba = min(a["branch_step"], a["actions"].shape[0] - 1)
        bb = min(b["branch_step"], b["actions"].shape[0] - 1)
        bstep = min(ba, bb)

        pre_n = max(2, bstep)

        pre_a = a["actions"][:pre_n, :3]
        pre_b = b["actions"][:pre_n, :3]
        n = min(len(pre_a), len(pre_b))
        d_pre = float(np.linalg.norm(pre_a[:n] - pre_b[:n], axis=1).mean())

        pre_ee_a = a["ee"][:pre_n]
        pre_ee_b = b["ee"][:pre_n]
        n2 = min(len(pre_ee_a), len(pre_ee_b))
        d_pre_ee = float(np.linalg.norm(pre_ee_a[:n2] - pre_ee_b[:n2], axis=1).mean())

        img_a = a["agentview"][bstep].astype(np.float32)
        img_b = b["agentview"][bstep].astype(np.float32)
        mse = float(np.mean((img_a - img_b) ** 2))

        post_a = a["actions"][bstep:, :3]
        post_b = b["actions"][bstep:, :3]
        m = min(len(post_a), len(post_b))
        d_post = float(np.mean(np.abs(post_a[:m] - post_b[:m])))

        post_ee_a = a["ee"][bstep:, :2]
        post_ee_b = b["ee"][bstep:, :2]
        m2 = min(len(post_ee_a), len(post_ee_b))
        d_post_path = float(np.linalg.norm(post_ee_a[:m2] - post_ee_b[:m2], axis=1).mean())

        pre_dists.append(d_pre)
        pre_ee_dists.append(d_pre_ee)
        branch_img_mse.append(mse)
        post_divergence.append(d_post)
        post_path_divergence.append(d_post_path)

        details.append(
            {
                "A": str(a_path),
                "B": str(b_path),
                "pre_action_l2": d_pre,
                "pre_ee_l2": d_pre_ee,
                "branch_image_mse": mse,
                "post_action_abs_diff": d_post,
                "post_path_separation": d_post_path,
                "branch_step": int(bstep),
                "A_success": int(a["success"]),
                "B_success": int(b["success"]),
                "A_collision": int(a["collision"]),
                "B_collision": int(b["collision"]),
            }
        )

    summary = {
        "num_pairs": len(details),
        "success_rate_A": success_a / len(details),
        "success_rate_B": success_b / len(details),
        "collision_free_rate_A": collision_free_a / len(details),
        "collision_free_rate_B": collision_free_b / len(details),
        "pre_action_l2_mean": float(np.mean(pre_dists)),
        "pre_ee_l2_mean": float(np.mean(pre_ee_dists)),
        "branch_image_mse_mean": float(np.mean(branch_img_mse)),
        "post_action_abs_diff_mean": float(np.mean(post_divergence)),
        "post_path_separation_mean": float(np.mean(post_path_divergence)),
    }

    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    out_json = Path(args.output_json) if args.output_json else analysis_dir / "per_group_similarity.json"
    out_json.write_text(json.dumps({"summary": summary, "details": details}, indent=2), encoding="utf-8")

    csv_path = analysis_dir / "summary_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    report_path = analysis_dir / "summary_report.md"
    lines = [
        "# Dual Strategy Collection Report",
        "",
        f"- Number of valid A/B pairs: {summary['num_pairs']}",
        f"- A success rate: {summary['success_rate_A']:.3f}",
        f"- B success rate: {summary['success_rate_B']:.3f}",
        f"- A collision-free rate: {summary['collision_free_rate_A']:.3f}",
        f"- B collision-free rate: {summary['collision_free_rate_B']:.3f}",
        "",
        "## Similarity / Divergence Metrics",
        "",
        f"- Pre-branch action L2 mean: {summary['pre_action_l2_mean']:.6f}",
        f"- Pre-branch ee L2 mean: {summary['pre_ee_l2_mean']:.6f}",
        f"- Branch image MSE mean: {summary['branch_image_mse_mean']:.3f}",
        f"- Post-branch action abs diff mean: {summary['post_action_abs_diff_mean']:.6f}",
        f"- Post-branch path separation mean: {summary['post_path_separation_mean']:.6f}",
        "",
        "## Acceptance Checklist",
        "",
        f"- [x] 5 A/B groups collected: {summary['num_pairs'] >= 5}",
        f"- [x] A/B trajectories successful: {summary['success_rate_A'] == 1.0 and summary['success_rate_B'] == 1.0}",
        f"- [x] A/B trajectories collision-free: {summary['collision_free_rate_A'] == 1.0 and summary['collision_free_rate_B'] == 1.0}",
        "",
        "Per-group values are listed in `summary_table.csv` and `per_group_similarity.json`.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_json}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
