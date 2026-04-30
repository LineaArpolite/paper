import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def _resize_video_frames(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize uint8 video frames shaped [T, H, W, C] to target resolution.

    Uses pure-NumPy nearest upsampling for integer scales to avoid extra deps.
    Falls back to PIL bilinear resize for non-integer scales.
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames [T,H,W,C], got shape={frames.shape}")
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB frames with C=3, got shape={frames.shape}")

    _, src_h, src_w, _ = frames.shape
    if src_h == target_h and src_w == target_w:
        return frames

    if target_h % src_h == 0 and target_w % src_w == 0:
        scale_h = target_h // src_h
        scale_w = target_w // src_w
        out = np.repeat(np.repeat(frames, scale_h, axis=1), scale_w, axis=2)
        return out.astype(frames.dtype, copy=False)

    # Non-integer ratio fallback.
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Non-integer resize requires Pillow. Install pillow or use integer scale factors."
        ) from exc

    resized = []
    for fr in frames:
        img = Image.fromarray(fr)
        img = img.resize((target_w, target_h), Image.BILINEAR)
        resized.append(np.asarray(img, dtype=frames.dtype))
    return np.stack(resized, axis=0)


def _process_one_file(src_file: Path, dst_file: Path, target_h: int, target_w: int) -> None:
    shutil.copy2(src_file, dst_file)

    with h5py.File(dst_file, "r+") as f:
        obs = f["data"]["demo_1"]["obs"]
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            ds = obs[key]
            arr = ds[()]
            resized = _resize_video_frames(arr, target_h, target_w)

            compression = ds.compression
            compression_opts = ds.compression_opts
            chunks = ds.chunks
            shuffle = ds.shuffle
            fletcher32 = ds.fletcher32

            del obs[key]
            obs.create_dataset(
                key,
                data=resized,
                compression=compression,
                compression_opts=compression_opts,
                chunks=chunks if chunks is None else True,
                shuffle=shuffle,
                fletcher32=fletcher32,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Input dataset root with trajectories/ and metadata/ subfolders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output dataset root (must be new unless --overwrite).",
    )
    parser.add_argument("--target-height", type=int, default=256)
    parser.add_argument("--target-width", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    input_traj = input_root / "trajectories"
    input_meta = input_root / "metadata"

    if not input_traj.exists():
        raise FileNotFoundError(f"Missing trajectories dir: {input_traj}")
    if not input_meta.exists():
        raise FileNotFoundError(f"Missing metadata dir: {input_meta}")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {output_root} (use --overwrite)")
        shutil.rmtree(output_root)

    output_traj = output_root / "trajectories"
    output_meta = output_root / "metadata"
    output_analysis = output_root / "analysis"
    output_vis = output_root / "visualizations"
    output_traj.mkdir(parents=True, exist_ok=True)
    output_meta.mkdir(parents=True, exist_ok=True)

    # Keep metadata/analysis/visualizations for convenience, without touching originals.
    shutil.copytree(input_meta, output_meta, dirs_exist_ok=True)
    if (input_root / "analysis").exists():
        shutil.copytree(input_root / "analysis", output_analysis, dirs_exist_ok=True)
    if (input_root / "visualizations").exists():
        shutil.copytree(input_root / "visualizations", output_vis, dirs_exist_ok=True)

    traj_files = sorted(input_traj.glob("group_*_[AB].hdf5"))
    if not traj_files:
        raise FileNotFoundError(f"No trajectory files under: {input_traj}")

    for i, src_file in enumerate(traj_files, start=1):
        dst_file = output_traj / src_file.name
        _process_one_file(src_file, dst_file, args.target_height, args.target_width)
        print(f"[{i:02d}/{len(traj_files):02d}] resized {src_file.name}")

    print(f"Done. New dataset root: {output_root}")
    print(f"Target resolution: {args.target_height}x{args.target_width}")


if __name__ == "__main__":
    main()
