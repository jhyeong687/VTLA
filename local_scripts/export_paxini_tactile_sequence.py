#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Paxini tactile parquet frames as PNG image sequence/video.")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("outputs/paxini_tactile_dataset/data/chunk-000/file-000.parquet"),
        help="Input LeRobot parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paxini_tactile_dataset/tactile_sequence"),
        help="Directory for frame_XXXXXX.png outputs.",
    )
    parser.add_argument("--column", default="observation.tactile.primary", help="Tactile column name.")
    parser.add_argument("--video", type=Path, default=None, help="Optional MP4 path for a tactile heatmap video.")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for --video output.")
    parser.add_argument("--no-frames", action="store_true", help="Only write --video, not frame_XXXXXX.png files.")
    parser.add_argument(
        "--normalization",
        choices=["global", "frame"],
        default="global",
        help="Use one scale for the whole stream, or rescale each frame independently.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete existing frame_*.png files first.")
    return parser.parse_args()


def to_uint8(arr: np.ndarray, scale_max: float) -> np.ndarray:
    scale_max = scale_max if scale_max > 0 else 1.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=scale_max, neginf=0.0)
    arr = np.clip(arr, 0.0, scale_max) / scale_max
    return np.rint(arr * 255.0).astype(np.uint8)


def colorize_heatmap(gray: np.ndarray) -> np.ndarray:
    """Map 0..255 pressure values to a dark-blue -> yellow -> red RGB heatmap."""
    x = gray.astype(np.float32) / 255.0
    red = np.clip(3.0 * x - 0.35, 0.0, 1.0)
    green = np.clip(3.0 * x - 0.95, 0.0, 1.0)
    blue = np.clip(1.25 - 2.4 * x, 0.0, 1.0)
    return np.rint(np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def make_panel(arr: np.ndarray, scale_max: float, size: tuple[int, int], label: str | None = None) -> Image.Image:
    gray = to_uint8(arr, scale_max)
    img = Image.fromarray(colorize_heatmap(gray), mode="RGB")
    img = img.resize(size, resample=Image.Resampling.NEAREST)
    if label:
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 82, 24), fill=(0, 0, 0))
        draw.text((8, 6), label, fill=(245, 245, 245))
    return img


def render_tactile_frame(arr: np.ndarray, frame_idx: int, timestamp: float, scale_max: float) -> Image.Image:
    if arr.ndim == 2:
        img = make_panel(arr, scale_max, (512, 512))
    elif arr.ndim == 3:
        default_labels = ["ring", "pinky"]
        panels = []
        for channel_idx, channel in enumerate(arr):
            label = default_labels[channel_idx] if channel_idx < len(default_labels) else f"sensor {channel_idx}"
            panels.append(make_panel(channel, scale_max, (256, 256), label=label))
        gap = 16
        img = Image.new("RGB", (len(panels) * 256 + max(len(panels) - 1, 0) * gap, 256), (0, 0, 0))
        x = 0
        for panel in panels:
            img.paste(panel, (x, 0))
            x += panel.width + gap
    else:
        flat = arr.reshape(1, -1)
        img = make_panel(flat, scale_max, (512, 128))

    bar_h = 28
    canvas = Image.new("RGB", (img.width, img.height + bar_h), (18, 18, 18))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, img.height + 7),
        f"frame={frame_idx}  t={timestamp:.3f}s  min={arr.min():.3g} max={arr.max():.3g}",
        fill=(235, 235, 235),
    )
    return canvas


def write_video(path: Path, images: list[Image.Image], fps: float) -> None:
    if not images:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = images[0].size
    raw_path = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {raw_path}")
    try:
        for image in images:
            frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            writer.write(frame)
    finally:
        writer.release()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raw_path.replace(path)
        return

    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    raw_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(args.parquet)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old in args.output_dir.glob("frame_*.png"):
            old.unlink()

    columns = [args.column]
    optional_columns = ["frame_index", "timestamp"]
    schema_names = set(pq.read_schema(args.parquet).names)
    columns.extend(column for column in optional_columns if column in schema_names)

    table = pq.read_table(args.parquet, columns=columns)
    tactile_col = table[args.column]
    arrays = [np.asarray(tactile_col[i].as_py(), dtype=np.float32) for i in range(len(tactile_col))]

    frame_indices = table["frame_index"].to_pylist() if "frame_index" in table.column_names else list(range(len(arrays)))
    timestamps = table["timestamp"].to_pylist() if "timestamp" in table.column_names else [0.0] * len(arrays)

    global_max = max((float(arr.max(initial=0.0)) for arr in arrays), default=0.0)
    nonzero = 0
    images = []
    for i, arr in enumerate(arrays):
        frame_max = float(arr.max(initial=0.0))
        if frame_max > 0:
            nonzero += 1
        scale_max = global_max if args.normalization == "global" else frame_max
        frame_idx = int(frame_indices[i]) if frame_indices[i] is not None else i
        timestamp = float(timestamps[i]) if timestamps[i] is not None else 0.0
        image = render_tactile_frame(arr, frame_idx, timestamp, scale_max)
        if not args.no_frames:
            image.save(args.output_dir / f"frame_{i:06d}.png")
        if args.video is not None:
            images.append(image)

    if args.video is not None:
        write_video(args.video, images, args.fps)

    shape = arrays[0].shape if arrays else None
    print(f"parquet: {args.parquet}")
    print(f"output_dir: {args.output_dir}")
    if args.video is not None:
        print(f"video: {args.video}")
    print(f"frames: {len(arrays)}")
    print(f"nonzero_frames: {nonzero}")
    print(f"shape: {shape}")
    print(f"global_max: {global_max}")


if __name__ == "__main__":
    main()
