#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Paxini tactile parquet frames as PNG image sequence.")
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


def render_tactile_frame(arr: np.ndarray, frame_idx: int, timestamp: float, scale_max: float) -> Image.Image:
    if arr.ndim == 2:
        img = Image.fromarray(to_uint8(arr, scale_max), mode="L")
        img = img.resize((512, 512), resample=Image.Resampling.NEAREST).convert("RGB")
    elif arr.ndim == 3:
        panels = [
            Image.fromarray(to_uint8(channel, scale_max), mode="L")
            .resize((256, 256), resample=Image.Resampling.NEAREST)
            .convert("RGB")
            for channel in arr
        ]
        gap = 16
        img = Image.new("RGB", (len(panels) * 256 + max(len(panels) - 1, 0) * gap, 256), (0, 0, 0))
        x = 0
        for panel in panels:
            img.paste(panel, (x, 0))
            x += panel.width + gap
    else:
        flat = arr.reshape(1, -1)
        img = Image.fromarray(to_uint8(flat, scale_max), mode="L")
        img = img.resize((512, 128), resample=Image.Resampling.NEAREST).convert("RGB")

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
    for i, arr in enumerate(arrays):
        frame_max = float(arr.max(initial=0.0))
        if frame_max > 0:
            nonzero += 1
        scale_max = global_max if args.normalization == "global" else frame_max
        frame_idx = int(frame_indices[i]) if frame_indices[i] is not None else i
        timestamp = float(timestamps[i]) if timestamps[i] is not None else 0.0
        image = render_tactile_frame(arr, frame_idx, timestamp, scale_max)
        image.save(args.output_dir / f"frame_{i:06d}.png")

    shape = arrays[0].shape if arrays else None
    print(f"parquet: {args.parquet}")
    print(f"output_dir: {args.output_dir}")
    print(f"frames: {len(arrays)}")
    print(f"nonzero_frames: {nonzero}")
    print(f"shape: {shape}")
    print(f"global_max: {global_max}")


if __name__ == "__main__":
    main()
