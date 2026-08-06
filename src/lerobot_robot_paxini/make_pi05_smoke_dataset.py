from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset

from .paxini_reader import CsvPaxiniReader, MockPaxiniReader, PaxiniReader
from .tactile_render import TactileHeatmapRenderer, make_mock_taxel_map, taxels_to_stacked_9x9

TACTILE_KEY = "observation.tactile.primary"
IMAGE_KEY = "observation.images.front"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

SO101_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _make_front_image(tactile: np.ndarray, image_size: int) -> np.ndarray:
    if tactile.ndim == 3:
        tactile = tactile.max(axis=0)
    scale_y = max(1, int(np.ceil(image_size / tactile.shape[0])))
    scale_x = max(1, int(np.ceil(image_size / tactile.shape[1])))
    resized = np.kron(tactile, np.ones((scale_y, scale_x), dtype=np.float32))[:image_size, :image_size]

    image = np.full((image_size, image_size, 3), 24, dtype=np.uint8)
    image[..., 0] = np.maximum(image[..., 0], np.rint(resized * 255.0).astype(np.uint8))
    image[..., 1] = np.maximum(image[..., 1], np.rint(resized * 96.0).astype(np.uint8))
    return image


def _make_features(tactile_shape: tuple[int, ...], camera_image_size: int) -> dict:
    return {
        STATE_KEY: {
            "dtype": "float32",
            "shape": (len(SO101_JOINT_NAMES),),
            "names": SO101_JOINT_NAMES,
        },
        IMAGE_KEY: {
            "dtype": "image",
            "shape": (camera_image_size, camera_image_size, 3),
            "names": ["height", "width", "channels"],
        },
        TACTILE_KEY: {
            "dtype": "float32",
            "shape": tactile_shape,
            "names": ["height", "width"] if len(tactile_shape) == 2 else ["sensor", "height", "width"],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": (len(SO101_JOINT_NAMES),),
            "names": SO101_JOINT_NAMES,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a tiny LeRobot dataset with camera, state, action, and Paxini tactile features."
    )
    parser.add_argument("--repo-id", default="local/paxini_pi05_smoke")
    parser.add_argument("--root", default=None)
    parser.add_argument("--csv", default=None, help="Optional PXSR DataLogging CSV path.")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default="touch with ring and pinky tactile sensors")
    parser.add_argument("--num-taxels", type=int, default=154)
    parser.add_argument("--tactile-image-size", type=int, default=64)
    parser.add_argument("--tactile-representation", choices=["heatmap", "stacked_9x9"], default="heatmap")
    parser.add_argument("--tactile-grid-normalize", action="store_true")
    parser.add_argument("--camera-image-size", type=int, default=224)
    parser.add_argument("--csv-scale", type=float, default=1.0)
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive.")

    taxel_map = make_mock_taxel_map(args.num_taxels)
    tactile_renderer = TactileHeatmapRenderer(taxel_map=taxel_map, image_size=args.tactile_image_size)

    tactile_reader: PaxiniReader
    if args.csv:
        tactile_reader = CsvPaxiniReader(
            args.csv,
            num_taxels=args.num_taxels,
            taxel_value_mode="z",
            taxel_scale=args.csv_scale,
            loop=True,
        )
    else:
        tactile_reader = MockPaxiniReader(num_taxels=args.num_taxels)

    tactile_shape = (
        (2, 9, 9)
        if args.tactile_representation == "stacked_9x9"
        else (args.tactile_image_size, args.tactile_image_size)
    )
    features = _make_features(tactile_shape, args.camera_image_size)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=Path(args.root) if args.root else None,
        robot_type="paxini_so101_follower",
        use_videos=False,
    )

    tactile_reader.connect()
    try:
        for frame_idx in range(args.frames):
            sample = tactile_reader.read_latest()
            if args.tactile_representation == "stacked_9x9":
                tactile = taxels_to_stacked_9x9(
                    sample.taxels,
                    normalize=args.tactile_grid_normalize,
                )
            else:
                tactile = tactile_renderer.render(sample.taxels)
            phase = frame_idx / max(args.frames - 1, 1)
            state = np.array(
                [
                    np.sin(phase * np.pi * 2.0),
                    np.cos(phase * np.pi * 2.0),
                    np.sin(phase * np.pi),
                    np.cos(phase * np.pi),
                    phase * 2.0 - 1.0,
                    float(tactile.max(initial=0.0) * 100.0),
                ],
                dtype=np.float32,
            )
            action = np.roll(state, -1).astype(np.float32)

            dataset.add_frame(
                {
                    STATE_KEY: state,
                    IMAGE_KEY: _make_front_image(tactile, args.camera_image_size),
                    TACTILE_KEY: tactile,
                    ACTION_KEY: action,
                    "task": args.task,
                }
            )
        dataset.save_episode()
    finally:
        tactile_reader.disconnect()
        dataset.finalize()

    print(f"Created dataset repo_id={args.repo_id}")
    print(f"Dataset root={dataset.root}")
    print(f"Frames={args.frames}, tactile_shape={tactile_shape}")


if __name__ == "__main__":
    main()
