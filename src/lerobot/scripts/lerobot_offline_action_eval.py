#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare policy predictions with demonstration actions on a recorded episode.

This is an offline diagnostic: it never connects to a robot or sends motor commands.
The saved checkpoint preprocessor and postprocessor are used for every prediction.
"""

import argparse
import csv
import gc
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata, resolve_delta_timestamps
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_TACTILE

logger = logging.getLogger(__name__)


def _parse_model(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model must have the form NAME=/path/to/pretrained_model")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Model name and path must both be non-empty")
    return name, Path(path).expanduser()


def _parse_phase(value: str) -> tuple[str, int, int]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Phase must have the form NAME:START_FRAME:END_FRAME")
    name, start, end = parts
    try:
        start_frame, end_frame = int(start), int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Phase frame bounds must be integers") from exc
    if start_frame > end_frame:
        raise argparse.ArgumentTypeError("Phase start frame must not exceed its end frame")
    return name, start_frame, end_frame


def _parse_num_steps(value: str) -> list[int | None]:
    """Parse a comma-separated inference-step sweep. Empty means the checkpoint default."""
    value = value.strip()
    if not value:
        return [None]
    steps: list[int | None] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n < 1:
            raise ValueError("--num-steps values must be at least 1")
        steps.append(n)
    return steps or [None]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_mean(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def _safe_max(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def compute_action_metrics(
    predicted: np.ndarray,
    demonstrated: np.ndarray,
    valid: np.ndarray,
    states: np.ndarray,
    direction_threshold: float,
) -> dict[str, Any]:
    """Compute chunk, first-action, direction, and state-to-action jump metrics."""
    if predicted.shape != demonstrated.shape:
        raise ValueError(f"Prediction/demo shape mismatch: {predicted.shape} != {demonstrated.shape}")
    if valid.shape != predicted.shape[:2]:
        raise ValueError(f"Valid-mask shape mismatch: {valid.shape} != {predicted.shape[:2]}")

    valid_3d = np.broadcast_to(valid[..., None], predicted.shape)
    error = predicted - demonstrated
    abs_error = np.where(valid_3d, np.abs(error), np.nan)
    sq_error = np.where(valid_3d, np.square(error), np.nan)

    first_error = predicted[:, 0] - demonstrated[:, 0]
    action_dim = predicted.shape[-1]
    comparable_dim = min(action_dim, states.shape[-1])
    pred_motion = predicted[:, 0, :comparable_dim] - states[:, :comparable_dim]
    demo_motion = demonstrated[:, 0, :comparable_dim] - states[:, :comparable_dim]
    active = np.abs(demo_motion) >= direction_threshold
    direction_match = np.where(active, np.sign(pred_motion) == np.sign(demo_motion), np.nan)

    per_joint = []
    for joint_idx in range(action_dim):
        joint_sq = sq_error[..., joint_idx]
        joint = {
            "joint_index": joint_idx,
            "chunk_mae": _safe_mean(abs_error[..., joint_idx]),
            "chunk_rmse": math.sqrt(_safe_mean(joint_sq) or 0.0),
            "first_action_mae": _safe_mean(np.abs(first_error[:, joint_idx])),
        }
        if joint_idx < comparable_dim:
            joint.update(
                {
                    "direction_agreement": _safe_mean(direction_match[:, joint_idx]),
                    "mean_abs_predicted_jump": _safe_mean(np.abs(pred_motion[:, joint_idx])),
                    "max_abs_predicted_jump": _safe_max(np.abs(pred_motion[:, joint_idx])),
                    "mean_abs_demo_jump": _safe_mean(np.abs(demo_motion[:, joint_idx])),
                }
            )
        per_joint.append(joint)

    chunk_mse = _safe_mean(sq_error)
    return {
        "chunk_mae": _safe_mean(abs_error),
        "chunk_rmse": math.sqrt(chunk_mse or 0.0),
        "first_action_mae": _safe_mean(np.abs(first_error)),
        "first_action_rmse": math.sqrt(_safe_mean(np.square(first_error)) or 0.0),
        "direction_agreement": _safe_mean(direction_match),
        "mean_abs_predicted_jump": _safe_mean(np.abs(pred_motion)),
        "max_abs_predicted_jump": _safe_max(np.abs(pred_motion)),
        "mean_abs_demo_jump": _safe_mean(np.abs(demo_motion)),
        "per_joint": per_joint,
    }


def _feature_names(meta: LeRobotDatasetMetadata, key: str, size: int) -> list[str]:
    names = meta.features.get(key, {}).get("names")
    if isinstance(names, list) and len(names) == size:
        return [str(name) for name in names]
    return [f"joint_{idx}" for idx in range(size)]


def _tensor_scalar(value: Any) -> int | float:
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _raw_tactile_stats(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats = {}
    for key, value in item.items():
        if key != OBS_TACTILE and not key.startswith(f"{OBS_TACTILE}."):
            continue
        tensor = torch.as_tensor(value).float()
        stats[key] = {
            "shape": list(tensor.shape),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "mean": float(tensor.mean()),
        }
    return stats


def _prepare_observation(item: dict[str, Any], camera_keys: list[str]) -> dict[str, Any]:
    observation = {}
    for key, value in item.items():
        if key == ACTION or key == f"{ACTION}_is_pad":
            continue
        if key in camera_keys and isinstance(value, torch.Tensor) and value.dtype == torch.uint8:
            value = value.to(dtype=torch.float32) / 255.0
        observation[key] = value
    return observation


def _plot_predictions(
    frame_indices: np.ndarray,
    predicted: np.ndarray,
    demonstrated: np.ndarray,
    joint_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is unavailable; skipping %s", output_path)
        return

    action_dim = predicted.shape[-1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(15, max(3 * action_dim, 5)), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, axis in enumerate(axes):
        axis.plot(frame_indices, demonstrated[:, 0, idx], label="demonstration", linewidth=2)
        axis.plot(frame_indices, predicted[:, 0, idx], label="prediction", alpha=0.85)
        axis.set_ylabel(joint_names[idx])
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("dataset frame_index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_first_action_csv(
    output_path: Path,
    frame_indices: np.ndarray,
    timestamps: np.ndarray,
    predicted: np.ndarray,
    demonstrated: np.ndarray,
    states: np.ndarray,
    joint_names: list[str],
    direction_threshold: float,
) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_index",
                "timestamp",
                "joint",
                "state",
                "demonstrated_action",
                "predicted_action",
                "absolute_error",
                "demo_delta_from_state",
                "predicted_delta_from_state",
                "direction_match",
            ],
        )
        writer.writeheader()
        comparable_dim = min(states.shape[-1], predicted.shape[-1])
        for sample_idx, frame_index in enumerate(frame_indices):
            for joint_idx, joint_name in enumerate(joint_names):
                state = states[sample_idx, joint_idx] if joint_idx < comparable_dim else np.nan
                demo = demonstrated[sample_idx, 0, joint_idx]
                pred = predicted[sample_idx, 0, joint_idx]
                demo_delta = demo - state
                pred_delta = pred - state
                direction_match: bool | None = None
                if joint_idx < comparable_dim and abs(demo_delta) >= direction_threshold:
                    direction_match = bool(np.sign(demo_delta) == np.sign(pred_delta))
                writer.writerow(
                    {
                        "frame_index": int(frame_index),
                        "timestamp": float(timestamps[sample_idx]),
                        "joint": joint_name,
                        "state": float(state),
                        "demonstrated_action": float(demo),
                        "predicted_action": float(pred),
                        "absolute_error": float(abs(pred - demo)),
                        "demo_delta_from_state": float(demo_delta),
                        "predicted_delta_from_state": float(pred_delta),
                        "direction_match": direction_match,
                    }
                )


def _evaluate_model(
    name: str,
    checkpoint: Path,
    dataset: LeRobotDataset,
    selected_indices: list[int],
    output_dir: Path,
    device: str,
    seed: int,
    direction_threshold: float,
    phases: list[tuple[str, int, int]],
    num_seeds: int,
    teacher_loss_repeats: int,
    num_steps: int | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    logger.info("Loading %s from %s", name, checkpoint)
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(cfg.type)
    policy = policy_class.from_pretrained(checkpoint).to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    predictions = []
    demonstrations = []
    valid_masks = []
    states = []
    frame_indices = []
    timestamps = []
    timing_rows = []
    tactile_rows = []
    task_strings = set()
    prediction_samples = []
    teacher_loss_samples = []

    for ordinal, dataset_idx in enumerate(selected_indices):
        item = dataset[dataset_idx]
        demonstration = torch.as_tensor(item[ACTION]).float()
        if demonstration.ndim == 1:
            demonstration = demonstration.unsqueeze(0)
        valid = ~torch.as_tensor(
            item.get(f"{ACTION}_is_pad", torch.zeros(demonstration.shape[0], dtype=torch.bool))
        ).bool()
        state = torch.as_tensor(item[OBS_STATE]).float().reshape(-1, item[OBS_STATE].shape[-1])[-1]
        tactile_stats = _raw_tactile_stats(item)
        task_strings.add(str(item.get("task", "")))

        observation = _prepare_observation(item, dataset.meta.camera_keys)
        if teacher_loss_repeats > 0:
            # Dataset items are unbatched. Observations are batched by the saved
            # processor, while a 50xD action chunk needs an explicit batch axis.
            observation[ACTION] = demonstration.unsqueeze(0)
        start = time.perf_counter()
        processed = preprocessor(observation)
        after_preprocess = time.perf_counter()

        frame_index = int(_tensor_scalar(item["frame_index"]))
        frame_predictions = []
        inference_start = time.perf_counter()
        for seed_offset in range(num_seeds):
            _set_seed(seed + frame_index * num_seeds + seed_offset)
            with torch.inference_mode():
                predicted_normalized = policy.predict_action_chunk(processed)
            predicted_sample = postprocessor(predicted_normalized)
            frame_predictions.append(torch.as_tensor(predicted_sample).detach().float().cpu().squeeze(0))
        after_inference = time.perf_counter()
        frame_teacher_losses = []
        for repeat in range(teacher_loss_repeats):
            _set_seed(seed + 1_000_000 + frame_index * max(teacher_loss_repeats, 1) + repeat)
            with torch.inference_mode():
                loss, _ = policy(processed, reduction="none")
            frame_teacher_losses.append(float(loss.detach().float().cpu().mean()))
        after_postprocess = time.perf_counter()

        predicted = frame_predictions[0]
        horizon = min(predicted.shape[0], demonstration.shape[0])
        predictions.append(predicted[:horizon].numpy())
        prediction_samples.append(np.stack([sample[:horizon].numpy() for sample in frame_predictions]))
        teacher_loss_samples.append(frame_teacher_losses)
        demonstrations.append(demonstration[:horizon].numpy())
        valid_masks.append(valid[:horizon].numpy())
        states.append(state.numpy())
        frame_indices.append(frame_index)
        timestamps.append(float(_tensor_scalar(item["timestamp"])))
        timing_rows.append(
            {
                "preprocessing_s": after_preprocess - start,
                "inference_s": (after_inference - inference_start) / num_seeds,
                "postprocessing_s": 0.0,
                "teacher_forward_s": (
                    (after_postprocess - after_inference) / teacher_loss_repeats
                    if teacher_loss_repeats > 0
                    else 0.0
                ),
                "total_s": after_postprocess - start,
            }
        )
        tactile_rows.append(
            {
                "frame_index": frame_index,
                "raw": tactile_stats,
                "expected_keys": list(cfg.tactile_features),
                "mask": {key: key in processed for key in cfg.tactile_features},
            }
        )
        logger.info("%s: frame %s (%s/%s)", name, frame_index, ordinal + 1, len(selected_indices))

    predicted_np = np.stack(predictions)
    demonstrated_np = np.stack(demonstrations)
    valid_np = np.stack(valid_masks)
    states_np = np.stack(states)
    frames_np = np.asarray(frame_indices)
    timestamps_np = np.asarray(timestamps)
    prediction_samples_np = np.stack(prediction_samples)
    teacher_losses_np = (
        np.asarray(teacher_loss_samples, dtype=np.float32)
        if teacher_loss_repeats > 0
        else np.empty((len(selected_indices), 0), dtype=np.float32)
    )
    metrics = compute_action_metrics(predicted_np, demonstrated_np, valid_np, states_np, direction_threshold)

    joint_names = _feature_names(dataset.meta, ACTION, predicted_np.shape[-1])
    for joint, joint_name in zip(metrics["per_joint"], joint_names, strict=True):
        joint["name"] = joint_name

    phase_metrics = {}
    for phase_name, start_frame, end_frame in phases:
        mask = (frames_np >= start_frame) & (frames_np <= end_frame)
        if mask.any():
            phase_metrics[phase_name] = compute_action_metrics(
                predicted_np[mask],
                demonstrated_np[mask],
                valid_np[mask],
                states_np[mask],
                direction_threshold,
            )

    timing_summary = {
        key: {
            "mean": float(np.mean([row[key] for row in timing_rows])),
            "p95": float(np.percentile([row[key] for row in timing_rows], 95)),
        }
        for key in timing_rows[0]
    }
    diagnostics = _compute_stochastic_diagnostics(
        prediction_samples=prediction_samples_np,
        demonstrated=demonstrated_np,
        valid=valid_np,
        states=states_np,
        teacher_losses=teacher_losses_np,
        direction_threshold=direction_threshold,
        joint_names=joint_names,
    )
    summary = {
        "model": name,
        "checkpoint": str(checkpoint),
        "policy_type": cfg.type,
        "tactile_injection_mode": getattr(cfg, "tactile_injection_mode", None),
        "num_evaluated_frames": len(selected_indices),
        "tasks": sorted(task_strings),
        "metrics": metrics,
        "phase_metrics": phase_metrics,
        "timing_seconds": timing_summary,
        "stochastic_diagnostics": diagnostics,
        "tactile": tactile_rows,
    }

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        model_dir / "action_chunks.npz",
        frame_indices=frames_np,
        timestamps=timestamps_np,
        predicted=predicted_np,
        predicted_samples=prediction_samples_np,
        demonstrated=demonstrated_np,
        valid=valid_np,
        states=states_np,
        teacher_losses=teacher_losses_np,
    )
    _write_first_action_csv(
        model_dir / "first_actions.csv",
        frames_np,
        timestamps_np,
        predicted_np,
        demonstrated_np,
        states_np,
        joint_names,
        direction_threshold,
    )
    _plot_predictions(
        frames_np,
        predicted_np,
        demonstrated_np,
        joint_names,
        model_dir / "first_action_comparison.png",
        f"{name}: predicted vs demonstrated first action",
    )

    del policy, preprocessor, postprocessor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, {
        "frame_indices": frames_np,
        "predicted": predicted_np,
        "demonstrated": demonstrated_np,
    }


def _compute_stochastic_diagnostics(
    prediction_samples: np.ndarray,
    demonstrated: np.ndarray,
    valid: np.ndarray,
    states: np.ndarray,
    teacher_losses: np.ndarray,
    direction_threshold: float,
    joint_names: list[str],
) -> dict[str, Any]:
    """Summarize seed variance and relate teacher-forced loss to sampled action error."""
    mean_prediction = prediction_samples.mean(axis=1)
    mean_prediction_metrics = compute_action_metrics(
        mean_prediction, demonstrated, valid, states, direction_threshold
    )
    for joint, joint_name in zip(mean_prediction_metrics["per_joint"], joint_names, strict=True):
        joint["name"] = joint_name

    first_actions = prediction_samples[:, :, 0, :]
    first_action_std = first_actions.std(axis=1)
    first_action_range = np.ptp(first_actions, axis=1)
    per_seed_first_mae = np.mean(np.abs(first_actions - demonstrated[:, None, 0, :]), axis=(0, 2))
    sampling = {
        "num_seeds": int(prediction_samples.shape[1]),
        "mean_first_action_std": float(first_action_std.mean()),
        "max_first_action_std": float(first_action_std.max()),
        "mean_first_action_seed_range": float(first_action_range.mean()),
        "max_first_action_seed_range": float(first_action_range.max()),
        "per_seed_first_action_mae": per_seed_first_mae.tolist(),
        "per_joint_first_action_std": {
            name: float(first_action_std[:, idx].mean()) for idx, name in enumerate(joint_names)
        },
        "mean_prediction_metrics": mean_prediction_metrics,
    }

    teacher_summary: dict[str, Any] = {"num_repeats": int(teacher_losses.shape[1])}
    if teacher_losses.shape[1] > 0:
        teacher_mean_per_frame = teacher_losses.mean(axis=1)
        sampled_error_per_frame = np.mean(np.abs(first_actions - demonstrated[:, None, 0, :]), axis=(1, 2))
        if np.std(teacher_mean_per_frame) > 0 and np.std(sampled_error_per_frame) > 0:
            correlation = float(np.corrcoef(teacher_mean_per_frame, sampled_error_per_frame)[0, 1])
        else:
            correlation = None
        teacher_summary.update(
            {
                "mean": float(teacher_losses.mean()),
                "std_across_all_samples": float(teacher_losses.std()),
                "mean_within_frame_std": float(teacher_losses.std(axis=1).mean()),
                "min_frame_mean": float(teacher_mean_per_frame.min()),
                "max_frame_mean": float(teacher_mean_per_frame.max()),
                "correlation_with_sampled_first_action_mae": correlation,
            }
        )
    return {"sampling": sampling, "teacher_forced_loss": teacher_summary}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo-id", default="local/paxini_cup_left_60")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--direction-threshold", type=float, default=1e-3)
    parser.add_argument("--phase", action="append", type=_parse_phase, default=[])
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--teacher-loss-repeats", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1")
    if args.teacher_loss_repeats < 0:
        raise ValueError("--teacher-loss-repeats must not be negative")
    for _, checkpoint in args.model:
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    first_cfg = PreTrainedConfig.from_pretrained(args.model[0][1])
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    delta_timestamps = resolve_delta_timestamps(first_cfg, metadata)
    dataset = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    selected_indices = list(range(0, len(dataset), args.stride))
    if args.max_frames > 0:
        selected_indices = selected_indices[: args.max_frames]
    if not selected_indices:
        raise ValueError("No dataset frames selected")

    # Decode one sample before loading the 14 GB PI0.5 base model. This fails fast when
    # FFmpeg/TorchCodec is unavailable instead of discovering it after model loading.
    logger.info("Validating dataset and video decoding with frame %s", selected_indices[0])
    try:
        dataset[selected_indices[0]]
    except RuntimeError as exc:
        if "libtorchcodec" in str(exc) or "FFmpeg" in str(exc):
            raise RuntimeError(
                "Video decoding is unavailable. Install FFmpeg on the evaluation host "
                "(Ubuntu: apt-get update && apt-get install -y ffmpeg)."
            ) from exc
        raise

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Offline-only evaluation: episode=%s frames=%s/%s stride=%s",
        args.episode,
        len(selected_indices),
        len(dataset),
        args.stride,
    )
    summaries = {}
    for name, checkpoint in args.model:
        summary, _ = _evaluate_model(
            name=name,
            checkpoint=checkpoint,
            dataset=dataset,
            selected_indices=selected_indices,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            direction_threshold=args.direction_threshold,
            phases=args.phase,
            num_seeds=args.num_seeds,
            teacher_loss_repeats=args.teacher_loss_repeats,
        )
        summaries[name] = summary

    report = {
        "dataset_repo_id": args.dataset_repo_id,
        "dataset_root": str(args.dataset_root),
        "episode": args.episode,
        "stride": args.stride,
        "selected_frames": len(selected_indices),
        "models": summaries,
    }
    (args.output_dir / "comparison_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Evaluation complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
