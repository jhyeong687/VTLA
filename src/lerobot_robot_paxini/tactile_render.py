from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TaxelMap:
    x_mm: np.ndarray
    y_mm: np.ndarray

    @property
    def num_taxels(self) -> int:
        return int(self.x_mm.shape[0])


def make_mock_taxel_map(num_taxels: int = 100) -> TaxelMap:
    cols = int(np.ceil(np.sqrt(num_taxels)))
    rows = int(np.ceil(num_taxels / cols))
    xs, ys = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    return TaxelMap(x_mm=xs.ravel()[:num_taxels], y_mm=ys.ravel()[:num_taxels])


def taxels_to_stacked_9x9(
    taxels: np.ndarray,
    *,
    sensors: int = 2,
    taxels_per_sensor: int = 77,
    normalize: bool = False,
) -> np.ndarray:
    """Convert 77-taxel M3025-like sensor arrays into a stacked 9x9 grid.

    The observed 77-point layout is treated as a 9x9 array with the four corner
    cells absent. Missing corners are filled with 0, and each sensor becomes one
    channel. For ring+pinky this returns shape ``(2, 9, 9)``.
    """
    values = np.asarray(taxels, dtype=np.float32).reshape(-1)
    expected = sensors * taxels_per_sensor
    if values.shape[0] < expected:
        padded = np.zeros(expected, dtype=np.float32)
        padded[: values.shape[0]] = values
        values = padded
    else:
        values = values[:expected]

    grid = np.zeros((sensors, 9, 9), dtype=np.float32)
    valid_positions = [
        (row, col) for row in range(9) for col in range(9) if not ((row in (0, 8)) and (col in (0, 8)))
    ]

    for sensor_idx in range(sensors):
        start = sensor_idx * taxels_per_sensor
        sensor_values = values[start : start + taxels_per_sensor]
        for value, (row, col) in zip(sensor_values, valid_positions, strict=True):
            grid[sensor_idx, row, col] = max(float(value), 0.0)

    if normalize:
        denom = float(grid.max(initial=0.0))
        if denom > 0:
            grid = np.clip(grid / denom, 0.0, 1.0)

    return grid.astype(np.float32, copy=False)


def load_taxel_map_xlsx(path: str | Path) -> TaxelMap:
    import pandas as pd

    df = pd.read_excel(path)
    columns = {str(col).strip().lower(): col for col in df.columns}
    x_col = columns.get("x_mm") or columns.get("x") or columns.get("x(mm)")
    y_col = columns.get("y_mm") or columns.get("y") or columns.get("y(mm)")
    if x_col is None or y_col is None:
        raise ValueError(f"Could not find x/y coordinate columns in {path}. Columns: {list(df.columns)}")

    return TaxelMap(
        x_mm=df[x_col].to_numpy(dtype=np.float32),
        y_mm=df[y_col].to_numpy(dtype=np.float32),
    )


class TactileHeatmapRenderer:
    def __init__(self, taxel_map: TaxelMap, image_size: int = 64, max_pressure: float | None = None):
        self.taxel_map = taxel_map
        self.image_size = image_size
        self.max_pressure = max_pressure

        x_range = max(float(taxel_map.x_mm.max() - taxel_map.x_mm.min()), 1e-6)
        y_range = max(float(taxel_map.y_mm.max() - taxel_map.y_mm.min()), 1e-6)
        self.x_px = np.rint((taxel_map.x_mm - taxel_map.x_mm.min()) / x_range * (image_size - 1)).astype(
            np.int32
        )
        self.y_px = np.rint((taxel_map.y_mm - taxel_map.y_mm.min()) / y_range * (image_size - 1)).astype(
            np.int32
        )

    def render(self, taxels: np.ndarray) -> np.ndarray:
        """Render taxel values into a normalized 2D tactile map."""
        values = np.asarray(taxels, dtype=np.float32)
        if values.shape[0] != self.taxel_map.num_taxels:
            raise ValueError(f"Expected {self.taxel_map.num_taxels} taxels, got {values.shape[0]}")

        grid = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        values = np.clip(values, 0.0, None)
        np.maximum.at(grid, (self.y_px, self.x_px), values)
        grid = self._smooth_3x3(grid)

        denom = self.max_pressure if self.max_pressure is not None else float(grid.max())
        if denom > 0:
            grid = np.clip(grid / denom, 0.0, 1.0)

        return grid.astype(np.float32, copy=False)

    def render_rgb(self, taxels: np.ndarray) -> np.ndarray:
        """Render taxel values as a uint8 RGB image for visualization only."""
        grid = self.render(taxels)
        gray = np.rint(grid * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)

    @staticmethod
    def _smooth_3x3(grid: np.ndarray) -> np.ndarray:
        padded = np.pad(grid, 1, mode="edge")
        return (
            padded[:-2, :-2]
            + 2.0 * padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2]
            + 4.0 * padded[1:-1, 1:-1]
            + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2]
            + 2.0 * padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 16.0
