from __future__ import annotations

import csv
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from lerobot.utils.import_utils import require_package

from .types import PaxiniSample

TaxelValueMode = Literal["x", "y", "z", "magnitude"]


class PaxiniReader:
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def read_latest(self) -> PaxiniSample:
        raise NotImplementedError


@dataclass
class MockPaxiniReader(PaxiniReader):
    num_taxels: int
    phase: float = 0.0
    connected: bool = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            return PaxiniSample(taxels=np.zeros(self.num_taxels, dtype=np.float32), timestamp_s=time.time())

        self.phase += 0.07
        idx = np.arange(self.num_taxels, dtype=np.float32)
        center = (np.sin(self.phase) * 0.5 + 0.5) * max(self.num_taxels - 1, 1)
        taxels = np.exp(-0.5 * ((idx - center) / max(self.num_taxels / 12.0, 1.0)) ** 2).astype(np.float32)
        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())


class CsvPaxiniReader(PaxiniReader):
    """Replay PXSR DataLogging CSV rows as tactile samples.

    PXSR logs distributed force columns such as ``3-0-NxN-Z[0]``.  For ring+pinky
    experiments this normally gives two 77-taxel arrays, concatenated in CSV
    column order into a 154-value tactile vector.
    """

    _NXN_COL_RE = re.compile(r"^(?P<sensor>.+)-NxN-(?P<axis>[XYZ])\[(?P<idx>\d+)\]$")

    def __init__(
        self,
        csv_path: str | Path,
        *,
        num_taxels: int,
        taxel_value_mode: TaxelValueMode = "z",
        taxel_scale: float = 1.0,
        loop: bool = True,
    ):
        if num_taxels <= 0:
            raise ValueError(f"num_taxels must be positive, got {num_taxels}.")
        if taxel_value_mode not in ("x", "y", "z", "magnitude"):
            raise ValueError(
                f"taxel_value_mode must be one of 'x', 'y', 'z', or 'magnitude', got {taxel_value_mode!r}."
            )

        self.csv_path = Path(csv_path)
        self.num_taxels = num_taxels
        self.taxel_value_mode = taxel_value_mode
        self.taxel_scale = float(taxel_scale)
        self.loop = loop
        self.connected = False
        self._rows: list[dict[str, str]] = []
        self._columns: list[str] = []
        self._selected_columns: list[str] = []
        self._triplet_columns: list[tuple[str, str, str]] = []
        self._cursor = 0

    def connect(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"PXSR CSV not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            self._columns = list(reader.fieldnames or [])
            self._rows = list(reader)

        if not self._rows:
            raise ValueError(f"PXSR CSV has no data rows: {self.csv_path}")

        self._select_columns()
        self._cursor = 0
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            raise RuntimeError("CsvPaxiniReader is not connected.")
        if self._cursor >= len(self._rows):
            if not self.loop:
                self._cursor = len(self._rows) - 1
            else:
                self._cursor = 0

        row = self._rows[self._cursor]
        self._cursor += 1

        if self.taxel_value_mode == "magnitude":
            values = []
            for x_col, y_col, z_col in self._triplet_columns:
                x = self._parse_float(row.get(x_col))
                y = self._parse_float(row.get(y_col))
                z = self._parse_float(row.get(z_col))
                values.append(float(np.sqrt(x * x + y * y + z * z)))
            values_np = np.asarray(values, dtype=np.float32)
        else:
            values_np = np.asarray(
                [self._parse_float(row.get(col)) for col in self._selected_columns],
                dtype=np.float32,
            )

        taxels = np.zeros(self.num_taxels, dtype=np.float32)
        count = min(self.num_taxels, values_np.shape[0])
        taxels[:count] = values_np[:count] * self.taxel_scale
        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())

    def _select_columns(self) -> None:
        parsed: list[tuple[int, str, str, int, str]] = []
        column_order = {col: order for order, col in enumerate(self._columns)}
        for col in self._columns:
            match = self._NXN_COL_RE.match(col)
            if not match:
                continue
            parsed.append(
                (
                    column_order[col],
                    match.group("sensor"),
                    match.group("axis").lower(),
                    int(match.group("idx")),
                    col,
                )
            )

        if not parsed:
            numeric_cols = [col for col in self._columns if col.lower() != "timestamp"]
            if len(numeric_cols) < self.num_taxels:
                raise ValueError(
                    f"Could not find PXSR NxN columns in {self.csv_path}; "
                    f"only {len(numeric_cols)} generic numeric columns are available."
                )
            self._selected_columns = numeric_cols[: self.num_taxels]
            return

        if self.taxel_value_mode == "magnitude":
            by_point: dict[tuple[str, int], dict[str, str]] = {}
            first_order: dict[tuple[str, int], int] = {}
            for order, sensor, axis, idx, col in parsed:
                key = (sensor, idx)
                by_point.setdefault(key, {})[axis] = col
                first_order[key] = min(first_order.get(key, order), order)

            triplets = []
            for key, axes in by_point.items():
                if {"x", "y", "z"} <= axes.keys():
                    triplets.append((first_order[key], axes["x"], axes["y"], axes["z"]))
            triplets.sort(key=lambda item: item[0])
            self._triplet_columns = [(x, y, z) for _, x, y, z in triplets[: self.num_taxels]]
            if not self._triplet_columns:
                raise ValueError(f"No complete NxN X/Y/Z triplets found in {self.csv_path}.")
            return

        axis = self.taxel_value_mode
        selected = [(order, col) for order, _, parsed_axis, _, col in parsed if parsed_axis == axis]
        selected.sort(key=lambda item: item[0])
        self._selected_columns = [col for _, col in selected[: self.num_taxels]]
        if not self._selected_columns:
            raise ValueError(f"No NxN {axis.upper()} columns found in {self.csv_path}.")

    @staticmethod
    def _parse_float(value: str | None) -> float:
        if value is None or value == "":
            return 0.0
        try:
            return float(value)
        except ValueError:
            return 0.0


class SerialPaxiniReader(PaxiniReader):
    """Read connected Paxini distribution-force registers through paxini-utils."""

    def __init__(
        self,
        port: str,
        baudrate: int = 921600,
        *,
        num_taxels: int = 158,
        timeout_s: float = 0.5,
        taxel_value_mode: TaxelValueMode = "z",
        taxel_scale: float = 0.1,
        skip_bytes: int = 0,
        async_read: bool = True,
    ):
        if num_taxels <= 0:
            raise ValueError(f"num_taxels must be positive, got {num_taxels}.")
        if skip_bytes < 0:
            raise ValueError(f"skip_bytes must be non-negative, got {skip_bytes}.")
        if taxel_value_mode not in ("x", "y", "z", "magnitude"):
            raise ValueError(
                f"taxel_value_mode must be one of 'x', 'y', 'z', or 'magnitude', got {taxel_value_mode!r}."
            )

        self.port = port
        self.baudrate = baudrate
        self.num_taxels = num_taxels
        self.timeout_s = timeout_s
        self.taxel_value_mode = taxel_value_mode
        self.taxel_scale = float(taxel_scale)
        self.skip_bytes = skip_bytes
        self.async_read = async_read
        self.connected = False
        self._board: Any | None = None
        self._read_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_sample: PaxiniSample | None = None
        self._last_error: Exception | None = None
        self._sample_lock = threading.Lock()

    def connect(self) -> None:
        require_package("paxini-utils", extra="paxini", import_name="paxini_utils")
        from paxini_utils.high_speed_board import HighSpeedBoard

        self._board = HighSpeedBoard(self.port, baudrate=self.baudrate, timeout=self.timeout_s)
        self._board.open()
        with self._sample_lock:
            self._last_sample = None
            self._last_error = None
        self._board.disable_auto_push(timeout=0)
        self._board.get_connected_sensors()
        self._board.get_distribution_point_counts()
        self.connected = True
        if self.async_read:
            self._start_background_reader()

    def disconnect(self) -> None:
        self._stop_background_reader()
        if self._board is not None and self._board.is_open:
            try:
                self._board.disable_auto_push(timeout=0)
                time.sleep(0.05)
            finally:
                self._board.close()
        self._board = None
        self.connected = False

    def read_latest(self) -> PaxiniSample:
        if not self.connected:
            raise RuntimeError("SerialPaxiniReader is not connected.")

        if self.async_read:
            return self._read_latest_from_background()

        return self._read_and_decode_once()

    def _read_and_decode_once(self) -> PaxiniSample:
        if self._board is None:
            raise RuntimeError("SerialPaxiniReader is not connected.")
        try:
            readings = self._board.read_connected_distribution_forces(
                refresh_status=False,
                refresh_point_counts=False,
            )
        except Exception:
            with self._sample_lock:
                if self._last_sample is not None:
                    return self._last_sample
            raise

        sample = self._readings_to_sample(readings)
        with self._sample_lock:
            self._last_sample = sample
            self._last_error = None
        return sample

    def _read_latest_from_background(self) -> PaxiniSample:
        deadline = time.monotonic() + self.timeout_s
        while True:
            with self._sample_lock:
                if self._last_sample is not None:
                    return self._last_sample
                last_error = self._last_error

            if time.monotonic() >= deadline:
                if last_error is not None:
                    raise RuntimeError(
                        f"PXSR background reader has no sample yet: {last_error}"
                    ) from last_error
                raise TimeoutError(f"Timed out waiting for first PXSR sample on {self.port}.")
            time.sleep(0.002)

    def _start_background_reader(self) -> None:
        self._stop_event.clear()
        self._read_thread = threading.Thread(
            target=self._background_read_loop,
            name=f"paxini-reader-{self.port}",
            daemon=True,
        )
        self._read_thread.start()

    def _stop_background_reader(self) -> None:
        self._stop_event.set()
        if self._read_thread is not None:
            self._read_thread.join(timeout=self.timeout_s + 0.2)
            self._read_thread = None

    def _background_read_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = self._read_and_decode_once()
                with self._sample_lock:
                    self._last_sample = sample
                    self._last_error = None
            except Exception as exc:
                with self._sample_lock:
                    self._last_error = exc
                self._stop_event.wait(0.01)

    def _readings_to_sample(self, readings: tuple[Any, ...]) -> PaxiniSample:
        values: list[float] = []
        for reading in readings:
            for point in reading.distribution:
                values.append(self._point_value(point.force))

        taxels = np.zeros(self.num_taxels, dtype=np.float32)
        if self.skip_bytes:
            skip_points = self.skip_bytes // 3
            values = values[skip_points:]
        if not values:
            return PaxiniSample(taxels=taxels, timestamp_s=time.time())

        values_np = np.asarray(values, dtype=np.float32)
        count = min(self.num_taxels, values_np.shape[0])
        taxels[:count] = values_np[:count]
        return PaxiniSample(taxels=taxels, fz=float(taxels.max(initial=0.0)), timestamp_s=time.time())

    def _point_value(self, force: Any) -> float:
        if self.taxel_value_mode == "x":
            value = abs(float(force.raw_x))
        elif self.taxel_value_mode == "y":
            value = abs(float(force.raw_y))
        elif self.taxel_value_mode == "magnitude":
            value = float(
                np.sqrt(force.raw_x * force.raw_x + force.raw_y * force.raw_y + force.raw_z * force.raw_z)
            )
        else:
            value = float(force.raw_z)
        return value * self.taxel_scale
