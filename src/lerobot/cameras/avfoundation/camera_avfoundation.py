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

"""Native macOS AVFoundation camera backend."""

import ctypes
import logging
import platform
import time
from threading import Event, Lock
from typing import Any

import numpy as np
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_avfoundation import AVFoundationCameraConfig, ColorMode

logger = logging.getLogger(__name__)
_SAMPLE_BUFFER_DELEGATE_CLASS: Any | None = None
FPS_MATCH_ABS_TOL = 0.1


def _require_darwin() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("AVFoundationCamera is only available on macOS.")


def _import_avfoundation_stack():
    _require_darwin()
    try:
        import AVFoundation  # type: ignore[import-not-found]
        import CoreMedia  # type: ignore[import-not-found]
        import objc  # type: ignore[import-not-found]
        import Quartz  # type: ignore[import-not-found]
        from Foundation import NSObject  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "AVFoundationCamera requires PyObjC AVFoundation support. "
            "Install dependencies with `uv sync`."
        ) from e

    return AVFoundation, CoreMedia, Quartz, objc, NSObject


def _list_avfoundation_video_devices() -> list[dict[str, Any]]:
    avfoundation, core_media, _quartz, _objc, _ns_object = _import_avfoundation_stack()

    devices: list[dict[str, Any]] = []
    for index, device in enumerate(
        avfoundation.AVCaptureDevice.devicesWithMediaType_(avfoundation.AVMediaTypeVideo)
    ):
        devices.append(
            {
                "index": index,
                "name": str(device.localizedName()),
                "unique_id": str(device.uniqueID()),
                "formats": _device_formats(device, core_media),
            }
        )

    return devices


def _device_formats(device: Any, core_media: Any) -> list[dict[str, Any]]:
    formats = []
    for av_format in device.formats():
        description = av_format.formatDescription()
        dimensions = core_media.CMVideoFormatDescriptionGetDimensions(description)
        ranges = []
        for fps_range in av_format.videoSupportedFrameRateRanges():
            ranges.append(
                {
                    "min_fps": float(fps_range.minFrameRate()),
                    "max_fps": float(fps_range.maxFrameRate()),
                }
            )
        formats.append(
            {
                "width": int(dimensions.width),
                "height": int(dimensions.height),
                "fps_ranges": ranges,
            }
        )
    return formats


def _fps_range_supports(fps_range: Any, fps: int | float) -> bool:
    return fps_range.minFrameRate() - FPS_MATCH_ABS_TOL <= fps <= fps_range.maxFrameRate() + FPS_MATCH_ABS_TOL


def _frame_duration_for_fps_range(fps_range: Any, fps: int | float, core_media: Any) -> Any:
    min_fps = float(fps_range.minFrameRate())
    max_fps = float(fps_range.maxFrameRate())

    if abs(fps - max_fps) <= FPS_MATCH_ABS_TOL:
        return fps_range.minFrameDuration()
    if abs(fps - min_fps) <= FPS_MATCH_ABS_TOL:
        return fps_range.maxFrameDuration()

    return core_media.CMTimeMake(1, int(fps))


def _format_summary(device: Any, core_media: Any) -> str:
    formats = _device_formats(device, core_media)
    if not formats:
        return "none"

    summary = []
    for av_format in formats:
        fps_ranges = ", ".join(
            f"{fps_range['min_fps']:.3f}-{fps_range['max_fps']:.3f}fps"
            for fps_range in av_format["fps_ranges"]
        )
        summary.append(f"{av_format['width']}x{av_format['height']} [{fps_ranges}]")
    return "; ".join(summary)


def _set_resolution_session_preset_if_supported(
    session: Any,
    avfoundation: Any,
    capture_width: int | None,
    capture_height: int | None,
) -> None:
    if capture_width != 1280 or capture_height != 720:
        logger.info("Using default AVFoundation session preset for capture size %sx%s.", capture_width, capture_height)
        return

    preset = getattr(avfoundation, "AVCaptureSessionPreset1280x720", None)
    if preset is None:
        logger.info("AVCaptureSessionPreset1280x720 is not available; using default session preset.")
        return

    if session.canSetSessionPreset_(preset):
        session.setSessionPreset_(preset)
        logger.info("Using AVFoundation session preset AVCaptureSessionPreset1280x720.")
    else:
        logger.info("AVCaptureSessionPreset1280x720 is not supported; using default session preset.")


def _make_video_settings(
    quartz: Any,
    capture_width: int | None,
    capture_height: int | None,
) -> dict[Any, Any]:
    settings: dict[Any, Any] = {
        quartz.kCVPixelBufferPixelFormatTypeKey: quartz.kCVPixelFormatType_32BGRA,
    }
    if capture_width is not None and capture_height is not None:
        settings[quartz.kCVPixelBufferWidthKey] = int(capture_width)
        settings[quartz.kCVPixelBufferHeightKey] = int(capture_height)
    return settings


def _resolve_avfoundation_device(identifier: str) -> Any:
    avfoundation, _core_media, _quartz, _objc, _ns_object = _import_avfoundation_stack()
    devices = list(avfoundation.AVCaptureDevice.devicesWithMediaType_(avfoundation.AVMediaTypeVideo))

    snapshot = ", ".join(
        f"{index}: {device.localizedName()} ({device.uniqueID()})" for index, device in enumerate(devices)
    )
    logger.info("AVFoundation devices: %s", snapshot or "none")

    unique_id_matches = [device for device in devices if str(device.uniqueID()) == identifier]
    if len(unique_id_matches) == 1:
        return unique_id_matches[0]

    name_matches = [device for device in devices if str(device.localizedName()) == identifier]
    if len(name_matches) == 1:
        return name_matches[0]

    if len(name_matches) > 1:
        matches = ", ".join(
            f"unique_id='{device.uniqueID()}'" for device in name_matches
        )
        raise ConnectionError(
            f"AVFoundation video device name '{identifier}' is ambiguous; matches: {matches}. "
            "Configure device_id with the device unique_id instead."
        )

    available = ", ".join(f"{device.localizedName()} ({device.uniqueID()})" for device in devices) or "none"
    raise ConnectionError(
        f"AVFoundation video device '{identifier}' was not found by name or unique_id. "
        f"Available devices: {available}."
    )


def _make_dispatch_queue(label: str) -> Any:
    _avfoundation, _core_media, _quartz, objc, _ns_object = _import_avfoundation_stack()

    libdispatch = ctypes.CDLL("/usr/lib/system/libdispatch.dylib")
    libdispatch.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    libdispatch.dispatch_queue_create.restype = ctypes.c_void_p

    queue_ptr = libdispatch.dispatch_queue_create(label.encode("utf-8"), None)
    if not queue_ptr:
        raise RuntimeError("Failed to create AVFoundation dispatch queue.")
    return objc.objc_object(c_void_p=queue_ptr)


def _base_address_as_buffer(base_address: Any, byte_count: int) -> memoryview | None:
    if base_address is None:
        return None
    if hasattr(base_address, "as_buffer"):
        return base_address.as_buffer(byte_count)
    raise TypeError(
        f"Unsupported CVPixelBuffer base address type: {type(base_address).__name__}. "
        "Expected a PyObjC object with as_buffer()."
    )


def _get_sample_buffer_delegate_class() -> type[Any]:
    global _SAMPLE_BUFFER_DELEGATE_CLASS
    if _SAMPLE_BUFFER_DELEGATE_CLASS is not None:
        return _SAMPLE_BUFFER_DELEGATE_CLASS

    _avfoundation, core_media, quartz, objc, ns_object = _import_avfoundation_stack()

    class AVFoundationSampleBufferDelegate(ns_object):
        def initWithCamera_(self, owner):  # noqa: N802
            self = objc.super(AVFoundationSampleBufferDelegate, self).init()
            if self is None:
                return None
            self.owner = owner
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(  # noqa: N802
            self, _output, sample_buffer, _connection
        ):
            image_buffer = core_media.CMSampleBufferGetImageBuffer(sample_buffer)
            if image_buffer is None:
                return

            quartz.CVPixelBufferLockBaseAddress(image_buffer, quartz.kCVPixelBufferLock_ReadOnly)
            try:
                width = int(quartz.CVPixelBufferGetWidth(image_buffer))
                height = int(quartz.CVPixelBufferGetHeight(image_buffer))
                bytes_per_row = int(quartz.CVPixelBufferGetBytesPerRow(image_buffer))
                base_address = quartz.CVPixelBufferGetBaseAddress(image_buffer)

                try:
                    buffer = _base_address_as_buffer(base_address, bytes_per_row * height)
                except TypeError as e:
                    logger.warning("Error reading CVPixelBuffer base address for %s: %s", self.owner._context(), e)
                    return
                if buffer is None:
                    logger.warning("CVPixelBuffer base address is None for %s.", self.owner._context())
                    return

                bgra = np.frombuffer(buffer, dtype=np.uint8).reshape((height, bytes_per_row // 4, 4))
                frame = bgra[:, :width, :3].copy()
                self.owner._receive_raw_frame(frame)
            finally:
                quartz.CVPixelBufferUnlockBaseAddress(image_buffer, quartz.kCVPixelBufferLock_ReadOnly)

    _SAMPLE_BUFFER_DELEGATE_CLASS = AVFoundationSampleBufferDelegate
    return AVFoundationSampleBufferDelegate


def _make_sample_buffer_delegate(camera: "AVFoundationCamera") -> Any:
    delegate_class = _get_sample_buffer_delegate_class()
    return delegate_class.alloc().initWithCamera_(camera)


class AVFoundationCamera(Camera):
    """Native macOS camera backed by AVFoundation sample-buffer capture."""

    def __init__(self, config: AVFoundationCameraConfig):
        super().__init__(config)

        self.config = config
        self.device_id = config.device_id
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in (0, 2):
                self.capture_width, self.capture_height = self.height, self.width
        else:
            self.capture_width = None
            self.capture_height = None

        self.device: Any | None = None
        self.session: Any | None = None
        self.input: Any | None = None
        self.output: Any | None = None
        self.queue: Any | None = None
        self.delegate: Any | None = None

        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self._is_connected = False
        self._resolved_device_name: str | None = None
        self._resolved_unique_id: str | None = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.device_id})"

    def _context(self) -> str:
        context = str(self)
        if self.camera_key is not None:
            context += f" camera_key='{self.camera_key}'"
        if self._resolved_device_name is not None:
            context += f" resolved_avfoundation_name='{self._resolved_device_name}'"
        if self._resolved_unique_id is not None:
            context += f" unique_id='{self._resolved_unique_id}'"
        return context

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        found_cameras_info = []
        for device in _list_avfoundation_video_devices():
            default_profile = {}
            if device["formats"]:
                first_format = device["formats"][0]
                default_profile = {
                    "width": first_format["width"],
                    "height": first_format["height"],
                    "fps_ranges": first_format["fps_ranges"],
                }

            found_cameras_info.append(
                {
                    "name": device["name"],
                    "type": "AVFoundation",
                    "id": device["unique_id"],
                    "unique_id": device["unique_id"],
                    "avfoundation_index": device["index"],
                    "default_stream_profile": default_profile,
                }
            )

        return found_cameras_info

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        avfoundation, core_media, quartz, _objc, _ns_object = _import_avfoundation_stack()

        self.device = _resolve_avfoundation_device(self.device_id)
        self._resolved_device_name = str(self.device.localizedName())
        self._resolved_unique_id = str(self.device.uniqueID())
        logger.info(
            "%sResolved AVFoundation device_id='%s' to '%s' unique_id='%s'.",
            f"Camera '{self.camera_key}': " if self.camera_key is not None else "",
            self.device_id,
            self._resolved_device_name,
            self._resolved_unique_id,
        )

        self._configure_capture_profile(core_media)

        self.session = avfoundation.AVCaptureSession.alloc().init()
        self.session.beginConfiguration()

        self.input, error = avfoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(self.device, None)
        if self.input is None:
            self.session.commitConfiguration()
            raise ConnectionError(f"{self._context()} failed to create AVFoundation input: {error}")

        if not self.session.canAddInput_(self.input):
            self.session.commitConfiguration()
            raise ConnectionError(f"{self._context()} cannot add AVFoundation input to capture session.")
        self.session.addInput_(self.input)
        _set_resolution_session_preset_if_supported(
            self.session,
            avfoundation,
            self.capture_width,
            self.capture_height,
        )

        self.output = avfoundation.AVCaptureVideoDataOutput.alloc().init()
        self.output.setAlwaysDiscardsLateVideoFrames_(True)
        video_settings = _make_video_settings(quartz, self.capture_width, self.capture_height)
        logger.info("%sAVFoundation video output settings: %s.", self._context(), video_settings)
        self.output.setVideoSettings_(video_settings)
        self.delegate = _make_sample_buffer_delegate(self)
        self.queue = _make_dispatch_queue(f"lerobot.avfoundation.{self.camera_key or self.device_id}")
        self.output.setSampleBufferDelegate_queue_(self.delegate, self.queue)

        if not self.session.canAddOutput_(self.output):
            self.session.commitConfiguration()
            raise ConnectionError(f"{self._context()} cannot add AVFoundation output to capture session.")
        self.session.addOutput_(self.output)
        self.session.commitConfiguration()
        self.session.startRunning()
        self._is_connected = True

        if warmup and self.warmup_s > 0:
            self.async_read(timeout_ms=self.warmup_s * 1000)

        logger.info("%s connected.", self._context())

    def _configure_capture_profile(self, core_media: Any) -> None:
        if self.device is None:
            raise DeviceNotConnectedError(f"{self} device is not initialized")

        if self.capture_width is None or self.capture_height is None or self.fps is None:
            return

        selected_format = None
        selected_fps_range = None
        for av_format in self.device.formats():
            dimensions = core_media.CMVideoFormatDescriptionGetDimensions(av_format.formatDescription())
            if int(dimensions.width) != self.capture_width or int(dimensions.height) != self.capture_height:
                continue
            for fps_range in av_format.videoSupportedFrameRateRanges():
                if _fps_range_supports(fps_range, self.fps):
                    selected_format = av_format
                    selected_fps_range = fps_range
                    break
            if selected_format is not None:
                break

        if selected_format is None or selected_fps_range is None:
            raise RuntimeError(
                f"{self._context()} could not find AVFoundation format "
                f"{self.capture_width}x{self.capture_height}@{self.fps}. "
                f"Available formats: {_format_summary(self.device, core_media)}."
            )

        success, error = self.device.lockForConfiguration_(None)
        if not success:
            raise RuntimeError(f"{self._context()} failed to lock AVFoundation device: {error}")

        try:
            self.device.setActiveFormat_(selected_format)
            frame_duration = _frame_duration_for_fps_range(selected_fps_range, self.fps, core_media)
            self.device.setActiveVideoMinFrameDuration_(frame_duration)
            self.device.setActiveVideoMaxFrameDuration_(frame_duration)
        finally:
            self.device.unlockForConfiguration()

    def _receive_raw_frame(self, raw_frame: NDArray[Any]) -> None:
        try:
            processed_frame = self._postprocess_image(raw_frame)
        except Exception as e:
            logger.warning("Error processing frame for %s: %s", self._context(), e)
            return

        with self.frame_lock:
            self.latest_frame = processed_frame
            self.latest_timestamp = time.perf_counter()
        self.new_frame_event.set()

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        if color_mode is not None:
            logger.warning(
                "%s read() color_mode parameter is deprecated and will be removed in future versions.",
                self,
            )

        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        if self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        h, w, c = image.shape
        if self.capture_width is None or self.capture_height is None:
            self.capture_width = w
            self.capture_height = h
            if self.rotation in (0, 2):
                self.width, self.height = h, w
            else:
                self.width, self.height = w, h

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self._context()} frame width={w} or height={h} do not match configured "
                f"width={self.capture_width} or height={self.capture_height}."
            )

        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels.")

        processed_image = image
        if self.color_mode == ColorMode.RGB:
            processed_image = image[:, :, ::-1].copy()

        if self.rotation is not None:
            processed_image = _rotate_image(processed_image, self.rotation)

        return processed_image

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.output is not None:
            self.output.setSampleBufferDelegate_queue_(None, None)
        if self.session is not None and self.session.isRunning():
            self.session.stopRunning()

        self._is_connected = False
        self.session = None
        self.input = None
        self.output = None
        self.delegate = None
        self.queue = None
        self.device = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info("%s disconnected.", self)


def _rotate_image(image: NDArray[Any], rotation: int) -> NDArray[Any]:
    # Keep this backend independent from cv2 at capture time; values match get_cv2_rotation output.
    if rotation == 0:
        return np.rot90(image, k=3).copy()
    if rotation == 1:
        return np.rot90(image, k=2).copy()
    if rotation == 2:
        return np.rot90(image, k=1).copy()
    return image
