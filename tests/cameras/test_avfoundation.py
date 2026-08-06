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

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.cameras.avfoundation import AVFoundationCamera, AVFoundationCameraConfig, camera_avfoundation
from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import make_cameras_from_configs


class FakeDevice:
    def __init__(self, name: str, unique_id: str, formats=None):
        self._name = name
        self._unique_id = unique_id
        self._formats = formats or []
        self.active_format = None
        self.active_min_frame_duration = None
        self.active_max_frame_duration = None

    def localizedName(self):  # noqa: N802
        return self._name

    def uniqueID(self):  # noqa: N802
        return self._unique_id

    def formats(self):
        return self._formats

    def lockForConfiguration_(self, _error):  # noqa: N802
        return True, None

    def unlockForConfiguration(self):  # noqa: N802
        pass

    def setActiveFormat_(self, av_format):  # noqa: N802
        self.active_format = av_format

    def setActiveVideoMinFrameDuration_(self, duration):  # noqa: N802
        self.active_min_frame_duration = duration

    def setActiveVideoMaxFrameDuration_(self, duration):  # noqa: N802
        self.active_max_frame_duration = duration


class FakeFormat:
    def __init__(self, width: int, height: int, min_fps: float, max_fps: float):
        self.width = width
        self.height = height
        self._fps_ranges = [FakeFpsRange(min_fps, max_fps)]

    def formatDescription(self):  # noqa: N802
        return self

    def videoSupportedFrameRateRanges(self):  # noqa: N802
        return self._fps_ranges


class FakeFpsRange:
    def __init__(self, min_fps: float, max_fps: float):
        self._min_fps = min_fps
        self._max_fps = max_fps

    def minFrameRate(self):  # noqa: N802
        return self._min_fps

    def maxFrameRate(self):  # noqa: N802
        return self._max_fps

    def minFrameDuration(self):  # noqa: N802
        return "min_frame_duration", self._max_fps

    def maxFrameDuration(self):  # noqa: N802
        return "max_frame_duration", self._min_fps


class FakeCoreMedia:
    @staticmethod
    def CMVideoFormatDescriptionGetDimensions(description):  # noqa: N802
        return SimpleNamespace(width=description.width, height=description.height)

    @staticmethod
    def CMTimeMake(numerator, denominator):  # noqa: N802
        return numerator, denominator


class FakeSession:
    def __init__(self, supported_presets=None):
        self.supported_presets = set(supported_presets or [])
        self.set_presets = []

    def canSetSessionPreset_(self, preset):  # noqa: N802
        return preset in self.supported_presets

    def setSessionPreset_(self, preset):  # noqa: N802
        self.set_presets.append(preset)


FAKE_QUARTZ = SimpleNamespace(
    kCVPixelBufferPixelFormatTypeKey="pixel_format",
    kCVPixelFormatType_32BGRA="32BGRA",
    kCVPixelBufferWidthKey="width",
    kCVPixelBufferHeightKey="height",
)


class FakeBaseAddress:
    def __init__(self, data: bytes):
        self.data = data
        self.requested_count = None

    def as_buffer(self, count):
        self.requested_count = count
        return memoryview(self.data)[:count]


def _patch_devices(monkeypatch, devices):
    fake_avfoundation = SimpleNamespace(
        AVMediaTypeVideo="video",
        AVCaptureDevice=SimpleNamespace(devicesWithMediaType_=lambda _media_type: devices),
    )
    monkeypatch.setattr(
        camera_avfoundation,
        "_import_avfoundation_stack",
        lambda: (fake_avfoundation, SimpleNamespace(), SimpleNamespace(), SimpleNamespace(), object),
    )


def test_config_requires_complete_stream_profile():
    with pytest.raises(ValueError):
        AVFoundationCameraConfig(device_id="camera-id", width=1280, height=720)


def test_config_requires_device_id():
    with pytest.raises(ValueError):
        AVFoundationCameraConfig(device_id="")


def test_resolve_device_by_unique_id(monkeypatch):
    target = FakeDevice("USB Camera", "unique-1")
    _patch_devices(monkeypatch, [FakeDevice("Other", "unique-0"), target])

    resolved = camera_avfoundation._resolve_avfoundation_device("unique-1")

    assert resolved is target


def test_resolve_device_by_unique_name(monkeypatch):
    target = FakeDevice("USB Camera", "unique-1")
    _patch_devices(monkeypatch, [target])

    resolved = camera_avfoundation._resolve_avfoundation_device("USB Camera")

    assert resolved is target


def test_resolve_device_rejects_ambiguous_name(monkeypatch):
    _patch_devices(monkeypatch, [FakeDevice("USB Camera", "unique-1"), FakeDevice("USB Camera", "unique-2")])

    with pytest.raises(ConnectionError, match="ambiguous"):
        camera_avfoundation._resolve_avfoundation_device("USB Camera")


def test_find_cameras_metadata(monkeypatch):
    _patch_devices(monkeypatch, [FakeDevice("USB Camera", "unique-1")])

    cameras = AVFoundationCamera.find_cameras()

    assert cameras == [
        {
            "name": "USB Camera",
            "type": "AVFoundation",
            "id": "unique-1",
            "unique_id": "unique-1",
            "avfoundation_index": 0,
            "default_stream_profile": {},
        }
    ]


def test_make_cameras_from_configs_creates_avfoundation_camera():
    cameras = make_cameras_from_configs(
        {
            "wrist": AVFoundationCameraConfig(device_id="unique-1"),
        }
    )

    assert isinstance(cameras["wrist"], AVFoundationCamera)
    assert cameras["wrist"].camera_key == "wrist"


def test_receive_raw_frame_buffers_rgb_image():
    camera = AVFoundationCamera(
        AVFoundationCameraConfig(
            device_id="unique-1",
            fps=30,
            width=2,
            height=1,
            color_mode=ColorMode.RGB,
            warmup_s=0,
        )
    )
    camera._is_connected = True

    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    camera._receive_raw_frame(bgr)

    frame = camera.async_read(timeout_ms=1)

    assert frame.tolist() == [[[3, 2, 1], [6, 5, 4]]]


def test_configure_capture_profile_accepts_avfoundation_fractional_fps():
    av_format = FakeFormat(width=1280, height=720, min_fps=30.00003000003, max_fps=30.00003000003)
    device = FakeDevice("USB Camera", "unique-1", formats=[av_format])
    camera = AVFoundationCamera(
        AVFoundationCameraConfig(device_id="unique-1", fps=30, width=1280, height=720)
    )
    camera.device = device

    camera._configure_capture_profile(FakeCoreMedia)

    assert device.active_format is av_format
    assert device.active_min_frame_duration == ("min_frame_duration", 30.00003000003)
    assert device.active_max_frame_duration == ("min_frame_duration", 30.00003000003)


def test_configure_capture_profile_rejects_unsupported_fps_with_available_formats():
    device = FakeDevice(
        "USB Camera",
        "unique-1",
        formats=[FakeFormat(width=1280, height=720, min_fps=30.00003000003, max_fps=30.00003000003)],
    )
    camera = AVFoundationCamera(
        AVFoundationCameraConfig(device_id="unique-1", fps=60, width=1280, height=720)
    )
    camera.device = device

    with pytest.raises(RuntimeError, match=r"Available formats: 1280x720"):
        camera._configure_capture_profile(FakeCoreMedia)


def test_1280x720_session_preset_is_skipped_when_unsupported():
    session = FakeSession()
    avfoundation = SimpleNamespace(AVCaptureSessionPreset1280x720="preset-1280x720")

    camera_avfoundation._set_resolution_session_preset_if_supported(session, avfoundation, 1280, 720)

    assert session.set_presets == []


def test_1280x720_session_preset_is_set_when_supported():
    session = FakeSession(supported_presets={"preset-1280x720"})
    avfoundation = SimpleNamespace(AVCaptureSessionPreset1280x720="preset-1280x720")

    camera_avfoundation._set_resolution_session_preset_if_supported(session, avfoundation, 1280, 720)

    assert session.set_presets == ["preset-1280x720"]


def test_resolution_session_preset_is_skipped_for_other_sizes():
    session = FakeSession(supported_presets={"preset-1280x720"})
    avfoundation = SimpleNamespace(AVCaptureSessionPreset1280x720="preset-1280x720")

    camera_avfoundation._set_resolution_session_preset_if_supported(session, avfoundation, 640, 480)

    assert session.set_presets == []


def test_video_settings_include_configured_capture_size():
    settings = camera_avfoundation._make_video_settings(FAKE_QUARTZ, 1280, 720)

    assert settings == {
        "pixel_format": "32BGRA",
        "width": 1280,
        "height": 720,
    }


def test_video_settings_without_configured_capture_size_only_include_pixel_format():
    settings = camera_avfoundation._make_video_settings(FAKE_QUARTZ, None, None)

    assert settings == {
        "pixel_format": "32BGRA",
    }


def test_base_address_as_buffer_uses_pyobjc_varlist_buffer():
    base_address = FakeBaseAddress(bytes(range(16)))

    buffer = camera_avfoundation._base_address_as_buffer(base_address, 8)

    assert base_address.requested_count == 8
    assert np.frombuffer(buffer, dtype=np.uint8).tolist() == list(range(8))


def test_base_address_as_buffer_allows_none():
    assert camera_avfoundation._base_address_as_buffer(None, 8) is None


def test_base_address_as_buffer_rejects_unsupported_type():
    with pytest.raises(TypeError, match="Unsupported CVPixelBuffer base address type"):
        camera_avfoundation._base_address_as_buffer(object(), 8)
