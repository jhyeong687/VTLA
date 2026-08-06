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

import asyncio

import numpy as np
import pytest

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.webrtc import WebRTCCamera, WebRTCCameraConfig, camera_webrtc


class FakeFrame:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.formats: list[str] = []

    def to_ndarray(self, format: str):
        self.formats.append(format)
        return self.image


class FakeTrack:
    def __init__(self, queue):
        self._queue = queue


class FakeReceiver:
    def __init__(self):
        self._RTCRtpReceiver__decoder_queue = object()
        self._RTCRtpReceiver__jitter_buffer = object()


class FakeTransceiver:
    def __init__(self):
        self.receiver = FakeReceiver()
        self.codec_preferences = None

    def setCodecPreferences(self, codecs):  # noqa: N802
        self.codec_preferences = codecs


class FakeHTTPResponse:
    def __init__(self, status: int, text: str = "", headers=None):
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass

    async def text(self):
        return self._text


class FakeHTTPSession:
    captured: dict[str, object] = {}
    post_response = FakeHTTPResponse(
        status=201,
        text="answer-sdp",
        headers={"Location": "/cam/session"},
    )
    options_response = FakeHTTPResponse(
        status=204,
        headers={"Link": '<stun:stun.example.com>; rel="ice-server"'},
    )

    def __init__(self, timeout=None, headers=None):
        self.captured["timeout"] = timeout
        self.captured["headers"] = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        pass

    def post(self, url, data=None):
        self.captured["post_url"] = url
        self.captured["post_data"] = data
        return self.post_response

    def options(self, url):
        self.captured["options_url"] = url
        return self.options_response

    async def delete(self, url):
        self.captured["delete_url"] = url


def test_config_requires_whep_url():
    with pytest.raises(ValueError):
        WebRTCCameraConfig(whep_url="")


def test_config_requires_positive_signaling_timeout():
    with pytest.raises(ValueError):
        WebRTCCameraConfig(whep_url="http://example.com/cam/whep", signaling_timeout_s=0)


def test_config_requires_valid_codec():
    with pytest.raises(ValueError, match="codec"):
        WebRTCCameraConfig(whep_url="http://example.com/cam/whep", codec="vp8")


def test_config_requires_power_of_two_jitter_buffer_capacity():
    with pytest.raises(ValueError, match="jitter_buffer_capacity"):
        WebRTCCameraConfig(whep_url="http://example.com/cam/whep", jitter_buffer_capacity=127)


def test_make_cameras_from_configs_creates_webrtc_camera():
    cameras = make_cameras_from_configs(
        {
            "remote": WebRTCCameraConfig(whep_url="http://example.com/cam/whep"),
        }
    )

    assert isinstance(cameras["remote"], WebRTCCamera)


def test_parse_ice_servers_from_link_headers():
    servers = camera_webrtc.parse_ice_servers_from_link_headers(
        [
            '<stun:stun.example.com>; rel="ice-server", '
            '<turn:turn.example.com>; rel="ice-server"; username="u"; credential="p"'
        ]
    )

    assert len(servers) == 2
    assert servers[0].urls == "stun:stun.example.com"
    assert servers[1].urls == "turn:turn.example.com"
    assert servers[1].username == "u"
    assert servers[1].credential == "p"


def test_latest_only_queue_keeps_newest_item():
    async def run():
        queue = camera_webrtc.LatestOnlyQueue()
        await queue.put("old")
        await queue.put("new")
        return await queue.get()

    assert asyncio.run(run()) == "new"


def test_latest_only_thread_queue_keeps_newest_item_and_stop_sentinel():
    queue = camera_webrtc.LatestOnlyThreadQueue()

    queue.put("old")
    queue.put("new")

    assert queue.drop_count == 1
    assert queue.get() == "new"
    assert queue.empty()

    queue.put("frame")
    queue.put(None)

    assert queue.get() is None


def test_configure_video_transceiver_prefers_h264_and_patches_decoder_queue():
    camera = WebRTCCamera(WebRTCCameraConfig(whep_url="http://example.com/cam/whep"))
    transceiver = FakeTransceiver()

    camera._configure_video_transceiver(transceiver)

    assert transceiver.codec_preferences
    assert all(codec.mimeType.lower() == "video/h264" for codec in transceiver.codec_preferences)
    assert isinstance(
        transceiver.receiver._RTCRtpReceiver__decoder_queue,
        camera_webrtc.LatestOnlyThreadQueue,
    )
    assert camera.decoder_queue is transceiver.receiver._RTCRtpReceiver__decoder_queue


def test_configure_video_transceiver_can_patch_jitter_buffer_without_codec_preference():
    camera = WebRTCCamera(
        WebRTCCameraConfig(
            whep_url="http://example.com/cam/whep",
            codec="default",
            jitter_buffer_capacity=16,
        )
    )
    transceiver = FakeTransceiver()

    camera._configure_video_transceiver(transceiver)

    assert transceiver.codec_preferences is None
    assert transceiver.receiver._RTCRtpReceiver__jitter_buffer.capacity == 16


def test_drain_to_newest_decoded_frame_uses_newest_queued_frame():
    async def run():
        queue = camera_webrtc.LatestOnlyQueue()
        await queue.put("queued-old")
        await queue.put("queued-new")
        camera = WebRTCCamera(WebRTCCameraConfig(whep_url="http://example.com/cam/whep"))
        camera.track = FakeTrack(queue)
        return camera._drain_to_newest_decoded_frame("initial")

    assert asyncio.run(run()) == "queued-new"


def test_frame_to_ndarray_uses_configured_color_mode_and_detects_resolution():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    frame = FakeFrame(image)
    camera = WebRTCCamera(
        WebRTCCameraConfig(whep_url="http://example.com/cam/whep", color_mode=ColorMode.BGR)
    )

    result = camera._frame_to_ndarray(frame)

    assert result is image
    assert frame.formats == ["bgr24"]
    assert camera.width == 4
    assert camera.height == 3


def test_frame_to_ndarray_rejects_unexpected_resolution():
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    frame = FakeFrame(image)
    camera = WebRTCCamera(WebRTCCameraConfig(whep_url="http://example.com/cam/whep", width=1280, height=720))

    with pytest.raises(RuntimeError, match="do not match configured"):
        camera._frame_to_ndarray(frame)


def test_whep_offer_post_uses_sdp_headers_and_location(monkeypatch):
    FakeHTTPSession.captured = {}
    monkeypatch.setattr(camera_webrtc.aiohttp, "ClientSession", FakeHTTPSession)
    camera = WebRTCCamera(
        WebRTCCameraConfig(
            whep_url="http://example.com/cam/whep",
            headers={"Authorization": "Bearer token"},
        )
    )

    answer, resource_url = asyncio.run(camera._post_whep_offer("offer-sdp"))

    assert answer == "answer-sdp"
    assert resource_url == "http://example.com/cam/session"
    assert FakeHTTPSession.captured["post_url"] == "http://example.com/cam/whep"
    assert FakeHTTPSession.captured["post_data"] == "offer-sdp"
    assert FakeHTTPSession.captured["headers"] == {
        "Authorization": "Bearer token",
        "Content-Type": "application/sdp",
        "Accept": "application/sdp",
    }


def test_whep_options_link_headers(monkeypatch):
    FakeHTTPSession.captured = {}
    monkeypatch.setattr(camera_webrtc.aiohttp, "ClientSession", FakeHTTPSession)
    camera = WebRTCCamera(WebRTCCameraConfig(whep_url="http://example.com/cam/whep"))

    servers = asyncio.run(camera._get_ice_servers_from_whep_options())

    assert FakeHTTPSession.captured["options_url"] == "http://example.com/cam/whep"
    assert len(servers) == 1
    assert servers[0].urls == "stun:stun.example.com"


def test_disconnect_deletes_whep_resource(monkeypatch):
    FakeHTTPSession.captured = {}
    monkeypatch.setattr(camera_webrtc.aiohttp, "ClientSession", FakeHTTPSession)
    camera = WebRTCCamera(WebRTCCameraConfig(whep_url="http://example.com/cam/whep"))
    camera.resource_url = "http://example.com/cam/session"

    asyncio.run(camera._disconnect_async())

    assert FakeHTTPSession.captured["delete_url"] == "http://example.com/cam/session"
