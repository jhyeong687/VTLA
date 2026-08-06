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

"""Low-latency WebRTC/WHEP camera backend."""

import asyncio
import contextlib
import logging
import queue
import time
from threading import Condition, Event, Lock, Thread
from typing import Any
from urllib.parse import urljoin

import aiohttp
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpReceiver, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import JitterBuffer
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from .configuration_webrtc import WebRTCCameraConfig

logger = logging.getLogger(__name__)


class LatestOnlyQueue:
    """Async queue that keeps only the newest item."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
        self.drop_count = 0
        self.last_put_s: float | None = None

    async def put(self, item: Any) -> None:
        self._clear()
        self.last_put_s = time.perf_counter()
        self._queue.put_nowait(item)

    async def get(self) -> Any:
        return await self._queue.get()

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def empty(self) -> bool:
        return self._queue.empty()

    def _clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
                self.drop_count += 1
            except asyncio.QueueEmpty:
                return


class LatestOnlyThreadQueue:
    """Thread queue for aiortc decoder input that keeps only the newest encoded frame."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._has_item = False
        self._item: Any | None = None
        self.drop_count = 0

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        del block, timeout
        with self._condition:
            if self._has_item and self._item is not None and item is not None:
                self.drop_count += 1
            self._item = item
            self._has_item = True
            self._condition.notify()

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        with self._condition:
            if not block and not self._has_item:
                raise queue.Empty

            if timeout is None:
                while not self._has_item:
                    self._condition.wait()
            else:
                deadline = time.monotonic() + timeout
                while not self._has_item:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise queue.Empty
                    self._condition.wait(remaining)

            item = self._item
            self._item = None
            self._has_item = False
            return item

    def empty(self) -> bool:
        with self._condition:
            return not self._has_item

    def qsize(self) -> int:
        with self._condition:
            return int(self._has_item)


def _split_link_header(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    in_quote = False
    angle_depth = 0

    for index, char in enumerate(value):
        if char == '"':
            in_quote = not in_quote
        elif not in_quote and char == "<":
            angle_depth += 1
        elif not in_quote and char == ">" and angle_depth > 0:
            angle_depth -= 1
        elif not in_quote and angle_depth == 0 and char == ",":
            part = value[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    part = value[start:].strip()
    if part:
        parts.append(part)
    return parts


def _unquote_http_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def parse_ice_servers_from_link_headers(link_headers: list[str]) -> list[RTCIceServer]:
    ice_servers: list[RTCIceServer] = []

    for header in link_headers:
        for link in _split_link_header(header):
            if not link.startswith("<") or ">" not in link:
                continue

            uri, params_raw = link[1:].split(">", 1)
            params: dict[str, str] = {}
            for item in params_raw.split(";"):
                item = item.strip()
                if not item or "=" not in item:
                    continue
                key, value = item.split("=", 1)
                params[key.strip().lower()] = _unquote_http_value(value)

            rel = params.get("rel", "")
            if "ice-server" not in rel.split():
                continue

            ice_servers.append(
                RTCIceServer(
                    urls=uri,
                    username=params.get("username"),
                    credential=params.get("credential"),
                    credentialType=params.get("credential-type", "password"),
                )
            )

    return ice_servers


def _get_link_header_values(headers: Any) -> list[str]:
    getall = getattr(headers, "getall", None)
    if getall is not None:
        return list(getall("Link", []))

    value = headers.get("Link") if hasattr(headers, "get") else None
    if value is None:
        return []
    return [value]


def _relative_or_absolute_url(base_url: str, maybe_url: str | None) -> str | None:
    if not maybe_url:
        return None
    return urljoin(base_url, maybe_url)


class WebRTCCamera(Camera):
    """Receive latest frames directly from a WebRTC/WHEP source."""

    def __init__(self, config: WebRTCCameraConfig):
        super().__init__(config)

        self.config = config
        self.whep_url = config.whep_url
        self.headers = config.headers
        self.use_link_headers = config.use_link_headers
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.signaling_timeout_s = config.signaling_timeout_s
        self.codec = config.codec
        self.jitter_buffer_capacity = config.jitter_buffer_capacity
        self.log_latency_stats = config.log_latency_stats

        self.thread: Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: Event | None = None
        self.started_event: Event = Event()

        self.pc: RTCPeerConnection | None = None
        self.track: Any | None = None
        self.receive_task: asyncio.Task[None] | None = None
        self.decoder_queue: LatestOnlyThreadQueue | None = None
        self.decoded_queue: LatestOnlyQueue | None = None
        self.resource_url: str | None = None
        self.connected = False
        self.start_error: BaseException | None = None
        self.read_error: BaseException | None = None

        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        self._latency_log_interval_s = 5.0
        self._last_latency_log_s = time.perf_counter()
        self._frames_since_latency_log = 0
        self._latest_receiver_age_ms: float | None = None
        self._latest_to_ndarray_ms: float | None = None

    def __str__(self) -> str:
        return f"WebRTCCamera({self.whep_url})"

    @property
    def is_connected(self) -> bool:
        return self.connected and self.loop is not None and self.thread is not None and self.thread.is_alive()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self.stop_event = Event()
        self.started_event.clear()
        self.start_error = None
        self.read_error = None

        self.thread = Thread(target=self._run_loop, daemon=True, name=f"{self}_loop")
        self.thread.start()

        if not self.started_event.wait(timeout=self.signaling_timeout_s):
            self.disconnect()
            raise TimeoutError(f"{self} timed out connecting after {self.signaling_timeout_s}s.")

        if self.start_error is not None:
            error = self.start_error
            self.disconnect()
            raise RuntimeError(f"Failed to connect to {self}: {error}") from error

        try:
            if warmup and self.warmup_s > 0:
                self.async_read(timeout_ms=self.warmup_s * 1000)
        except Exception:
            self.disconnect()
            raise

        logger.info("%s connected.", self)

    def _run_loop(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._connect_async())
            self.connected = True
            self.started_event.set()
            self.loop.run_forever()
        except BaseException as e:
            self.start_error = e
            self.started_event.set()
        finally:
            self.loop.run_until_complete(self._disconnect_async())
            self.loop.close()
            self.loop = None

    async def _connect_async(self) -> None:
        ice_servers: list[RTCIceServer] = []
        if self.use_link_headers:
            ice_servers = await self._get_ice_servers_from_whep_options()

        self.pc = RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))
        track_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

        @self.pc.on("track")
        def on_track(track):
            if track.kind != "video":
                track.stop()
                return
            if hasattr(track, "_queue"):
                self.decoded_queue = LatestOnlyQueue()
                track._queue = self.decoded_queue
            if not track_future.done():
                track_future.set_result(track)

        transceiver = self.pc.addTransceiver("video", direction="recvonly")
        self._configure_video_transceiver(transceiver)

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)

        answer_sdp, resource_url = await self._post_whep_offer(self.pc.localDescription.sdp)
        self.resource_url = resource_url
        await self.pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        self.track = await asyncio.wait_for(track_future, timeout=self.signaling_timeout_s)
        self.receive_task = asyncio.create_task(self._receive_frames_loop())

    def _configure_video_transceiver(self, transceiver: Any) -> None:
        if self.codec == "h264":
            h264_codecs = [
                codec
                for codec in RTCRtpReceiver.getCapabilities("video").codecs
                if codec.mimeType.lower() == "video/h264"
            ]
            if not h264_codecs:
                raise RuntimeError(f"{self} could not find an aiortc H264 decoder capability.")
            transceiver.setCodecPreferences(h264_codecs)

        self._patch_receiver_latency_buffers(transceiver.receiver)

    def _patch_receiver_latency_buffers(self, receiver: Any) -> None:
        decoder_queue_attr = "_RTCRtpReceiver__decoder_queue"
        if hasattr(receiver, decoder_queue_attr):
            self.decoder_queue = LatestOnlyThreadQueue()
            setattr(receiver, decoder_queue_attr, self.decoder_queue)
        else:
            logger.warning(
                "%s could not patch aiortc decoder queue; stale encoded frames may add latency.", self
            )

        if self.jitter_buffer_capacity is None:
            return

        jitter_buffer_attr = "_RTCRtpReceiver__jitter_buffer"
        if not hasattr(receiver, jitter_buffer_attr):
            raise RuntimeError(f"{self} could not patch aiortc jitter buffer for low-latency mode.")

        setattr(
            receiver,
            jitter_buffer_attr,
            JitterBuffer(capacity=self.jitter_buffer_capacity, is_video=True),
        )

    async def _get_ice_servers_from_whep_options(self) -> list[RTCIceServer]:
        timeout = aiohttp.ClientTimeout(total=self.signaling_timeout_s)
        try:
            async with (
                aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session,
                session.options(self.whep_url) as response,
            ):
                if response.status >= 400:
                    logger.debug("%s WHEP OPTIONS returned HTTP %s.", self, response.status)
                    return []
                return parse_ice_servers_from_link_headers(_get_link_header_values(response.headers))
        except Exception as e:
            logger.debug("%s failed to fetch WHEP Link headers: %s", self, e)
            return []

    async def _post_whep_offer(self, offer_sdp: str) -> tuple[str, str | None]:
        timeout = aiohttp.ClientTimeout(total=self.signaling_timeout_s)
        headers = {
            **self.headers,
            "Content-Type": "application/sdp",
            "Accept": "application/sdp",
        }
        async with (
            aiohttp.ClientSession(timeout=timeout, headers=headers) as session,
            session.post(self.whep_url, data=offer_sdp) as response,
        ):
            answer_sdp = await response.text()
            if response.status >= 400:
                raise ConnectionError(f"WHEP offer failed with HTTP {response.status}: {answer_sdp}")

            resource_url = _relative_or_absolute_url(self.whep_url, response.headers.get("Location"))
            return answer_sdp, resource_url

    async def _receive_frames_loop(self) -> None:
        if self.track is None:
            raise RuntimeError(f"{self} WebRTC track is not initialized.")

        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                frame = await self.track.recv()
                frame = self._drain_to_newest_decoded_frame(frame)
                receive_time_s = time.perf_counter()
                if self.decoded_queue is not None and self.decoded_queue.last_put_s is not None:
                    self._latest_receiver_age_ms = (receive_time_s - self.decoded_queue.last_put_s) * 1e3

                conversion_start_s = time.perf_counter()
                image = self._frame_to_ndarray(frame)
                self._latest_to_ndarray_ms = (time.perf_counter() - conversion_start_s) * 1e3
            except MediaStreamError:
                break
            except Exception as e:
                self.read_error = e
                logger.warning("Error reading frame from %s: %s", self, e)
                await asyncio.sleep(0)
                continue

            with self.frame_lock:
                self.latest_frame = image
                self.latest_timestamp = time.perf_counter()
            self.new_frame_event.set()
            self._frames_since_latency_log += 1
            self._maybe_log_latency_stats()

    def _drain_to_newest_decoded_frame(self, frame: Any) -> Any:
        queue = getattr(self.track, "_queue", None)
        get_nowait = getattr(queue, "get_nowait", None)
        if get_nowait is None:
            return frame

        while True:
            try:
                next_frame = get_nowait()
            except asyncio.QueueEmpty:
                return frame

            if next_frame is None:
                raise MediaStreamError
            frame = next_frame

    def _frame_to_ndarray(self, frame: Any) -> NDArray[Any]:
        pixel_format = "rgb24" if self.color_mode == ColorMode.RGB else "bgr24"
        image = frame.to_ndarray(format=pixel_format)

        h, w = image.shape[:2]
        if self.width is not None and self.height is not None and (w != self.width or h != self.height):
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured "
                f"width={self.width} or height={self.height}."
            )

        if self.width is None or self.height is None:
            self.width = w
            self.height = h

        return image

    def _maybe_log_latency_stats(self) -> None:
        if not self.log_latency_stats:
            return

        now_s = time.perf_counter()
        elapsed_s = now_s - self._last_latency_log_s
        if elapsed_s < self._latency_log_interval_s:
            return

        decoded_fps = self._frames_since_latency_log / elapsed_s if elapsed_s > 0 else 0.0
        encoded_drops = self.decoder_queue.drop_count if self.decoder_queue is not None else 0
        decoded_drops = self.decoded_queue.drop_count if self.decoded_queue is not None else 0
        receiver_age_ms = self._latest_receiver_age_ms if self._latest_receiver_age_ms is not None else -1.0
        to_ndarray_ms = self._latest_to_ndarray_ms if self._latest_to_ndarray_ms is not None else -1.0

        logger.info(
            "%s latency stats: decoded_fps=%.1f receiver_age_ms=%.1f "
            "to_ndarray_ms=%.1f encoded_queue_drops=%d decoded_queue_drops=%d.",
            self,
            decoded_fps,
            receiver_age_ms,
            to_ndarray_ms,
            encoded_drops,
            decoded_drops,
        )

        self._last_latency_log_s = now_s
        self._frames_since_latency_log = 0

    def _submit_async(self, coro) -> Any:
        if self.loop is None:
            raise DeviceNotConnectedError(f"{self} loop is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

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
            detail = f" Last read error: {self.read_error}" if self.read_error is not None else ""
            raise TimeoutError(f"{self} async_read timeout after {timeout_ms}ms.{detail}")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self} no frame available")

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

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.stop_event is not None:
            self.stop_event.set()

        loop = self.loop
        if loop is not None and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._disconnect_async(), loop)
            future.result(timeout=self.signaling_timeout_s)
            loop.call_soon_threadsafe(loop.stop)

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=self.signaling_timeout_s)

        self.thread = None
        self.stop_event = None
        self.connected = False

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    async def _disconnect_async(self) -> None:
        if self.receive_task is not None:
            self.receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.receive_task
            self.receive_task = None

        if self.pc is not None:
            await self.pc.close()
            self.pc = None

        if self.resource_url is not None:
            timeout = aiohttp.ClientTimeout(total=self.signaling_timeout_s)
            try:
                async with aiohttp.ClientSession(timeout=timeout, headers=self.headers) as session:
                    await session.delete(self.resource_url)
            except Exception as e:
                logger.debug("%s failed to DELETE WHEP resource %s: %s", self, self.resource_url, e)
            self.resource_url = None

        self.track = None
        self.connected = False
