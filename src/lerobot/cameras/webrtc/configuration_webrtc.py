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

from dataclasses import dataclass, field

from ..configs import CameraConfig, ColorMode

__all__ = ["WebRTCCameraConfig", "ColorMode"]


@CameraConfig.register_subclass("webrtc")
@dataclass
class WebRTCCameraConfig(CameraConfig):
    """Configuration for low-latency WebRTC/WHEP camera streams."""

    whep_url: str
    headers: dict[str, str] = field(default_factory=dict)
    use_link_headers: bool = True
    color_mode: ColorMode = ColorMode.RGB
    warmup_s: int = 1
    signaling_timeout_s: float = 10.0
    codec: str = "h264"
    jitter_buffer_capacity: int | None = None
    log_latency_stats: bool = False

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.codec = self.codec.lower()

        if not self.whep_url:
            raise ValueError("`whep_url` cannot be empty.")

        if self.signaling_timeout_s <= 0:
            raise ValueError(
                f"`signaling_timeout_s` must be positive, but {self.signaling_timeout_s} is provided."
            )

        if self.codec not in {"h264", "default"}:
            raise ValueError(f"`codec` must be 'h264' or 'default', but {self.codec!r} is provided.")

        if self.jitter_buffer_capacity is not None and (
            self.jitter_buffer_capacity <= 0
            or self.jitter_buffer_capacity & (self.jitter_buffer_capacity - 1) != 0
        ):
            raise ValueError(
                "`jitter_buffer_capacity` must be a positive power of two when provided, "
                f"but {self.jitter_buffer_capacity} is provided."
            )
