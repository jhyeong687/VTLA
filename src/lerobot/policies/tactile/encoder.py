#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Reusable tactile encoders.

These modules consume a normalized tactile map and produce token embeddings that
can be attached to policies such as ACT, Diffusion, or VLA-style token models.
Expected input shapes are (H, W), (B, H, W), (C, H, W), (B, C, H, W), or (B, C, D, H, W).
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


def _parse_input_shape(input_shape: tuple[int, ...]) -> tuple[int, tuple[int, int]]:
    if len(input_shape) == 2:
        return 1, (input_shape[0], input_shape[1])
    if len(input_shape) == 3:
        # 💡 현재 데이터셋의 (2, 9, 9) 형상을 정확히 채널=2, 공간=(9, 9)로 매핑합니다.
        return input_shape[0], (input_shape[1], input_shape[2])
    if len(input_shape) == 4:
        return input_shape[0], (input_shape[2], input_shape[3])
    raise ValueError(f"Expected tactile input_shape (H, W), (C, H, W) or (C, D, H, W), got {input_shape}.")


def _as_bchw(x: Tensor, input_channels: int) -> Tensor:
    if x.dim() == 2 and input_channels == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3 and input_channels == 1:
        return x.unsqueeze(1)
    if x.dim() == 3 and x.shape[0] == input_channels:
        return x.unsqueeze(0)
    if x.dim() == 4 and x.shape[1] == input_channels:
        return x
    raise ValueError(
        "Expected tactile tensor shape (H, W), (B, H, W), (C, H, W), or (B, C, H, W), "
        f"got {tuple(x.shape)}."
    )


class TactileCNN(nn.Module):
    """CNN tactile backbone that outputs one feature vector per sample."""

    def __init__(self, input_shape: tuple[int, ...] = (2, 9, 9), feature_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.input_channels, _ = _parse_input_shape(input_shape)

        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        conv_output_dim = 128

        self.fc1 = nn.Linear(conv_output_dim, 512)
        # 💡 지적 반영: 고정된 256이 아니라 상위 파이프라인이 요구하는 feature_dim(1024 등)에 동적 연동
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x: Tensor) -> Tensor:
        # 데이터 정밀도 정렬
        x = _as_bchw(x, self.input_channels).float()

        # 인풋 해상도가 너무 작을 경우 레이어 오버플로우 방지를 위한 어댑티브 풀링 처리
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.global_pool(x)

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TactileAttentionCNN(nn.Module):
    """CNN tactile backbone with spatial attention."""

    def __init__(self, input_shape: tuple[int, ...] = (2, 9, 9), feature_dim: int = 1024, dropout: float = 0.4):
        super().__init__()
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        self.input_channels, _ = _parse_input_shape(input_shape)

        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim), # 💡 동적 차원 연동
        )

    def forward(self, x: Tensor) -> Tensor:
        x = _as_bchw(x, self.input_channels).float()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x * self.attention(x)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(x.flatten(1))


class DualPaxini3DCNN(nn.Module):
    """6월 데이터셋 고도화(시계열/멀티채널) 가동 시 스위칭할 확장 선행 구현 백본"""
    def __init__(self, input_shape: tuple[int, ...] = (3, 2, 9, 9), feature_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.input_channels = input_shape[0]
        self.feature_dim = feature_dim

        self.conv3d_1 = nn.Conv3d(self.input_channels, 32, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv3d_2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3d_3 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)

        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dropout = nn.Dropout(dropout)

        conv_output_dim = 128 * 1 * 1 * 1

        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, feature_dim) # 💡 동적 차원 연동
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()

        # 🚨 [임시 가상 방어 코드] 만약 데이터셋 고도화 전이라 4차원 [B, C, H, W]로 들어온다면
        # 3D Conv가 터지지 않게 가상 시간축(Depth=2)을 강제로 복사 확장하여 에러를 임시 방어합니다.
        if x.dim() == 4:
            # x shape: [B, 2, 9, 9] -> 가상 복사 후 [B, 2, 2, 9, 9]로 변환
            x = x.unsqueeze(2).repeat(1, 1, 2, 1, 1)

        x = F.relu(self.bn1(self.conv3d_1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3d_2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3d_3(x)))
        x = self.pool(x)

        x = x.flatten(1)
        return self.fc(x)


class TactileTokenEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: str,
        input_shape: tuple[int, ...],
        feature_dim: int,
        n_tokens: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.feature_dim = feature_dim

        if encoder_type == "cnn":
            self.backbone = TactileCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "attention":
            self.backbone = TactileAttentionCNN(input_shape, feature_dim, dropout)
        elif encoder_type == "dual_paxini_3d":
            self.backbone = DualPaxini3DCNN(input_shape, feature_dim, dropout)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Choose 'cnn', 'attention', or 'dual_paxini_3d'.")

        self.token_proj = nn.Linear(feature_dim, n_tokens * feature_dim) if n_tokens > 1 else None

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone(x)
        if self.n_tokens == 1:
            return feat.unsqueeze(1)

        feat = self.token_proj(feat)
        return feat.view(feat.size(0), self.n_tokens, self.feature_dim)
