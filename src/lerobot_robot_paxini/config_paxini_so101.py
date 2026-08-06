from __future__ import annotations

from dataclasses import dataclass

from lerobot.robots import RobotConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

from .paxini_reader import TaxelValueMode


@RobotConfig.register_subclass("paxini_so101_follower")
@dataclass
class PaxiniSO101FollowerConfig(RobotConfig, SOFollowerConfig):
    tactile_port: str = "/dev/ttyACM0"
    tactile_baudrate: int = 921600
    tactile_timeout_s: float = 0.5
    tactile_taxel_scale: float = 0.1
    tactile_taxel_value_mode: TaxelValueMode = "z"
    tactile_skip_bytes: int = 0
    tactile_async_read: bool = True
    tactile_map_path: str | None = None
    tactile_image_size: int = 64
    tactile_num_taxels: int = 154
    tactile_representation: str = "heatmap"
    tactile_grid_normalize: bool = False
    tactile_mock: bool = False
    tactile_csv_path: str | None = None
    tactile_csv_scale: float = 1.0
    tactile_csv_loop: bool = True
