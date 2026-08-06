from __future__ import annotations

from functools import cached_property

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.types import RobotObservation
from lerobot.utils.constants import OBS_TACTILE

from .config_paxini_so101 import PaxiniSO101FollowerConfig
from .paxini_reader import CsvPaxiniReader, MockPaxiniReader, PaxiniReader, SerialPaxiniReader
from .tactile_render import (
    TactileHeatmapRenderer,
    load_taxel_map_xlsx,
    make_mock_taxel_map,
    taxels_to_stacked_9x9,
)

PAXINI_TACTILE_KEY = f"{OBS_TACTILE}.primary"


class PaxiniSO101Follower(SOFollower):
    config_class = PaxiniSO101FollowerConfig
    name = "paxini_so101_follower"

    def __init__(self, config: PaxiniSO101FollowerConfig):
        super().__init__(config)
        self.config = config
        if config.tactile_representation not in ("heatmap", "stacked_9x9"):
            raise ValueError(
                "tactile_representation must be one of 'heatmap' or 'stacked_9x9', "
                f"got {config.tactile_representation!r}."
            )

        if config.tactile_map_path is None:
            taxel_map = make_mock_taxel_map(config.tactile_num_taxels)
        else:
            taxel_map = load_taxel_map_xlsx(config.tactile_map_path)

        self.tactile_renderer = TactileHeatmapRenderer(
            taxel_map=taxel_map,
            image_size=config.tactile_image_size,
        )
        self.tactile: PaxiniReader
        if config.tactile_csv_path is not None:
            self.tactile = CsvPaxiniReader(
                config.tactile_csv_path,
                num_taxels=taxel_map.num_taxels,
                taxel_value_mode=config.tactile_taxel_value_mode,
                taxel_scale=config.tactile_csv_scale,
                loop=config.tactile_csv_loop,
            )
        elif config.tactile_mock:
            self.tactile = MockPaxiniReader(num_taxels=taxel_map.num_taxels)
        else:
            self.tactile = SerialPaxiniReader(
                config.tactile_port,
                config.tactile_baudrate,
                num_taxels=taxel_map.num_taxels,
                timeout_s=config.tactile_timeout_s,
                taxel_scale=config.tactile_taxel_scale,
                taxel_value_mode=config.tactile_taxel_value_mode,
                skip_bytes=config.tactile_skip_bytes,
            )

    @cached_property
    def observation_features(self) -> dict[str, type | tuple | PolicyFeature]:  # type: ignore[override]
        tactile_shape = (
            (2, 9, 9)
            if self.config.tactile_representation == "stacked_9x9"
            else (self.config.tactile_image_size, self.config.tactile_image_size)
        )
        return {
            **self._motors_ft,
            **self._cameras_ft,
            PAXINI_TACTILE_KEY: PolicyFeature(
                type=FeatureType.TACTILE,
                shape=tactile_shape,
            ),
        }

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate=calibrate)
        self.tactile.connect()

    def get_observation(self) -> RobotObservation:
        obs = super().get_observation()
        sample = self.tactile.read_latest()
        if self.config.tactile_representation == "stacked_9x9":
            obs[PAXINI_TACTILE_KEY] = taxels_to_stacked_9x9(
                sample.taxels,
                normalize=self.config.tactile_grid_normalize,
            )
        else:
            obs[PAXINI_TACTILE_KEY] = self.tactile_renderer.render(sample.taxels)
        return obs

    def disconnect(self) -> None:
        try:
            self.tactile.disconnect()
        finally:
            super().disconnect()
