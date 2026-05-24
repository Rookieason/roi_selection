from dataclasses import dataclass
import json
from typing import Optional

import numpy as np


@dataclass
class RFGeometryConfig:
    tx_positions_xyz: np.ndarray
    rx_positions_xyz: np.ndarray
    tx_orientations: Optional[np.ndarray]
    rx_orientations: Optional[np.ndarray]
    center_frequency_hz: float
    speed_of_light_mps: float = 299792458.0
    coordinate_frame: str = "world"


def load_rf_geometry_config(path: str) -> RFGeometryConfig:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    block = raw.get("rf_geometry", raw)
    tx_positions = np.asarray(block["tx_positions_xyz"], dtype=float)
    rx_positions = np.asarray(block["rx_positions_xyz"], dtype=float)
    tx_orientations = _optional_array(block.get("tx_orientations"))
    rx_orientations = _optional_array(block.get("rx_orientations"))

    config = RFGeometryConfig(
        tx_positions_xyz=tx_positions,
        rx_positions_xyz=rx_positions,
        tx_orientations=tx_orientations,
        rx_orientations=rx_orientations,
        center_frequency_hz=float(block["center_frequency_hz"]),
        speed_of_light_mps=float(block.get("speed_of_light_mps", 299792458.0)),
        coordinate_frame=str(block.get("coordinate_frame", "world")),
    )
    validate_rf_geometry_config(config)
    return config


def validate_rf_geometry_config(config: RFGeometryConfig) -> None:
    _validate_positions("tx_positions_xyz", config.tx_positions_xyz)
    _validate_positions("rx_positions_xyz", config.rx_positions_xyz)
    if config.center_frequency_hz <= 0:
        raise ValueError("center_frequency_hz must be positive")
    if config.speed_of_light_mps <= 0:
        raise ValueError("speed_of_light_mps must be positive")
    if config.coordinate_frame != "world":
        raise ValueError("RDITH currently expects RF geometry in the world coordinate frame")
    _validate_orientations("tx_orientations", config.tx_orientations, config.tx_positions_xyz.shape[0])
    _validate_orientations("rx_orientations", config.rx_orientations, config.rx_positions_xyz.shape[0])


def _optional_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def _validate_positions(name: str, value: np.ndarray) -> None:
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3)")


def _validate_orientations(name: str, value: Optional[np.ndarray], expected_count: int) -> None:
    if value is None:
        return
    if value.shape != (expected_count, 3, 3):
        raise ValueError(f"{name} must have shape ({expected_count}, 3, 3)")
