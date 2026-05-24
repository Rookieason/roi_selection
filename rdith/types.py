from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HeatmapTensor:
    energy: np.ndarray
    time_s: np.ndarray
    doppler_hz: np.ndarray
    tof_s: Optional[np.ndarray]
    aoa_rad: Optional[np.ndarray]


@dataclass
class Pose6DoF:
    timestamp_s: float
    position_xyz: np.ndarray
    rotation_matrix: np.ndarray
    linear_velocity_xyz: np.ndarray
    angular_velocity_xyz: np.ndarray


@dataclass
class CandidateROI:
    roi_id: int
    center_xyz: np.ndarray
    bbox_extent_xyz: np.ndarray
    visibility_score: float
    occlusion_score: float


@dataclass
class ResidualHeatmap:
    residual_energy: np.ndarray
    residual_velocity: np.ndarray
    confidence: np.ndarray


@dataclass
class VoxelRFMap:
    voxel_positions_xyz: np.ndarray
    energy: np.ndarray
    observed_doppler_hz: np.ndarray
    confidence: np.ndarray


@dataclass
class ResidualVoxelMap:
    voxel_positions_xyz: np.ndarray
    residual_energy: np.ndarray
    residual_doppler_hz: np.ndarray
    expected_doppler_hz: np.ndarray
    observed_doppler_hz: np.ndarray
    confidence: np.ndarray


@dataclass
class RFBlob:
    centroid_xyz: np.ndarray
    velocity_xyz: np.ndarray
    energy: float
    doppler_bandwidth: float
    confidence: float
    lifetime_frames: int


@dataclass
class ROIFeatureVector:
    roi_id: int
    feature_names: list[str]
    features: np.ndarray
