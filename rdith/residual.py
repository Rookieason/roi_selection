import numpy as np

from .config import RFGeometryConfig
from .doppler_projection import project_velocity_to_bistatic_doppler
from .types import HeatmapTensor, Pose6DoF, ResidualHeatmap
from .types import ResidualVoxelMap, VoxelRFMap


def estimate_expected_velocity_field(
    pose: Pose6DoF,
    voxel_positions_xyz: np.ndarray,
) -> np.ndarray:
    """Estimate rigid-body velocity induced by headset 6DoF motion."""
    positions = np.asarray(voxel_positions_xyz, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("voxel_positions_xyz must have shape (N_voxel, 3)")

    head_position = np.asarray(pose.position_xyz, dtype=float).reshape(3)
    linear_velocity = np.asarray(pose.linear_velocity_xyz, dtype=float).reshape(3)
    angular_velocity = np.asarray(pose.angular_velocity_xyz, dtype=float).reshape(3)
    relative_positions = positions - head_position
    return linear_velocity[None, :] + np.cross(angular_velocity[None, :], relative_positions)


def compute_residual_heatmap(
    heatmap: HeatmapTensor,
    expected_velocity_xyz: np.ndarray,
) -> ResidualHeatmap:
    """Suppress heatmap energy explained by headset motion.

    The current upstream heatmaps expose Doppler bins but not per-bin 3D
    velocity vectors. Until calibrated voxel geometry is available, expected
    3D velocity is reduced to speed magnitude and compared against the Doppler
    axis as a residual proxy.
    """
    energy = np.asarray(heatmap.energy, dtype=float)
    if energy.ndim < 2:
        raise ValueError("heatmap.energy must include time and doppler axes")

    doppler = np.asarray(heatmap.doppler_hz, dtype=float)
    if energy.shape[1] != doppler.size:
        raise ValueError("heatmap.energy axis 1 must match doppler_hz")

    expected = np.asarray(expected_velocity_xyz, dtype=float)
    if expected.ndim == 1:
        expected = expected.reshape(1, -1)
    if expected.ndim != 2 or expected.shape[1] != 3:
        raise ValueError("expected_velocity_xyz must have shape (N, 3)")

    expected_speed = np.linalg.norm(expected, axis=1)
    spatial_shape = energy.shape[2:]
    expected_field = _reshape_expected_speed(expected_speed, spatial_shape)
    observed = _doppler_grid(doppler, energy.ndim, spatial_shape)

    residual_velocity = observed - expected_field
    residual_scale = np.abs(residual_velocity) / (np.abs(observed) + np.abs(expected_field) + 1e-9)
    residual_energy = np.clip(energy, 0.0, None) * residual_scale
    confidence = _normalize(residual_energy)
    return ResidualHeatmap(
        residual_energy=residual_energy,
        residual_velocity=np.broadcast_to(residual_velocity, energy.shape).copy(),
        confidence=confidence,
    )


def compute_residual_voxel_map(
    voxel_rf_map: VoxelRFMap,
    pose: Pose6DoF,
    rf_geometry: RFGeometryConfig,
) -> ResidualVoxelMap:
    voxels = np.asarray(voxel_rf_map.voxel_positions_xyz, dtype=float)
    expected_velocity = estimate_expected_velocity_field(pose, voxels)
    link_tx = rf_geometry.tx_positions_xyz[0]
    link_rx = rf_geometry.rx_positions_xyz[0]
    expected_doppler = project_velocity_to_bistatic_doppler(
        expected_velocity,
        voxels,
        link_tx,
        link_rx,
        rf_geometry.center_frequency_hz,
        rf_geometry.speed_of_light_mps,
    )
    observed = np.asarray(voxel_rf_map.observed_doppler_hz, dtype=float)
    if observed.ndim == 1:
        observed = np.broadcast_to(observed[None, :], voxel_rf_map.energy.shape)
    if observed.shape != voxel_rf_map.energy.shape:
        raise ValueError("observed_doppler_hz must have shape (T, N)")

    expected = np.broadcast_to(expected_doppler[None, :], observed.shape)
    residual_doppler = observed - expected
    residual_scale = np.abs(residual_doppler) / (np.abs(observed) + np.abs(expected) + 1e-9)
    residual_energy = np.asarray(voxel_rf_map.energy, dtype=float) * residual_scale
    confidence = _normalize(residual_energy) * np.asarray(voxel_rf_map.confidence, dtype=float)
    return ResidualVoxelMap(
        voxel_positions_xyz=voxels,
        residual_energy=residual_energy,
        residual_doppler_hz=residual_doppler,
        expected_doppler_hz=expected,
        observed_doppler_hz=observed,
        confidence=confidence,
    )


def _reshape_expected_speed(expected_speed: np.ndarray, spatial_shape: tuple[int, ...]) -> np.ndarray:
    if not spatial_shape:
        return float(np.mean(expected_speed)) if expected_speed.size else 0.0

    spatial_size = int(np.prod(spatial_shape))
    if expected_speed.size == spatial_size:
        return expected_speed.reshape((1, 1) + spatial_shape)
    if expected_speed.size == 1:
        return np.full((1, 1) + spatial_shape, float(expected_speed[0]))
    return np.full((1, 1) + spatial_shape, float(np.mean(expected_speed)))


def _doppler_grid(doppler_hz: np.ndarray, energy_ndim: int, spatial_shape: tuple[int, ...]) -> np.ndarray:
    return doppler_hz.reshape((1, doppler_hz.size) + (1,) * len(spatial_shape))


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= 1e-12:
        return np.zeros_like(values)
    return values / max_value
