import numpy as np

from .config import RFGeometryConfig
from .types import HeatmapTensor, VoxelRFMap


def build_world_voxel_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution_m: float,
) -> np.ndarray:
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    xs = _axis_from_range(x_range, resolution_m)
    ys = _axis_from_range(y_range, resolution_m)
    zs = _axis_from_range(z_range, resolution_m)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)


def compute_bistatic_tof_for_voxels(
    voxel_positions_xyz: np.ndarray,
    tx_position_xyz: np.ndarray,
    rx_position_xyz: np.ndarray,
    speed_of_light_mps: float = 299792458.0,
) -> np.ndarray:
    voxels = np.asarray(voxel_positions_xyz, dtype=float)
    tx = np.asarray(tx_position_xyz, dtype=float).reshape(3)
    rx = np.asarray(rx_position_xyz, dtype=float).reshape(3)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError("voxel_positions_xyz must have shape (N, 3)")
    tx_distance = np.linalg.norm(voxels - tx[None, :], axis=1)
    rx_distance = np.linalg.norm(voxels - rx[None, :], axis=1)
    return (tx_distance + rx_distance) / speed_of_light_mps


def map_heatmap_bins_to_voxels(
    heatmap: HeatmapTensor,
    rf_geometry: RFGeometryConfig,
    voxel_positions_xyz: np.ndarray,
) -> VoxelRFMap:
    voxels = np.asarray(voxel_positions_xyz, dtype=float)
    if voxels.ndim != 2 or voxels.shape[1] != 3:
        raise ValueError("voxel_positions_xyz must have shape (N, 3)")
    if heatmap.tof_s is None:
        raise ValueError("ToF axis is required to map heatmap bins to world voxels")

    energy = np.asarray(heatmap.energy, dtype=float)
    if energy.ndim not in (3, 4):
        raise ValueError("heatmap.energy must be (time, doppler, tof[, aoa])")

    doppler = np.asarray(heatmap.doppler_hz, dtype=float)
    tof = np.asarray(heatmap.tof_s, dtype=float)
    aoa = None if heatmap.aoa_rad is None else np.asarray(heatmap.aoa_rad, dtype=float)
    link_tx, link_rx = _default_link(rf_geometry)
    voxel_tof = compute_bistatic_tof_for_voxels(
        voxels,
        link_tx,
        link_rx,
        rf_geometry.speed_of_light_mps,
    )
    tof_indices = _nearest_indices(tof, voxel_tof)

    if energy.ndim == 4:
        if aoa is None:
            raise ValueError("AoA axis is required for 4D heatmap energy")
        voxel_aoa = np.arctan2(voxels[:, 1] - link_rx[1], voxels[:, 0] - link_rx[0])
        aoa_indices = _nearest_indices(aoa, voxel_aoa)
        voxel_doppler_indices = np.argmax(energy[:, :, tof_indices, aoa_indices], axis=1)
        voxel_energy = energy[
            np.arange(energy.shape[0])[:, None],
            voxel_doppler_indices,
            tof_indices[None, :],
            aoa_indices[None, :],
        ]
    else:
        voxel_doppler_indices = np.argmax(energy[:, :, tof_indices], axis=1)
        voxel_energy = energy[
            np.arange(energy.shape[0])[:, None],
            voxel_doppler_indices,
            tof_indices[None, :],
        ]

    observed_doppler = doppler[voxel_doppler_indices]
    confidence = _normalize_per_time(np.clip(voxel_energy, 0.0, None))
    return VoxelRFMap(
        voxel_positions_xyz=voxels,
        energy=np.clip(voxel_energy, 0.0, None),
        observed_doppler_hz=observed_doppler,
        confidence=confidence,
    )


def _axis_from_range(axis_range: tuple[float, float], resolution_m: float) -> np.ndarray:
    start, stop = axis_range
    if stop < start:
        raise ValueError("range stop must be greater than or equal to start")
    return np.arange(start, stop + resolution_m * 0.5, resolution_m, dtype=float)


def _default_link(rf_geometry: RFGeometryConfig) -> tuple[np.ndarray, np.ndarray]:
    return rf_geometry.tx_positions_xyz[0], rf_geometry.rx_positions_xyz[0]


def _nearest_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError("axis must be a non-empty 1D array")
    return np.abs(axis[:, None] - values[None, :]).argmin(axis=0)


def _normalize_per_time(values: np.ndarray) -> np.ndarray:
    max_per_time = np.max(values, axis=1, keepdims=True)
    return np.divide(values, max_per_time, out=np.zeros_like(values), where=max_per_time > 1e-12)
