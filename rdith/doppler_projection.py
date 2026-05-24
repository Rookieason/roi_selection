import numpy as np


def project_velocity_to_bistatic_doppler(
    velocity_xyz: np.ndarray,
    voxel_positions_xyz: np.ndarray,
    tx_position_xyz: np.ndarray,
    rx_position_xyz: np.ndarray,
    carrier_frequency_hz: float,
    speed_of_light_mps: float = 299792458.0,
) -> np.ndarray:
    velocity = np.asarray(velocity_xyz, dtype=float)
    voxels = np.asarray(voxel_positions_xyz, dtype=float)
    tx = np.asarray(tx_position_xyz, dtype=float).reshape(3)
    rx = np.asarray(rx_position_xyz, dtype=float).reshape(3)
    if velocity.shape != voxels.shape:
        raise ValueError("velocity_xyz and voxel_positions_xyz must both have shape (N, 3)")
    if carrier_frequency_hz <= 0:
        raise ValueError("carrier_frequency_hz must be positive")

    wavelength = speed_of_light_mps / carrier_frequency_hz
    bistatic_direction = _unit_vectors(voxels - tx[None, :]) + _unit_vectors(voxels - rx[None, :])
    return -np.sum(velocity * bistatic_direction, axis=1) / wavelength


def _unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors, dtype=float), where=norms > 1e-12)
