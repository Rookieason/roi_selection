import numpy as np

from rdith.config import RFGeometryConfig
from rdith.doppler_projection import project_velocity_to_bistatic_doppler
from rdith.residual import compute_residual_voxel_map, estimate_expected_velocity_field
from rdith.types import Pose6DoF, VoxelRFMap


def test_rigid_velocity_field_includes_angular_component():
    pose = Pose6DoF(
        timestamp_s=0.0,
        position_xyz=np.zeros(3),
        rotation_matrix=np.eye(3),
        linear_velocity_xyz=np.array([1.0, 0.0, 0.0]),
        angular_velocity_xyz=np.array([0.0, 0.0, 1.0]),
    )
    velocity = estimate_expected_velocity_field(pose, np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(velocity, np.array([[1.0, 1.0, 0.0]]))


def test_bistatic_projection_shape_and_value():
    velocity = np.array([[1.0, 0.0, 0.0]])
    voxel = np.array([[1.0, 0.0, 0.0]])
    doppler = project_velocity_to_bistatic_doppler(
        velocity,
        voxel,
        tx_position_xyz=np.array([0.0, 0.0, 0.0]),
        rx_position_xyz=np.array([0.0, 1.0, 0.0]),
        carrier_frequency_hz=1.0,
        speed_of_light_mps=1.0,
    )
    expected = -(1.0 + 1.0 / np.sqrt(2.0))
    assert np.allclose(doppler, np.array([expected]))


def test_residual_voxel_map_is_zero_when_observed_matches_expected():
    geometry = RFGeometryConfig(
        tx_positions_xyz=np.array([[0.0, 0.0, 0.0]]),
        rx_positions_xyz=np.array([[0.0, 1.0, 0.0]]),
        tx_orientations=None,
        rx_orientations=None,
        center_frequency_hz=1.0,
        speed_of_light_mps=1.0,
    )
    pose = Pose6DoF(
        timestamp_s=0.0,
        position_xyz=np.zeros(3),
        rotation_matrix=np.eye(3),
        linear_velocity_xyz=np.array([1.0, 0.0, 0.0]),
        angular_velocity_xyz=np.zeros(3),
    )
    voxel = np.array([[1.0, 0.0, 0.0]])
    expected_velocity = estimate_expected_velocity_field(pose, voxel)
    observed = project_velocity_to_bistatic_doppler(
        expected_velocity,
        voxel,
        geometry.tx_positions_xyz[0],
        geometry.rx_positions_xyz[0],
        geometry.center_frequency_hz,
        geometry.speed_of_light_mps,
    ).reshape(1, 1)
    voxel_map = VoxelRFMap(
        voxel_positions_xyz=voxel,
        energy=np.array([[10.0]]),
        observed_doppler_hz=observed,
        confidence=np.ones((1, 1)),
    )
    residual = compute_residual_voxel_map(voxel_map, pose, geometry)
    assert np.allclose(residual.residual_energy, 0.0)
