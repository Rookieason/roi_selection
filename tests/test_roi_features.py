import numpy as np

from rdith.roi_features import build_roi_feature_vector, compute_roi_motion_energy
from rdith.types import CandidateROI, ResidualVoxelMap, VoxelRFMap


def test_roi_local_pooling_prefers_nearby_hot_voxel():
    residual = ResidualVoxelMap(
        voxel_positions_xyz=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        residual_energy=np.array([[10.0, 0.0]]),
        residual_doppler_hz=np.array([[3.0, 0.0]]),
        expected_doppler_hz=np.zeros((1, 2)),
        observed_doppler_hz=np.array([[3.0, 0.0]]),
        confidence=np.ones((1, 2)),
    )
    roi_a = CandidateROI(1, np.zeros(3), np.ones(3), 1.0, 0.0)
    roi_b = CandidateROI(2, np.array([5.0, 0.0, 0.0]), np.ones(3), 1.0, 0.0)
    assert compute_roi_motion_energy(roi_a, residual, radius_m=0.5) > compute_roi_motion_energy(roi_b, residual, radius_m=0.5)


def test_feature_vector_is_finite_and_named():
    residual = ResidualVoxelMap(
        voxel_positions_xyz=np.array([[0.0, 0.0, 0.0]]),
        residual_energy=np.array([[10.0]]),
        residual_doppler_hz=np.array([[3.0]]),
        expected_doppler_hz=np.zeros((1, 1)),
        observed_doppler_hz=np.array([[3.0]]),
        confidence=np.ones((1, 1)),
    )
    voxel_map = VoxelRFMap(
        voxel_positions_xyz=residual.voxel_positions_xyz,
        energy=np.array([[10.0]]),
        observed_doppler_hz=np.array([[3.0]]),
        confidence=np.ones((1, 1)),
    )
    roi = CandidateROI(1, np.zeros(3), np.ones(3), 0.2, 0.8)
    vector = build_roi_feature_vector(roi, [], residual, voxel_map)
    assert len(vector.feature_names) == vector.features.size
    assert np.all(np.isfinite(vector.features))
