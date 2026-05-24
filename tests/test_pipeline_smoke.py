import numpy as np

from rdith.config import RFGeometryConfig
from rdith.heatmap_adapter import wrap_existing_heatmap
from rdith.pipeline import run_rdith_pipeline
from rdith.types import CandidateROI, Pose6DoF


def test_pipeline_returns_finite_features_with_world_voxels():
    tof = np.array([2.0, 3.0])
    doppler = np.array([-1.0, 1.0])
    energy = np.array([[[0.0, 5.0], [10.0, 0.0]]])
    heatmap = wrap_existing_heatmap(energy, np.array([0.0]), doppler, tof_s=tof)
    geometry = RFGeometryConfig(
        tx_positions_xyz=np.array([[0.0, 0.0, 0.0]]),
        rx_positions_xyz=np.array([[1.0, 0.0, 0.0]]),
        tx_orientations=None,
        rx_orientations=None,
        center_frequency_hz=1.0,
        speed_of_light_mps=1.0,
    )
    pose = Pose6DoF(
        timestamp_s=0.0,
        position_xyz=np.zeros(3),
        rotation_matrix=np.eye(3),
        linear_velocity_xyz=np.zeros(3),
        angular_velocity_xyz=np.zeros(3),
    )
    rois = [CandidateROI(1, np.array([0.5, 0.0, 0.0]), np.ones(3), 0.8, 0.1)]
    result = run_rdith_pipeline(
        heatmap,
        pose,
        rois,
        rf_geometry=geometry,
        voxel_positions_xyz=np.array([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        residual_threshold=0.1,
    )
    assert result["residual_map"].voxel_positions_xyz.shape == (2, 3)
    assert len(result["roi_features"]) == 1
    assert np.all(np.isfinite(result["roi_features"][0].features))
