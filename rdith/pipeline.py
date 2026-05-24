from typing import List, Optional

import numpy as np

from .blob_extraction import extract_rf_blobs, threshold_residual_heatmap, track_rf_blobs
from .residual import compute_residual_heatmap, estimate_expected_velocity_field
from .roi_features import build_roi_feature_vector
from .types import CandidateROI, HeatmapTensor, Pose6DoF, RFBlob


def run_rdith_pipeline(
    heatmap: HeatmapTensor,
    pose: Pose6DoF,
    candidate_rois: List[CandidateROI],
    previous_blobs: Optional[List[RFBlob]] = None,
):
    voxel_positions = _default_voxel_positions(heatmap)
    expected_velocity = estimate_expected_velocity_field(pose, voxel_positions)
    residual_heatmap = compute_residual_heatmap(heatmap, expected_velocity)
    threshold = _adaptive_threshold(residual_heatmap.residual_energy)
    binary_mask = threshold_residual_heatmap(residual_heatmap, threshold)
    blobs = extract_rf_blobs(residual_heatmap, binary_mask)
    tracked_blobs = track_rf_blobs(blobs, previous_blobs or [])
    roi_features = [
        build_roi_feature_vector(roi, tracked_blobs, residual_heatmap, heatmap)
        for roi in candidate_rois
    ]
    return {
        "residual_heatmap": residual_heatmap,
        "tracked_blobs": tracked_blobs,
        "roi_features": roi_features,
    }


def _default_voxel_positions(heatmap: HeatmapTensor) -> np.ndarray:
    tof = np.array([0.0]) if heatmap.tof_s is None else np.asarray(heatmap.tof_s, dtype=float)
    aoa = np.array([0.0]) if heatmap.aoa_rad is None else np.asarray(heatmap.aoa_rad, dtype=float)
    radius = tof * 299792458.0 / 2.0
    rr, aa = np.meshgrid(radius, aoa, indexing="ij")
    x = rr * np.cos(aa)
    y = rr * np.sin(aa)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def _adaptive_threshold(residual_energy: np.ndarray) -> float:
    data = np.asarray(residual_energy, dtype=float)
    if data.size == 0:
        return 0.0
    positive = data[data > 0.0]
    if positive.size == 0:
        return float("inf")
    return float(np.percentile(positive, 90.0))
