from typing import List, Optional

import numpy as np

from .blob_extraction import (
    estimate_blob_velocities_from_previous,
    extract_rf_blobs_from_voxels,
)
from .config import RFGeometryConfig
from .residual import compute_residual_voxel_map
from .roi_features import build_roi_feature_vector
from .types import CandidateROI, HeatmapTensor, Pose6DoF, RFBlob
from .voxelization import build_world_voxel_grid, map_heatmap_bins_to_voxels


def run_rdith_pipeline(
    heatmap: HeatmapTensor,
    pose: Pose6DoF,
    candidate_rois: List[CandidateROI],
    rf_geometry: RFGeometryConfig,
    voxel_positions_xyz: Optional[np.ndarray] = None,
    voxel_bounds: Optional[dict[str, tuple[float, float]]] = None,
    voxel_resolution_m: float = 0.25,
    previous_blobs: Optional[List[RFBlob]] = None,
    residual_threshold: Optional[float] = None,
    blob_distance_eps_m: float = 0.5,
    blob_min_samples: int = 1,
    roi_radius_m: float = 0.75,
):
    voxel_positions = _resolve_voxel_positions(voxel_positions_xyz, voxel_bounds, voxel_resolution_m)
    voxel_rf_map = map_heatmap_bins_to_voxels(heatmap, rf_geometry, voxel_positions)
    residual_map = compute_residual_voxel_map(voxel_rf_map, pose, rf_geometry)
    threshold = residual_threshold
    if threshold is None:
        threshold = _adaptive_threshold(residual_map.residual_energy)
    blobs = extract_rf_blobs_from_voxels(
        residual_map,
        threshold=threshold,
        distance_eps_m=blob_distance_eps_m,
        min_samples=blob_min_samples,
    )
    tracked_blobs = estimate_blob_velocities_from_previous(
        blobs,
        previous_blobs or [],
        dt_s=_estimate_dt_s(heatmap),
        max_match_distance_m=blob_distance_eps_m * 2.0,
    )
    roi_features = [
        build_roi_feature_vector(roi, tracked_blobs, residual_map, voxel_rf_map, radius_m=roi_radius_m)
        for roi in candidate_rois
    ]
    return {
        "voxel_rf_map": voxel_rf_map,
        "residual_map": residual_map,
        "tracked_blobs": tracked_blobs,
        "roi_features": roi_features,
    }


def _resolve_voxel_positions(
    voxel_positions_xyz: Optional[np.ndarray],
    voxel_bounds: Optional[dict[str, tuple[float, float]]],
    voxel_resolution_m: float,
) -> np.ndarray:
    if voxel_positions_xyz is not None:
        voxels = np.asarray(voxel_positions_xyz, dtype=float)
        if voxels.ndim != 2 or voxels.shape[1] != 3:
            raise ValueError("voxel_positions_xyz must have shape (N, 3)")
        return voxels
    if voxel_bounds is None:
        raise ValueError("Provide voxel_positions_xyz or voxel_bounds for calibrated RDITH")
    return build_world_voxel_grid(
        voxel_bounds["x"],
        voxel_bounds["y"],
        voxel_bounds["z"],
        voxel_resolution_m,
    )


def _adaptive_threshold(residual_energy: np.ndarray) -> float:
    data = np.asarray(residual_energy, dtype=float)
    if data.size == 0:
        return 0.0
    positive = data[data > 0.0]
    if positive.size == 0:
        return float("inf")
    return float(np.percentile(positive, 90.0))


def _estimate_dt_s(heatmap: HeatmapTensor) -> float:
    time_s = np.asarray(heatmap.time_s, dtype=float)
    if time_s.size < 2:
        return 1.0
    dt = float(np.median(np.diff(time_s)))
    return dt if dt > 0 else 1.0
