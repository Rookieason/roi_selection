from typing import List

import numpy as np

from .feature_schema import ROI_FEATURE_NAMES
from .geometry import compute_distance_point_to_roi, compute_roi_direction_vector
from .types import CandidateROI, HeatmapTensor, ResidualHeatmap, ResidualVoxelMap, RFBlob, ROIFeatureVector, VoxelRFMap
from .validation import validate_feature_vector


def compute_roi_motion_energy(
    roi: CandidateROI,
    residual_map: ResidualVoxelMap,
    radius_m: float = 0.75,
) -> float:
    voxels = np.asarray(residual_map.voxel_positions_xyz, dtype=float)
    residual_energy = np.asarray(residual_map.residual_energy, dtype=float)
    if voxels.size == 0 or residual_energy.size == 0:
        return 0.0

    distances = np.array([compute_distance_point_to_roi(point, roi) for point in voxels])
    sigma = max(radius_m, 1e-6)
    weights = np.exp(-(distances**2) / (2.0 * sigma**2))
    weighted = residual_energy * weights[None, :]
    denominator = float(np.sum(weights) * residual_energy.shape[0])
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(weighted) / denominator)


def compute_motion_to_roi_alignment(
    roi: CandidateROI,
    blob: RFBlob,
) -> float:
    direction = compute_roi_direction_vector(blob.centroid_xyz, roi)
    velocity = np.asarray(blob.velocity_xyz, dtype=float)
    speed = np.linalg.norm(velocity)
    if speed <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(velocity / speed, direction), -1.0, 1.0))


def compute_time_to_contact(
    roi: CandidateROI,
    blob: RFBlob,
    max_ttc_s: float = 10.0,
) -> float:
    distance = compute_distance_point_to_roi(blob.centroid_xyz, roi)
    direction = compute_roi_direction_vector(blob.centroid_xyz, roi)
    closing_speed = float(np.dot(np.asarray(blob.velocity_xyz, dtype=float), direction))
    if closing_speed <= 1e-12:
        return float(max_ttc_s)
    return float(np.clip(distance / closing_speed, 0.0, max_ttc_s))


def compute_microdoppler_bandwidth(
    heatmap: HeatmapTensor,
    roi: CandidateROI,
) -> float:
    energy = np.asarray(heatmap.energy, dtype=float)
    doppler = np.asarray(heatmap.doppler_hz, dtype=float)
    if energy.ndim < 2 or energy.shape[1] != doppler.size:
        raise ValueError("heatmap.energy axis 1 must match doppler_hz")

    doppler_profile = np.sum(np.clip(energy, 0.0, None), axis=tuple(range(0, energy.ndim))[0:1] + tuple(range(2, energy.ndim)))
    total = float(np.sum(doppler_profile))
    if total <= 1e-12:
        return 0.0
    mean = float(np.sum(doppler * doppler_profile) / total)
    variance = float(np.sum(((doppler - mean) ** 2) * doppler_profile) / total)
    return float(np.sqrt(max(variance, 0.0)))


def compute_roi_microdoppler_bandwidth(
    roi: CandidateROI,
    voxel_rf_map: VoxelRFMap,
    radius_m: float = 0.75,
) -> float:
    voxels = np.asarray(voxel_rf_map.voxel_positions_xyz, dtype=float)
    energy = np.asarray(voxel_rf_map.energy, dtype=float)
    doppler = np.asarray(voxel_rf_map.observed_doppler_hz, dtype=float)
    distances = np.array([compute_distance_point_to_roi(point, roi) for point in voxels])
    weights = np.exp(-(distances**2) / (2.0 * max(radius_m, 1e-6) ** 2))
    weighted_energy = energy * weights[None, :]
    total = float(np.sum(weighted_energy))
    if total <= 1e-12:
        return 0.0
    mean = float(np.sum(doppler * weighted_energy) / total)
    variance = float(np.sum(((doppler - mean) ** 2) * weighted_energy) / total)
    return float(np.sqrt(max(variance, 0.0)))


def compute_visibility_conflict_score(
    roi: CandidateROI,
    residual_energy: float,
) -> float:
    visibility = float(np.clip(roi.visibility_score, 0.0, 1.0))
    occlusion = float(np.clip(roi.occlusion_score, 0.0, 1.0))
    return float(max(residual_energy, 0.0) * (1.0 - visibility) * (1.0 + occlusion))


def build_roi_feature_vector(
    roi: CandidateROI,
    blobs: List[RFBlob],
    residual_map: ResidualVoxelMap,
    voxel_rf_map: VoxelRFMap,
    radius_m: float = 0.75,
    max_ttc_s: float = 10.0,
) -> ROIFeatureVector:
    roi_motion_energy = compute_roi_motion_energy(roi, residual_map, radius_m=radius_m)
    nearest_blob = _nearest_blob(roi, blobs)

    if nearest_blob is None:
        alignment = 0.0
        time_to_contact = max_ttc_s
        blob_energy = 0.0
        blob_confidence = 0.0
        temporal_growth = 0.0
    else:
        alignment = compute_motion_to_roi_alignment(roi, nearest_blob)
        time_to_contact = compute_time_to_contact(roi, nearest_blob, max_ttc_s=max_ttc_s)
        blob_energy = nearest_blob.energy
        blob_confidence = nearest_blob.confidence
        temporal_growth = float(max(nearest_blob.lifetime_frames - 1, 0))

    time_to_contact_score = float(np.exp(-time_to_contact / max(max_ttc_s / 3.0, 1e-6)))
    entropy = _voxel_doppler_entropy(voxel_rf_map, roi, radius_m)
    bandwidth = compute_roi_microdoppler_bandwidth(roi, voxel_rf_map, radius_m)
    visibility_conflict = compute_visibility_conflict_score(roi, roi_motion_energy)
    features = np.array(
        [
            roi_motion_energy,
            blob_energy,
            alignment,
            time_to_contact,
            time_to_contact_score,
            entropy,
            bandwidth,
            blob_confidence,
            visibility_conflict,
            temporal_growth,
        ],
        dtype=float,
    )
    feature_vector = ROIFeatureVector(
        roi_id=roi.roi_id,
        feature_names=list(ROI_FEATURE_NAMES),
        features=np.nan_to_num(features, nan=0.0, posinf=max_ttc_s, neginf=0.0),
    )
    validate_feature_vector(feature_vector)
    return feature_vector


def _nearest_blob(roi: CandidateROI, blobs: List[RFBlob]) -> RFBlob | None:
    if not blobs:
        return None
    distances = [compute_distance_point_to_roi(blob.centroid_xyz, roi) for blob in blobs]
    return blobs[int(np.argmin(distances))]


def _doppler_entropy(heatmap: HeatmapTensor) -> float:
    energy = np.clip(np.asarray(heatmap.energy, dtype=float), 0.0, None)
    if energy.size == 0:
        return 0.0
    profile = np.sum(energy, axis=tuple(range(0, energy.ndim))[0:1] + tuple(range(2, energy.ndim)))
    total = float(np.sum(profile))
    if total <= 1e-12:
        return 0.0
    probabilities = profile / total
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log(probabilities)))


def _voxel_doppler_entropy(voxel_rf_map: VoxelRFMap, roi: CandidateROI, radius_m: float) -> float:
    voxels = np.asarray(voxel_rf_map.voxel_positions_xyz, dtype=float)
    energy = np.asarray(voxel_rf_map.energy, dtype=float)
    doppler = np.asarray(voxel_rf_map.observed_doppler_hz, dtype=float)
    distances = np.array([compute_distance_point_to_roi(point, roi) for point in voxels])
    weights = np.exp(-(distances**2) / (2.0 * max(radius_m, 1e-6) ** 2))
    weighted = energy * weights[None, :]
    bins = np.unique(doppler)
    profile = np.array([np.sum(weighted[doppler == bin_value]) for bin_value in bins])
    total = float(np.sum(profile))
    if total <= 1e-12:
        return 0.0
    probabilities = profile / total
    probabilities = probabilities[probabilities > 0.0]
    return float(-np.sum(probabilities * np.log(probabilities)))
