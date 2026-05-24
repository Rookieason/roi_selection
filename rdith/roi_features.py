from typing import List

import numpy as np

from .geometry import compute_distance_point_to_roi, compute_roi_direction_vector
from .types import CandidateROI, HeatmapTensor, ResidualHeatmap, RFBlob, ROIFeatureVector


def compute_roi_motion_energy(
    roi: CandidateROI,
    residual_heatmap: ResidualHeatmap,
) -> float:
    residual_energy = np.asarray(residual_heatmap.residual_energy, dtype=float)
    if residual_energy.size == 0:
        return 0.0

    # Until calibrated RF voxel-to-world geometry is supplied, use total residual
    # energy as the ROI-local proxy and let blob-to-ROI features add geometry.
    return float(np.mean(residual_energy))


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
) -> float:
    distance = compute_distance_point_to_roi(blob.centroid_xyz, roi)
    direction = compute_roi_direction_vector(blob.centroid_xyz, roi)
    closing_speed = float(np.dot(np.asarray(blob.velocity_xyz, dtype=float), direction))
    if closing_speed <= 1e-12:
        return float("inf")
    return float(distance / closing_speed)


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
    residual_heatmap: ResidualHeatmap,
    heatmap: HeatmapTensor,
) -> ROIFeatureVector:
    roi_motion_energy = compute_roi_motion_energy(roi, residual_heatmap)
    nearest_blob = _nearest_blob(roi, blobs)

    if nearest_blob is None:
        alignment = 0.0
        time_to_contact = float("inf")
        blob_energy = 0.0
        blob_confidence = 0.0
        temporal_growth = 0.0
    else:
        alignment = compute_motion_to_roi_alignment(roi, nearest_blob)
        time_to_contact = compute_time_to_contact(roi, nearest_blob)
        blob_energy = nearest_blob.energy
        blob_confidence = nearest_blob.confidence
        temporal_growth = float(max(nearest_blob.lifetime_frames - 1, 0))

    entropy = _doppler_entropy(heatmap)
    bandwidth = compute_microdoppler_bandwidth(heatmap, roi)
    visibility_conflict = compute_visibility_conflict_score(roi, roi_motion_energy)
    features = np.array(
        [
            roi_motion_energy,
            blob_energy,
            alignment,
            time_to_contact,
            entropy,
            bandwidth,
            blob_confidence,
            visibility_conflict,
            temporal_growth,
        ],
        dtype=float,
    )
    return ROIFeatureVector(roi_id=roi.roi_id, features=features)


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
