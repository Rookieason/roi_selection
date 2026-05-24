import numpy as np

from .types import CandidateROI


def compute_distance_point_to_roi(point_xyz: np.ndarray, roi: CandidateROI) -> float:
    point = np.asarray(point_xyz, dtype=float)
    center = np.asarray(roi.center_xyz, dtype=float)
    half_extent = np.asarray(roi.bbox_extent_xyz, dtype=float) / 2.0
    outside_delta = np.maximum(np.abs(point - center) - half_extent, 0.0)
    return float(np.linalg.norm(outside_delta))


def compute_roi_direction_vector(source_xyz: np.ndarray, roi: CandidateROI) -> np.ndarray:
    source = np.asarray(source_xyz, dtype=float)
    direction = np.asarray(roi.center_xyz, dtype=float) - source
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return direction / norm


def is_point_inside_roi(point_xyz: np.ndarray, roi: CandidateROI) -> bool:
    point = np.asarray(point_xyz, dtype=float)
    center = np.asarray(roi.center_xyz, dtype=float)
    half_extent = np.asarray(roi.bbox_extent_xyz, dtype=float) / 2.0
    return bool(np.all(np.abs(point - center) <= half_extent))
