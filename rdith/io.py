import csv
import json
from typing import List

import numpy as np

from .types import CandidateROI, Pose6DoF


def load_pose_sequence_csv(path: str) -> list[Pose6DoF]:
    poses: list[Pose6DoF] = []
    with open(path, "r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            poses.append(_pose_from_row(row))
    poses.sort(key=lambda pose: pose.timestamp_s)
    return poses


def interpolate_pose_at_time(
    poses: list[Pose6DoF],
    timestamp_s: float,
) -> Pose6DoF:
    if not poses:
        raise ValueError("poses cannot be empty")
    if timestamp_s <= poses[0].timestamp_s:
        return poses[0]
    if timestamp_s >= poses[-1].timestamp_s:
        return poses[-1]

    times = np.array([pose.timestamp_s for pose in poses])
    right = int(np.searchsorted(times, timestamp_s, side="right"))
    left_pose = poses[right - 1]
    right_pose = poses[right]
    alpha = (timestamp_s - left_pose.timestamp_s) / (right_pose.timestamp_s - left_pose.timestamp_s)
    rotation = (1.0 - alpha) * left_pose.rotation_matrix + alpha * right_pose.rotation_matrix
    rotation = _orthonormalize_rotation(rotation)
    return Pose6DoF(
        timestamp_s=float(timestamp_s),
        position_xyz=_lerp(left_pose.position_xyz, right_pose.position_xyz, alpha),
        rotation_matrix=rotation,
        linear_velocity_xyz=_lerp(left_pose.linear_velocity_xyz, right_pose.linear_velocity_xyz, alpha),
        angular_velocity_xyz=_lerp(left_pose.angular_velocity_xyz, right_pose.angular_velocity_xyz, alpha),
    )


def load_candidate_rois_json(path: str) -> list[CandidateROI]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, dict) and "rois" in raw:
        raw = raw["rois"]
    return [_roi_from_dict(item) for item in raw]


def load_candidate_roi_sequence_json(path: str) -> dict[float, list[CandidateROI]]:
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)
    if isinstance(raw, list):
        return {0.0: [_roi_from_dict(item) for item in raw]}

    sequence = raw.get("frames", raw)
    result: dict[float, list[CandidateROI]] = {}
    if isinstance(sequence, list):
        for frame in sequence:
            result[float(frame["timestamp_s"])] = [_roi_from_dict(item) for item in frame["rois"]]
    else:
        for timestamp, rois in sequence.items():
            result[float(timestamp)] = [_roi_from_dict(item) for item in rois]
    return result


def select_rois_at_time(roi_sequence: dict[float, list[CandidateROI]], timestamp_s: float) -> list[CandidateROI]:
    if not roi_sequence:
        return []
    times = np.array(sorted(roi_sequence.keys()), dtype=float)
    nearest = float(times[np.abs(times - timestamp_s).argmin()])
    return roi_sequence[nearest]


def _pose_from_row(row: dict[str, str]) -> Pose6DoF:
    rotation = np.array(
        [[float(row[f"rotation_{r}{c}"]) for c in range(3)] for r in range(3)],
        dtype=float,
    )
    rotation = _orthonormalize_rotation(rotation)
    return Pose6DoF(
        timestamp_s=float(row["timestamp_s"]),
        position_xyz=np.array([float(row[f"position_{axis}"]) for axis in ("x", "y", "z")], dtype=float),
        rotation_matrix=rotation,
        linear_velocity_xyz=_vector_from_row(row, "linear_velocity", default=np.zeros(3)),
        angular_velocity_xyz=_vector_from_row(row, "angular_velocity", default=np.zeros(3)),
    )


def _vector_from_row(row: dict[str, str], prefix: str, default: np.ndarray) -> np.ndarray:
    keys = [f"{prefix}_{axis}" for axis in ("x", "y", "z")]
    if any(key not in row or row[key] == "" for key in keys):
        return default.astype(float)
    return np.array([float(row[key]) for key in keys], dtype=float)


def _roi_from_dict(item: dict) -> CandidateROI:
    return CandidateROI(
        roi_id=int(item["roi_id"]),
        center_xyz=np.asarray(item["center_xyz"], dtype=float),
        bbox_extent_xyz=np.asarray(item["bbox_extent_xyz"], dtype=float),
        visibility_score=float(np.clip(item.get("visibility_score", 1.0), 0.0, 1.0)),
        occlusion_score=float(np.clip(item.get("occlusion_score", 0.0), 0.0, 1.0)),
    )


def _lerp(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    return (1.0 - alpha) * np.asarray(left, dtype=float) + alpha * np.asarray(right, dtype=float)


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation)
    normalized = u @ vh
    if np.linalg.det(normalized) < 0:
        u[:, -1] *= -1
        normalized = u @ vh
    return normalized
