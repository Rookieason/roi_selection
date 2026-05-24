from typing import List

import numpy as np

from .types import ResidualHeatmap, RFBlob


def threshold_residual_heatmap(
    residual_heatmap: ResidualHeatmap,
    threshold: float,
) -> np.ndarray:
    residual_energy = np.asarray(residual_heatmap.residual_energy, dtype=float)
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return residual_energy >= threshold


def extract_rf_blobs(
    residual_heatmap: ResidualHeatmap,
    binary_mask: np.ndarray,
) -> List[RFBlob]:
    residual_energy = np.asarray(residual_heatmap.residual_energy, dtype=float)
    residual_velocity = np.asarray(residual_heatmap.residual_velocity, dtype=float)
    confidence = np.asarray(residual_heatmap.confidence, dtype=float)
    mask = np.asarray(binary_mask, dtype=bool)
    if mask.shape != residual_energy.shape:
        raise ValueError("binary_mask must match residual_heatmap.residual_energy shape")

    components, num_components = _connected_components(mask)
    blobs: List[RFBlob] = []
    for component_id in range(1, num_components + 1):
        indices = np.argwhere(components == component_id)
        if indices.size == 0:
            continue

        weights = residual_energy[tuple(indices.T)]
        total_weight = float(np.sum(weights))
        if total_weight <= 1e-12:
            centroid_index = np.mean(indices, axis=0)
        else:
            centroid_index = np.average(indices, axis=0, weights=weights)

        velocity_values = residual_velocity[tuple(indices.T)]
        velocity_scalar = float(np.average(velocity_values, weights=weights)) if total_weight > 1e-12 else 0.0
        blobs.append(
            RFBlob(
                centroid_xyz=_index_to_placeholder_xyz(centroid_index),
                velocity_xyz=np.array([velocity_scalar, 0.0, 0.0], dtype=float),
                energy=total_weight,
                doppler_bandwidth=float(np.std(velocity_values)),
                confidence=float(np.mean(confidence[tuple(indices.T)])),
                lifetime_frames=1,
            )
        )
    return blobs


def track_rf_blobs(
    blobs: List[RFBlob],
    previous_blobs: List[RFBlob],
) -> List[RFBlob]:
    if not previous_blobs:
        return blobs
    if not blobs:
        return []

    max_match_distance = 2.0
    unmatched_previous = set(range(len(previous_blobs)))
    for blob in blobs:
        if not unmatched_previous:
            break
        previous_candidates = list(unmatched_previous)
        distances = [
            np.linalg.norm(blob.centroid_xyz - previous_blobs[previous_id].centroid_xyz)
            for previous_id in previous_candidates
        ]
        best_offset = int(np.argmin(distances))
        best_previous_id = previous_candidates[best_offset]
        if distances[best_offset] <= max_match_distance:
            blob.lifetime_frames = previous_blobs[best_previous_id].lifetime_frames + 1
            unmatched_previous.remove(best_previous_id)
    return blobs


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    components = np.zeros(mask.shape, dtype=np.int32)
    component_id = 0
    visited = np.zeros(mask.shape, dtype=bool)
    neighbor_offsets = _neighbor_offsets(mask.ndim)

    for start_index in np.argwhere(mask):
        start = tuple(int(value) for value in start_index)
        if visited[start]:
            continue

        component_id += 1
        stack = [start]
        visited[start] = True
        components[start] = component_id

        while stack:
            current = stack.pop()
            for offset in neighbor_offsets:
                neighbor = tuple(current[axis] + offset[axis] for axis in range(mask.ndim))
                if not _in_bounds(neighbor, mask.shape):
                    continue
                if visited[neighbor] or not mask[neighbor]:
                    continue
                visited[neighbor] = True
                components[neighbor] = component_id
                stack.append(neighbor)

    return components, component_id


def _neighbor_offsets(ndim: int) -> list[tuple[int, ...]]:
    offsets = []
    for axis in range(ndim):
        for step in (-1, 1):
            offset = [0] * ndim
            offset[axis] = step
            offsets.append(tuple(offset))
    return offsets


def _in_bounds(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= index[axis] < shape[axis] for axis in range(len(shape)))


def _index_to_placeholder_xyz(index: np.ndarray) -> np.ndarray:
    padded = np.zeros(3, dtype=float)
    usable_dims = min(3, index.size)
    padded[:usable_dims] = index[:usable_dims]
    return padded
