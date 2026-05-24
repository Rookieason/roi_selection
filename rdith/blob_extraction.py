from typing import List

import numpy as np

from .types import ResidualHeatmap, RFBlob
from .types import ResidualVoxelMap


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


def extract_rf_blobs_from_voxels(
    residual_map: ResidualVoxelMap,
    threshold: float,
    distance_eps_m: float,
    min_samples: int,
) -> List[RFBlob]:
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if distance_eps_m <= 0:
        raise ValueError("distance_eps_m must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")

    voxels = np.asarray(residual_map.voxel_positions_xyz, dtype=float)
    residual_energy = np.asarray(residual_map.residual_energy, dtype=float)
    residual_doppler = np.asarray(residual_map.residual_doppler_hz, dtype=float)
    confidence = np.asarray(residual_map.confidence, dtype=float)
    if residual_energy.ndim != 2 or residual_energy.shape[1] != voxels.shape[0]:
        raise ValueError("residual_energy must have shape (T, N_voxel)")

    active_pairs = np.argwhere(residual_energy >= threshold)
    if active_pairs.size == 0:
        return []

    active_voxel_ids = np.unique(active_pairs[:, 1])
    active_points = voxels[active_voxel_ids]
    clusters = _cluster_points(active_points, distance_eps_m, min_samples)
    blobs: List[RFBlob] = []
    for cluster in clusters:
        voxel_ids = active_voxel_ids[cluster]
        voxel_energy = np.sum(residual_energy[:, voxel_ids], axis=0)
        total_energy = float(np.sum(voxel_energy))
        if total_energy <= 1e-12:
            continue
        centroid = np.average(voxels[voxel_ids], axis=0, weights=voxel_energy)
        doppler_values = residual_doppler[:, voxel_ids].reshape(-1)
        blobs.append(
            RFBlob(
                centroid_xyz=centroid,
                velocity_xyz=np.zeros(3, dtype=float),
                energy=total_energy,
                doppler_bandwidth=float(np.std(doppler_values)),
                confidence=float(np.mean(confidence[:, voxel_ids])),
                lifetime_frames=1,
            )
        )
    return blobs


def estimate_blob_velocities_from_previous(
    blobs: List[RFBlob],
    previous_blobs: List[RFBlob],
    dt_s: float,
    max_match_distance_m: float = 1.0,
) -> List[RFBlob]:
    if dt_s <= 0 or not previous_blobs:
        return blobs
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
        if distances[best_offset] <= max_match_distance_m:
            previous = previous_blobs[best_previous_id]
            blob.velocity_xyz = (blob.centroid_xyz - previous.centroid_xyz) / dt_s
            blob.lifetime_frames = previous.lifetime_frames + 1
            unmatched_previous.remove(best_previous_id)
    return blobs


def _cluster_points(points: np.ndarray, distance_eps_m: float, min_samples: int) -> list[np.ndarray]:
    visited = np.zeros(points.shape[0], dtype=bool)
    assigned = np.zeros(points.shape[0], dtype=bool)
    clusters: list[np.ndarray] = []
    for point_id in range(points.shape[0]):
        if visited[point_id]:
            continue
        visited[point_id] = True
        neighbors = _region_query(points, point_id, distance_eps_m)
        if neighbors.size < min_samples:
            continue
        cluster_ids = _expand_cluster(points, neighbors, visited, assigned, distance_eps_m, min_samples)
        clusters.append(cluster_ids)
    return clusters


def _expand_cluster(
    points: np.ndarray,
    seed_neighbors: np.ndarray,
    visited: np.ndarray,
    assigned: np.ndarray,
    distance_eps_m: float,
    min_samples: int,
) -> np.ndarray:
    cluster = set(int(item) for item in seed_neighbors)
    queue = list(seed_neighbors)
    while queue:
        candidate = int(queue.pop())
        if not visited[candidate]:
            visited[candidate] = True
            candidate_neighbors = _region_query(points, candidate, distance_eps_m)
            if candidate_neighbors.size >= min_samples:
                for neighbor in candidate_neighbors:
                    if int(neighbor) not in cluster:
                        queue.append(int(neighbor))
                    cluster.add(int(neighbor))
        if not assigned[candidate]:
            assigned[candidate] = True
            cluster.add(candidate)
    return np.array(sorted(cluster), dtype=int)


def _region_query(points: np.ndarray, point_id: int, distance_eps_m: float) -> np.ndarray:
    distances = np.linalg.norm(points - points[point_id], axis=1)
    return np.flatnonzero(distances <= distance_eps_m)


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
