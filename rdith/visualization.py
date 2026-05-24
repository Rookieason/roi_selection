from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .types import ResidualHeatmap, RFBlob, ROIFeatureVector
from .config import RFGeometryConfig
from .types import CandidateROI, Pose6DoF, ResidualVoxelMap


def visualize_residual_heatmap(
    residual_heatmap: ResidualHeatmap,
):
    data = np.asarray(residual_heatmap.residual_energy, dtype=float)
    projected = _project_to_2d(data)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(projected, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Residual Heatmap")
    ax.set_xlabel("Bin")
    ax.set_ylabel("Doppler Bin")
    fig.colorbar(image, ax=ax, label="Residual Energy")
    fig.tight_layout()
    return fig


def visualize_rf_blobs(
    blobs: List[RFBlob],
):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    if blobs:
        centroids = np.vstack([blob.centroid_xyz for blob in blobs])
        energies = np.array([blob.energy for blob in blobs])
        scatter = ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c=energies, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label="Energy")
    ax.set_title("RF Blobs")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    return fig


def visualize_roi_features(
    roi_features: List[ROIFeatureVector],
):
    feature_matrix = np.vstack([item.features for item in roi_features]) if roi_features else np.zeros((0, 0))
    fig, ax = plt.subplots(figsize=(9, 5))
    image = ax.imshow(feature_matrix, aspect="auto", origin="upper", cmap="cividis")
    ax.set_title("ROI Feature Vectors")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("ROI Index")
    if roi_features:
        ax.set_yticks(np.arange(len(roi_features)))
        ax.set_yticklabels([str(item.roi_id) for item in roi_features])
    fig.colorbar(image, ax=ax, label="Value")
    fig.tight_layout()
    return fig


def visualize_scene_rdith(
    rf_geometry: RFGeometryConfig,
    pose: Pose6DoF,
    rois: List[CandidateROI],
    blobs: List[RFBlob],
    residual_map: ResidualVoxelMap,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    tx = np.asarray(rf_geometry.tx_positions_xyz, dtype=float)
    rx = np.asarray(rf_geometry.rx_positions_xyz, dtype=float)
    ax.scatter(tx[:, 0], tx[:, 1], tx[:, 2], marker="^", s=80, label="Tx")
    ax.scatter(rx[:, 0], rx[:, 1], rx[:, 2], marker="s", s=80, label="Rx")

    hmd = np.asarray(pose.position_xyz, dtype=float)
    forward = np.asarray(pose.rotation_matrix, dtype=float) @ np.array([1.0, 0.0, 0.0])
    ax.scatter([hmd[0]], [hmd[1]], [hmd[2]], marker="o", s=80, label="HMD")
    ax.quiver(hmd[0], hmd[1], hmd[2], forward[0], forward[1], forward[2], length=0.5, color="black")

    residual_energy = np.mean(np.asarray(residual_map.residual_energy, dtype=float), axis=0)
    active = residual_energy > np.percentile(residual_energy, 90) if residual_energy.size else np.array([], dtype=bool)
    voxels = np.asarray(residual_map.voxel_positions_xyz, dtype=float)
    if voxels.size and np.any(active):
        scatter = ax.scatter(
            voxels[active, 0],
            voxels[active, 1],
            voxels[active, 2],
            c=residual_energy[active],
            cmap="magma",
            s=10,
            alpha=0.45,
            label="Residual Voxels",
        )
        fig.colorbar(scatter, ax=ax, label="Residual Energy")

    for roi in rois:
        _draw_roi_box(ax, roi)

    if blobs:
        centroids = np.vstack([blob.centroid_xyz for blob in blobs])
        velocities = np.vstack([blob.velocity_xyz for blob in blobs])
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker="*", s=120, label="RF Blobs")
        ax.quiver(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            velocities[:, 0],
            velocities[:, 1],
            velocities[:, 2],
            length=1.0,
            normalize=False,
            color="tab:red",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("RDITH Scene")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _project_to_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return np.mean(data, axis=tuple(i for i in range(data.ndim) if i not in (1, 2)))


def _draw_roi_box(ax, roi: CandidateROI) -> None:
    center = np.asarray(roi.center_xyz, dtype=float)
    half = np.asarray(roi.bbox_extent_xyz, dtype=float) / 2.0
    corners = np.array(
        [
            [center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (-1, 1)
        ]
    )
    edges = [
        (0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
        (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7),
    ]
    for left, right in edges:
        ax.plot(
            [corners[left, 0], corners[right, 0]],
            [corners[left, 1], corners[right, 1]],
            [corners[left, 2], corners[right, 2]],
            color="tab:blue",
            linewidth=1,
        )
