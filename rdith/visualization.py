from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .types import ResidualHeatmap, RFBlob, ROIFeatureVector


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


def _project_to_2d(data: np.ndarray) -> np.ndarray:
    if data.ndim == 0:
        return data.reshape(1, 1)
    if data.ndim == 1:
        return data.reshape(1, -1)
    if data.ndim == 2:
        return data
    return np.mean(data, axis=tuple(i for i in range(data.ndim) if i not in (1, 2)))
