import argparse
import json
from pathlib import Path

import numpy as np

from rdith.config import load_rf_geometry_config
from rdith.heatmap_adapter import wrap_existing_heatmap
from rdith.io import (
    interpolate_pose_at_time,
    load_candidate_roi_sequence_json,
    load_pose_sequence_csv,
    select_rois_at_time,
)
from rdith.pipeline import run_rdith_pipeline
from rdith.voxelization import build_world_voxel_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RDITH from a saved heatmap result.")
    parser.add_argument("--heatmap", required=True, help="Path to .npz or .mat heatmap file")
    parser.add_argument("--config", required=True, help="Path to upstream config.json")
    parser.add_argument("--rf_geometry", required=True, help="Path to RF geometry JSON")
    parser.add_argument("--pose_csv", required=True, help="Path to 6DoF pose CSV")
    parser.add_argument("--roi_json", required=True, help="Path to ROI JSON or ROI sequence JSON")
    parser.add_argument("--output_npz", required=True, help="Output feature matrix .npz")
    parser.add_argument("--x_range", nargs=2, type=float, required=True)
    parser.add_argument("--y_range", nargs=2, type=float, required=True)
    parser.add_argument("--z_range", nargs=2, type=float, required=True)
    parser.add_argument("--voxel_resolution_m", type=float, default=0.25)
    args = parser.parse_args()

    heatmap_data = _load_heatmap_array(args.heatmap)
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    setting = config["heatmap_setting"]
    tof_s = np.arange(*setting["tau_axis_param"], dtype=float) / 299792458.0
    doppler_hz = np.arange(*setting["f_axis_param"], dtype=float)
    time_s = np.arange(heatmap_data.shape[0], dtype=float) / float(setting["fs"])

    aoa_rad = None
    if heatmap_data.ndim == 4:
        aoa_rad = np.deg2rad(np.arange(*setting["theta_deg_axis_param"], dtype=float))

    heatmap = wrap_existing_heatmap(heatmap_data, time_s, doppler_hz, tof_s=tof_s, aoa_rad=aoa_rad)
    rf_geometry = load_rf_geometry_config(args.rf_geometry)
    poses = load_pose_sequence_csv(args.pose_csv)
    roi_sequence = load_candidate_roi_sequence_json(args.roi_json)
    voxels = build_world_voxel_grid(
        tuple(args.x_range),
        tuple(args.y_range),
        tuple(args.z_range),
        args.voxel_resolution_m,
    )

    previous_blobs = None
    feature_frames = []
    roi_id_frames = []
    feature_names = None
    for timestamp in heatmap.time_s:
        frame_heatmap = wrap_existing_heatmap(
            heatmap.energy[np.newaxis, int(np.argmin(np.abs(heatmap.time_s - timestamp)))],
            np.array([timestamp]),
            heatmap.doppler_hz,
            tof_s=heatmap.tof_s,
            aoa_rad=heatmap.aoa_rad,
        )
        pose = interpolate_pose_at_time(poses, float(timestamp))
        rois = select_rois_at_time(roi_sequence, float(timestamp))
        result = run_rdith_pipeline(
            frame_heatmap,
            pose,
            rois,
            rf_geometry=rf_geometry,
            voxel_positions_xyz=voxels,
            previous_blobs=previous_blobs,
        )
        previous_blobs = result["tracked_blobs"]
        roi_features = result["roi_features"]
        feature_frames.append(np.vstack([item.features for item in roi_features]) if roi_features else np.zeros((0, 0)))
        roi_id_frames.append(np.array([item.roi_id for item in roi_features], dtype=int))
        if roi_features and feature_names is None:
            feature_names = roi_features[0].feature_names

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=np.array(feature_frames, dtype=object),
        roi_ids=np.array(roi_id_frames, dtype=object),
        feature_names=np.array(feature_names or [], dtype=str),
        time_s=heatmap.time_s,
    )


def _load_heatmap_array(path: str) -> np.ndarray:
    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        if "spectrums" in data:
            return np.asarray(data["spectrums"], dtype=float)
        if "spectrum" in data:
            return np.asarray(data["spectrum"], dtype=float)
        first_key = data.files[0]
        return np.asarray(data[first_key], dtype=float)
    if suffix == ".mat":
        try:
            from scipy.io import loadmat
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("scipy is required to load .mat heatmap files") from exc
        data = loadmat(path)
        if "spectrums" in data:
            return np.asarray(data["spectrums"], dtype=float)
        if "spectrum" in data:
            return np.asarray(data["spectrum"], dtype=float)
        raise ValueError(".mat file must contain 'spectrum' or 'spectrums'")
    raise ValueError("heatmap path must end in .npz or .mat")


if __name__ == "__main__":
    main()
