# 3DF-CSI-HPE CSI Heatmap Generation Tool

This repository contains the upstream CSI Doppler heatmap generator and a downstream RDITH extension for VR ROI feature generation.

The upstream files remain responsible for CSI preprocessing and MUSIC spectrum generation:

```text
heatmap.py
utilsforheatmap.py
plot_utils.py
```

The RDITH package converts saved heatmaps into calibrated residual RF motion features:

```text
existing heatmap -> RF voxel map -> 6DoF residual Doppler -> RF blobs -> ROI feature vectors
```

## RDITH Inputs

RDITH expects:

- A saved ToF-Doppler or AoA-ToF-Doppler heatmap (`.npz` or `.mat`)
- RF Tx/Rx geometry in world coordinates
- A 6DoF pose CSV
- Candidate ROI JSON
- A world voxel grid or voxel bounds

`config.json` includes an example `rf_geometry` block. Replace those Tx/Rx coordinates with calibrated environment coordinates before running real experiments.

## Run RDITH From A Saved Heatmap

```bash
python experiments/run_rdith_from_heatmap.py \
  --heatmap path/to/heatmap.npz \
  --config config.json \
  --rf_geometry config.json \
  --pose_csv path/to/pose.csv \
  --roi_json path/to/rois.json \
  --output_npz output/roi_features.npz \
  --x_range -1 5 \
  --y_range -1 5 \
  --z_range 0 3 \
  --voxel_resolution_m 0.25
```

The output `.npz` contains `features`, `roi_ids`, `feature_names`, and `time_s`.

## Test

```bash
python -m pytest
```

The tests cover heatmap axis adaptation, ROI geometry, bistatic Doppler projection, residualization, ROI-local pooling, and a calibrated pipeline smoke test.
