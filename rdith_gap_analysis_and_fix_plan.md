# RDITH Gap Analysis and Fix Plan
## For `3df-csi-hpe-csi-heatmap-generation-tool`

This document records what is currently missing in the uploaded zip and how Codex or another implementation agent should fix it.

The current repository already contains a valid upstream CSI Doppler heatmap generator and a first-pass `rdith/` extension. However, the RDITH part is still mostly a structural scaffold. It matches the intended architecture, but not yet the full scientific / engineering implementation needed for real VR ROI selection.

---

# 1. Current Status Summary

## 1.1 What already exists

The repository contains the original heatmap-generation code:

```text
heatmap.py
utilsforheatmap.py
plot_utils.py
config.json
```

It also contains the RDITH extension modules:

```text
rdith/
  __init__.py
  types.py
  heatmap_adapter.py
  residual.py
  blob_extraction.py
  roi_features.py
  geometry.py
  pipeline.py
  visualization.py
```

The current high-level flow exists:

```text
Existing CSI heatmap
→ wrap_existing_heatmap()
→ estimate_expected_velocity_field()
→ compute_residual_heatmap()
→ threshold_residual_heatmap()
→ extract_rf_blobs()
→ build_roi_feature_vector()
```

This means the code structurally matches the intended design.

---

## 1.2 What is still missing

The current code does **not yet fully implement RDITH**.

It lacks:

1. Real Tx/Rx geometry modeling.
2. True mapping from ToF / AoA / Doppler bins to world-space RF voxels.
3. Correct projection of 6DoF velocity into RF radial / bistatic Doppler space.
4. Calibrated RF blob positions.
5. True ROI-local feature pooling.
6. Real 6DoF pose input loader.
7. Real candidate ROI input loader.
8. Integration between `heatmap.py` output and RDITH pipeline.
9. Strong validation / tests.
10. Evaluation scripts for ROI selection.

The existing RDITH code is therefore best described as:

```text
API skeleton + toy implementation + placeholder geometry
```

not:

```text
complete RDITH research implementation
```

---

# 2. Most Important Conceptual Gap

The intended RDITH contribution is:

```text
RF Doppler motion - 6DoF-explained motion = residual motion intent
```

But the current implementation approximates this too simply.

Current `compute_residual_heatmap()` compares:

```text
Doppler bin value
vs.
expected 6DoF speed magnitude
```

This is not physically correct enough.

The correct implementation should compare motion in the same measurement space:

```text
Observed RF Doppler
vs.
Expected RF Doppler induced by 6DoF motion
```

That requires Tx/Rx geometry and Doppler projection.

---

# 3. Gap-by-Gap Fix Plan

---

# Gap 1: No Real Tx/Rx Geometry Model

## Current State

`config.json` contains RF parameters such as:

```json
{
  "num_rx": 8,
  "num_tx": 2,
  "center_frequency": 5.57e9,
  "antenna_spacing": 0.015
}
```

But it does not define the actual physical placement of Tx/Rx devices in the environment.

Currently, RDITH has no way to know:

- where each transmitter is located,
- where each receiver is located,
- how antenna arrays are oriented,
- what coordinate system is used.

## Why This Matters

Without Tx/Rx geometry, the system cannot correctly convert:

```text
ToF / AoA / Doppler bins
```

into:

```text
world-space 3D motion cells
```

It also cannot compute bistatic Doppler projection.

## Fix

Create a new file:

```text
rdith/config.py
```

Add a dataclass:

```python
@dataclass
class RFGeometryConfig:
    tx_positions_xyz: np.ndarray      # shape: [num_tx, 3]
    rx_positions_xyz: np.ndarray      # shape: [num_rx, 3]
    tx_orientations: Optional[np.ndarray]
    rx_orientations: Optional[np.ndarray]
    center_frequency_hz: float
    speed_of_light_mps: float = 299792458.0
    coordinate_frame: str = "world"
```

Add loader:

```python
def load_rf_geometry_config(path: str) -> RFGeometryConfig:
```

## Expected Input

A new config block, for example:

```json
{
  "rf_geometry": {
    "coordinate_frame": "world",
    "tx_positions_xyz": [[0.0, 0.0, 2.0], [4.0, 0.0, 2.0]],
    "rx_positions_xyz": [[0.0, 4.0, 2.0], [4.0, 4.0, 2.0]],
    "center_frequency_hz": 5.57e9
  }
}
```

## Acceptance Criteria

- The system can load Tx/Rx positions.
- The number of Tx/Rx in config matches upstream CSI dimensions.
- All coordinates are documented as world coordinates.
- Unit tests verify shape validation.

---

# Gap 2: Placeholder Voxel Coordinates

## Current State

`rdith/pipeline.py` currently uses:

```python
def _default_voxel_positions(heatmap: HeatmapTensor) -> np.ndarray:
```

This function approximately maps ToF and AoA to x/y coordinates using:

```python
radius = tof * c / 2
x = radius * cos(aoa)
y = radius * sin(aoa)
z = 0
```

This is only a placeholder.

## Why This Matters

For bistatic or multi-link RF sensing, ToF is not simply:

```text
range = tof * c / 2
```

unless it is a monostatic radar assumption.

For Tx/Rx sensing, the correct relation is:

```text
||x - Tx|| + ||x - Rx|| = c * ToF
```

This forms an ellipsoid, not a simple sphere.

## Fix

Create:

```text
rdith/voxelization.py
```

Add:

```python
def build_world_voxel_grid(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    resolution_m: float,
) -> np.ndarray:
```

Output:

```text
[N_voxel, 3]
```

Add:

```python
def compute_bistatic_tof_for_voxels(
    voxel_positions_xyz: np.ndarray,
    tx_position_xyz: np.ndarray,
    rx_position_xyz: np.ndarray,
    speed_of_light_mps: float = 299792458.0,
) -> np.ndarray:
```

Formula:

```text
tof(x) = (||x - Tx|| + ||x - Rx||) / c
```

Add:

```python
def map_heatmap_bins_to_voxels(
    heatmap: HeatmapTensor,
    rf_geometry: RFGeometryConfig,
    voxel_positions_xyz: np.ndarray,
) -> VoxelRFMap:
```

## New Dataclass

Add to `types.py`:

```python
@dataclass
class VoxelRFMap:
    voxel_positions_xyz: np.ndarray       # [N, 3]
    energy: np.ndarray                    # [T, N]
    observed_doppler_hz: np.ndarray       # [T, N]
    confidence: np.ndarray                # [T, N]
```

## Acceptance Criteria

- No RDITH module should use fake heatmap index as xyz.
- Every RF energy value used by RDITH should have an associated world-space voxel position.
- Unit tests should verify ToF mapping for a simple Tx/Rx geometry.

---

# Gap 3: Incorrect 6DoF Residual Doppler Projection

## Current State

`rdith/residual.py` computes expected 6DoF velocity in xyz:

```python
v_expected = v_head + omega × r
```

This part is reasonable.

But then `compute_residual_heatmap()` reduces this to speed magnitude and compares it directly to Doppler bins.

This is not physically correct.

## Why This Matters

Doppler is not full 3D speed.

Doppler measures the velocity projection along RF propagation geometry.

For bistatic Tx/Rx sensing:

```text
f_d(x) = -(1 / lambda) * v(x) · (u_tx_to_x + u_rx_to_x)
```

where:

```text
u_tx_to_x = unit vector from Tx to voxel
u_rx_to_x = unit vector from Rx to voxel
lambda = c / carrier_frequency
```

## Fix

Modify `residual.py` to add:

```python
def project_velocity_to_bistatic_doppler(
    velocity_xyz: np.ndarray,
    voxel_positions_xyz: np.ndarray,
    tx_position_xyz: np.ndarray,
    rx_position_xyz: np.ndarray,
    carrier_frequency_hz: float,
    speed_of_light_mps: float = 299792458.0,
) -> np.ndarray:
```

Formula:

```text
lambda = c / fc
b(x) = unit(x - tx) + unit(x - rx)
expected_doppler_hz = -(velocity_xyz · b(x)) / lambda
```

Then replace current residual logic with:

```python
residual_doppler_hz = observed_doppler_hz - expected_doppler_hz
residual_energy = energy * normalized_abs(residual_doppler_hz)
```

## Expected New Function

```python
def compute_residual_voxel_map(
    voxel_rf_map: VoxelRFMap,
    pose: Pose6DoF,
    rf_geometry: RFGeometryConfig,
) -> ResidualVoxelMap:
```

## New Dataclass

```python
@dataclass
class ResidualVoxelMap:
    voxel_positions_xyz: np.ndarray       # [N, 3]
    residual_energy: np.ndarray           # [T, N]
    residual_doppler_hz: np.ndarray       # [T, N]
    expected_doppler_hz: np.ndarray       # [T, N]
    observed_doppler_hz: np.ndarray       # [T, N]
    confidence: np.ndarray                # [T, N]
```

## Acceptance Criteria

- Residual is computed in Doppler space, not raw speed magnitude space.
- Unit test: if RF observed Doppler equals 6DoF-projected expected Doppler, residual energy should be near zero.
- Unit test: if RF observed Doppler is different from expected Doppler, residual energy should remain high.

---

# Gap 4: RF Blob Coordinates Are Placeholder Index Values

## Current State

`rdith/blob_extraction.py` uses:

```python
_index_to_placeholder_xyz(index)
```

This converts heatmap indices into fake xyz values.

## Why This Matters

ROI features such as:

- distance to ROI,
- motion-to-ROI alignment,
- time-to-contact,

are invalid if blob positions are fake.

## Fix

Change blob extraction to operate on `ResidualVoxelMap`, not raw dense heatmap index space.

New signature:

```python
def extract_rf_blobs_from_voxels(
    residual_map: ResidualVoxelMap,
    threshold: float,
    distance_eps_m: float,
    min_samples: int,
) -> list[RFBlob]:
```

Suggested algorithm:

1. Select active voxels:

```text
residual_energy[t, n] >= threshold
```

2. Cluster active world coordinates using DBSCAN or connected voxel neighborhoods.
3. Compute blob centroid by energy-weighted average:

```text
centroid = sum(x_n * energy_n) / sum(energy_n)
```

4. Compute blob velocity from Doppler projection or temporal centroid differences.

## Acceptance Criteria

- `RFBlob.centroid_xyz` is in world coordinates.
- No blob uses array index as xyz.
- ROI distance functions produce values in meters.

---

# Gap 5: ROI Motion Energy Is Global, Not ROI-Local

## Current State

`compute_roi_motion_energy()` currently returns:

```python
float(np.mean(residual_energy))
```

This is global heatmap energy, not ROI-local energy.

## Why This Matters

Every ROI may receive almost the same RF energy feature.

That destroys the usefulness of ROI selection.

ROI feature extraction must answer:

```text
How much residual RF motion is near this specific ROI?
```

## Fix

Update function signature:

```python
def compute_roi_motion_energy(
    roi: CandidateROI,
    residual_map: ResidualVoxelMap,
    radius_m: float,
) -> float:
```

Implementation:

```python
distances = distance(voxel_positions_xyz, roi.center_xyz or ROI bbox)
weights = exp(-distances^2 / (2 * sigma^2))
energy = sum(weights * residual_energy) / sum(weights)
```

or bbox pooling:

```python
select voxels inside ROI bbox expanded by margin
sum residual energy
```

## Acceptance Criteria

- Different ROIs get different RF energy values when motion is spatially localized.
- Test with synthetic residual map: a hot voxel near ROI A should give ROI A higher energy than ROI B.

---

# Gap 6: Micro-Doppler Bandwidth Is Global, Not ROI-Local

## Current State

`compute_microdoppler_bandwidth()` sums the whole heatmap over time and spatial axes.

It ignores the ROI argument except for the function signature.

## Why This Matters

A global Doppler bandwidth cannot tell whether a specific ROI is associated with hand motion or irrelevant room motion.

## Fix

Use ROI-local voxel selection first.

New signature:

```python
def compute_roi_microdoppler_bandwidth(
    roi: CandidateROI,
    voxel_rf_map: VoxelRFMap,
    radius_m: float,
) -> float:
```

Implementation:

1. Select voxels near ROI.
2. Build Doppler profile only from those voxels or from heatmap bins contributing to those voxels.
3. Compute weighted Doppler standard deviation:

```text
bandwidth = sqrt(sum((fd - mean_fd)^2 * E_fd) / sum(E_fd))
```

## Acceptance Criteria

- Bandwidth changes depending on which ROI is queried.
- Synthetic test: ROI near high-Doppler spread region should have higher bandwidth.

---

# Gap 7: Time-to-Contact Depends on Weak Velocity Estimate

## Current State

`RFBlob.velocity_xyz` is currently approximated as:

```python
[velocity_scalar, 0, 0]
```

where `velocity_scalar` comes from residual velocity values.

This is not a true 3D velocity vector.

## Why This Matters

`compute_time_to_contact()` uses:

```python
closing_speed = dot(blob.velocity_xyz, direction_to_roi)
```

If velocity is fake, time-to-contact is fake.

## Fix

Implement at least one of these velocity estimators:

### Option A: Temporal centroid velocity

Track blob centroids over time:

```text
v_blob = (centroid_t - centroid_t-1) / dt
```

This is simple and robust.

### Option B: Multi-link Doppler velocity reconstruction

Use multiple Tx/Rx Doppler projections:

```text
fd_l = -(1/lambda) * b_l(x)^T v
```

Solve least squares:

```text
v = argmin_v Σ_l ||fd_l - predicted_fd_l||²
```

### Recommendation

Start with Option A, then add Option B when multi-link geometry is ready.

## Acceptance Criteria

- `RFBlob.velocity_xyz` is in meters/second.
- `time_to_contact` is in seconds.
- Moving blob toward ROI gives finite TTC.
- Moving blob away from ROI gives infinite or clipped TTC.

---

# Gap 8: No Real 6DoF Pose Loader

## Current State

The code defines:

```python
Pose6DoF
```

but there is no loader for real headset pose data.

## Why This Matters

Without real 6DoF input, residualization cannot be evaluated.

## Fix

Create:

```text
rdith/io.py
```

Add:

```python
def load_pose_sequence_csv(path: str) -> list[Pose6DoF]:
```

Expected CSV columns:

```text
timestamp_s,
position_x, position_y, position_z,
rotation_00, rotation_01, ..., rotation_22,
linear_velocity_x, linear_velocity_y, linear_velocity_z,
angular_velocity_x, angular_velocity_y, angular_velocity_z
```

Also add interpolation:

```python
def interpolate_pose_at_time(
    poses: list[Pose6DoF],
    timestamp_s: float,
) -> Pose6DoF:
```

## Acceptance Criteria

- Pose can be aligned to heatmap timestamps.
- Missing velocity can be estimated from neighboring poses.
- Rotation matrices are validated or normalized.

---

# Gap 9: No Candidate ROI Loader

## Current State

The code defines:

```python
CandidateROI
```

but no data loader exists.

## Why This Matters

ROI features cannot be used unless candidate ROIs from frustum / occlusion culling are provided.

## Fix

Add to `rdith/io.py`:

```python
def load_candidate_rois_json(path: str) -> list[CandidateROI]:
```

Expected JSON:

```json
[
  {
    "roi_id": 1,
    "center_xyz": [1.0, 0.5, 1.2],
    "bbox_extent_xyz": [0.4, 0.4, 0.4],
    "visibility_score": 0.8,
    "occlusion_score": 0.1
  }
]
```

For time-varying ROI:

```python
def load_candidate_roi_sequence_json(path: str) -> dict[float, list[CandidateROI]]:
```

## Acceptance Criteria

- ROIs can be loaded per timestamp.
- ROI coordinates use the same world frame as RF voxels.
- Visibility and occlusion scores are clipped to `[0, 1]`.

---

# Gap 10: Heatmap Generator Is Not Connected to RDITH Pipeline

## Current State

`heatmap.py` generates spectrums and saves `.mat` / figures.

It does not call:

```python
run_rdith_pipeline()
```

## Why This Matters

The current system is two separate pieces:

```text
heatmap generation
RDITH extension
```

They need an experiment-level runner.

## Fix

Do not force RDITH into `heatmap.py` immediately.

Instead create:

```text
experiments/run_rdith_from_heatmap.py
```

Responsibilities:

1. Load saved heatmap `.mat` or `.npz`.
2. Build axis arrays from `config.json`.
3. Wrap into `HeatmapTensor`.
4. Load RF geometry config.
5. Load 6DoF pose sequence.
6. Load candidate ROI sequence.
7. Run RDITH per timestamp/window.
8. Save ROI features.

Suggested CLI:

```bash
python experiments/run_rdith_from_heatmap.py \
  --heatmap_mat path/to/smoothed_CSI_avg.mat \
  --config config.json \
  --rf_geometry rf_geometry.json \
  --pose_csv pose.csv \
  --roi_json rois.json \
  --output_npz output/roi_features.npz
```

## Acceptance Criteria

- One command can run heatmap output through RDITH.
- Output contains feature matrix shaped like:

```text
[num_timestamps, num_rois, num_features]
```

---

# Gap 11: No Feature Names / Feature Schema

## Current State

`ROIFeatureVector.features` is a raw numpy array.

There is no official feature order except implicit code order.

## Why This Matters

ML training and debugging become fragile.

## Fix

Add to `roi_features.py`:

```python
ROI_FEATURE_NAMES = [
    "roi_motion_energy",
    "nearest_blob_energy",
    "motion_to_roi_alignment",
    "time_to_contact_s",
    "doppler_entropy",
    "microdoppler_bandwidth_hz",
    "nearest_blob_confidence",
    "visibility_conflict_score",
    "temporal_growth",
]
```

Better: update dataclass:

```python
@dataclass
class ROIFeatureVector:
    roi_id: int
    feature_names: list[str]
    features: np.ndarray
```

## Acceptance Criteria

- Every exported feature matrix includes feature names.
- Feature order is stable and documented.

---

# Gap 12: Infinite Feature Values Are Not ML-Safe

## Current State

`compute_time_to_contact()` may return:

```python
float("inf")
```

## Why This Matters

Many ML models cannot safely consume `inf` values.

## Fix

Clip or transform TTC:

```python
max_ttc_s = 10.0
if not finite:
    ttc = max_ttc_s
```

Add inverse TTC feature:

```python
ttc_score = exp(-ttc / tau)
```

Recommended features:

```text
time_to_contact_s_clipped
time_to_contact_score
```

## Acceptance Criteria

- Feature vectors contain no NaN or Inf.
- Add `validate_feature_vector()`.

---

# Gap 13: No Tests

## Current State

No tests are included.

## Why This Matters

The project contains geometry-heavy calculations. Silent coordinate mistakes are very likely.

## Fix

Create:

```text
tests/
  test_heatmap_adapter.py
  test_geometry.py
  test_residual.py
  test_roi_features.py
  test_pipeline_smoke.py
```

Minimum tests:

1. Axis ordering conversion.
2. Point-to-ROI distance.
3. 6DoF rigid velocity field.
4. Bistatic Doppler projection.
5. Residual becomes low when expected and observed Doppler match.
6. ROI-local pooling prefers nearby hot voxels.
7. Pipeline returns finite ML-safe features.

## Acceptance Criteria

Run:

```bash
pytest
```

and all tests pass.

---

# Gap 14: Visualization Is Too Generic

## Current State

`visualization.py` can show heatmap, blobs, and features.

But it does not overlay:

- Tx/Rx positions,
- headset pose,
- candidate ROIs,
- residual motion vectors.

## Why This Matters

RDITH is difficult to debug without spatial visualization.

## Fix

Add:

```python
def visualize_scene_rdith(
    rf_geometry: RFGeometryConfig,
    pose: Pose6DoF,
    rois: list[CandidateROI],
    blobs: list[RFBlob],
    residual_map: ResidualVoxelMap,
):
```

This should show:

- Tx points,
- Rx points,
- HMD position and forward vector,
- ROI boxes,
- RF blob centroids,
- RF velocity arrows,
- residual voxel energy.

## Acceptance Criteria

- One plot can visually confirm whether RF blobs are near correct ROIs.
- Debug images can be saved during experiments.

---

# Gap 15: No Evaluation Pipeline

## Current State

No evaluation code exists for ROI prediction.

## Why This Matters

The research goal is not heatmap generation alone. It is ROI selection improvement.

## Fix

Create:

```text
experiments/evaluate_roi_prediction.py
```

Inputs:

```text
ROI feature matrix
ROI ground truth labels
baseline 6DoF features
```

Metrics:

- top-k ROI recall,
- ROI F1,
- future ROI IoU,
- early prediction gain,
- rendering / bandwidth saving proxy,
- false high-priority ROI ratio.

Baselines:

```text
A: 6DoF only
B: 6DoF + frustum / occlusion
C: 6DoF + raw Doppler summary
D: 6DoF + normal Doppler heatmap features
E: Proposed RDITH features
```

## Acceptance Criteria

- The proposed RDITH feature model can be compared against at least two baselines.
- Evaluation script saves metrics as JSON and plots as PNG.

---

# 4. Required Final Pipeline After Fixes

The final intended pipeline should be:

```text
1. Run existing CSI heatmap generator.

2. Load generated Doppler / ToF-Doppler / AoA-ToF-Doppler spectrum.

3. Wrap spectrum using HeatmapTensor.

4. Load RF geometry.

5. Build calibrated world-space voxel grid.

6. Map heatmap bins to voxel RF map.

7. Load HMD 6DoF pose aligned to heatmap timestamp.

8. Compute headset-induced velocity field.

9. Project headset-induced velocity field into bistatic Doppler space.

10. Compute residual Doppler map.

11. Threshold residual energy.

12. Extract RF blobs in world coordinates.

13. Track RF blobs over time.

14. Load candidate ROI set from frustum / occlusion module.

15. Pool RF residual features around each ROI.

16. Export ML-safe ROI feature matrix.

17. Train / evaluate ROI selection model.
```

---

# 5. Files That Should Be Added

Recommended new files:

```text
rdith/config.py
rdith/io.py
rdith/voxelization.py
rdith/doppler_projection.py
rdith/validation.py
rdith/feature_schema.py
rdith/tracking.py
experiments/run_rdith_from_heatmap.py
experiments/evaluate_roi_prediction.py
tests/test_heatmap_adapter.py
tests/test_geometry.py
tests/test_residual.py
tests/test_roi_features.py
tests/test_pipeline_smoke.py
```

---

# 6. Files That Should Be Modified

```text
rdith/types.py
rdith/residual.py
rdith/blob_extraction.py
rdith/roi_features.py
rdith/pipeline.py
rdith/visualization.py
config.json
README.md
requirements.txt
```

---

# 7. Priority Order for Codex

## Priority 1: Scientific Correctness

Implement:

1. RF geometry config.
2. World voxel grid.
3. Bistatic Doppler projection.
4. Residual Doppler map.

Without this, RDITH is not physically correct.

---

## Priority 2: ROI Correctness

Implement:

1. ROI-local pooling.
2. Calibrated RF blob centroids.
3. Real motion-to-ROI alignment.
4. ML-safe feature vector.

Without this, features are not meaningful for ROI selection.

---

## Priority 3: Data Integration

Implement:

1. Pose loader.
2. ROI loader.
3. Heatmap-to-RDITH runner.
4. Export feature matrix.

Without this, experiments cannot run end to end.

---

## Priority 4: Validation and Evaluation

Implement:

1. Unit tests.
2. Visualization.
3. Ablation evaluation.

Without this, Codex cannot prove the implementation is correct.

---

# 8. What Codex Should NOT Do

Codex should NOT rewrite the existing MUSIC / CSI heatmap generator unless absolutely necessary.

Do not replace:

```text
CSI_preprocessing()
create_steering_matrix_ToF_Doppler()
create_steering_matrix_F3D()
run_music_algorithm()
pipeline_3D()
```

Those are upstream sensing functions.

The intended work is downstream:

```text
existing Doppler heatmap
→ physically calibrated RDITH
→ ROI features
```

---

# 9. Definition of Done

The implementation is acceptable only when all of the following are true:

1. Existing heatmap generation still works.
2. RDITH can load a saved heatmap result.
3. RDITH can load RF geometry.
4. RDITH can load 6DoF pose.
5. RDITH can load candidate ROIs.
6. RDITH maps heatmap data into world-space voxels.
7. RDITH computes residual Doppler using RF projection, not speed magnitude proxy.
8. RF blobs have real world-space centroids.
9. ROI features are spatially local to each ROI.
10. Feature vectors contain no NaN or Inf.
11. Pipeline can export feature matrices for ML.
12. Tests cover geometry, residualization, and ROI pooling.
13. Visualization can show Tx/Rx, HMD, RF blobs, and ROIs together.
14. Evaluation script can compare RDITH against at least one baseline.

---

# 10. Final Summary

The uploaded zip matches the intended RDITH architecture, but only as a first-pass scaffold.

The existing Doppler heatmap generator is real and should be preserved.

The RDITH extension still needs to be upgraded from placeholder implementation to physically meaningful implementation by adding:

```text
RF geometry
+ voxel calibration
+ bistatic Doppler projection
+ real residualization
+ world-space blob extraction
+ ROI-local pooling
+ data loaders
+ evaluation
```

The most important fix is:

```text
Do not compare 6DoF speed magnitude to Doppler bins.
Instead, project 6DoF-induced velocity into the same RF Doppler measurement space and subtract there.
```

That is the difference between a toy scaffold and the actual RDITH research contribution.
