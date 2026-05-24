# RDITH Project Master Plan
## Residual Doppler Intent Tomographic Heatmap for VR ROI Selection

---

# 1. Project Goal

This project aims to improve ROI (Region of Interest) selection for VR systems by integrating RF sensing information into the ROI prediction pipeline.

Current pipeline:

```text
Current 6DoF
    → Frustum / Occlusion Culling
    → ML Model
    → ROI Selection
```

Proposed pipeline:

```text
Current 6DoF
    → Frustum / Occlusion Culling
    → Candidate ROI Set

Tx/Rx RF Signal
    → Doppler Heatmap Generation
    → Residual Doppler Intent Heatmap (RDITH)
    → RF Blob Extraction
    → RF-ROIAlign Feature Extraction

Candidate ROI + 6DoF Features + RF Features
    → ML Model
    → ROI Selection
```

The key idea is:

> 6DoF describes where the head is currently looking.
> RF Doppler sensing describes body/environment motion that may predict future attention or interaction.

The project does NOT replace the existing Doppler heatmap generator.
Instead:

```text
Existing heatmap generator
    = upstream sensing module

RDITH extension
    = downstream semantic / ROI-oriented motion interpretation module
```

---

# 2. Core Research Insight

The existing Doppler heatmap only answers:

```text
Where is motion happening?
```

The proposed RDITH answers:

```text
Which motion is NOT already explained by headset 6DoF,
and which motion is relevant to future ROI selection?
```

This is the key novelty.

---

# 3. Objective Difference Between Existing Heatmap Generator and RDITH

## Existing Heatmap Generator

The existing codebase already performs:

```text
CSI
→ preprocessing
→ Doppler / ToF-Doppler / AoA-ToF-Doppler MUSIC spectrum
→ heatmap visualization
```

Its purpose:

- sensing
- localization
- motion visualization
- activity recognition
- Doppler spectrum estimation

Outputs are mainly:

- heatmap tensors
- spectrum energy
- Doppler spectrum
- AoA spectrum
- ToF spectrum

It is a standard Doppler heatmap generation system.

---

## RDITH Extension

RDITH adds:

### 1. 6DoF-aware motion interpretation

Subtract motion already explained by:

- headset translation
- headset rotation
- rigid body head motion

---

### 2. ROI-oriented feature extraction

The system no longer only generates heatmaps.

It extracts:

- motion-to-ROI alignment
- residual motion energy
- time-to-contact
- visibility conflict
- micro-Doppler complexity
- interaction intent

---

### 3. Physical-intent representation

The heatmap becomes:

```text
Motion Intent Prior
```

instead of:

```text
Pure sensing output
```

---

# 4. High-Level Architecture

```text
                ┌────────────────────┐
                │ Existing Doppler   │
                │ Heatmap Generator  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Heatmap Adapter    │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ 6DoF Residualizer  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ RF Blob Extraction │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ RF-ROIAlign        │
                │ Feature Extraction │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ ROI ML Model       │
                └────────────────────┘
```

---

# 5. Recommended Repository Structure

```text
project_root/
│
├── existing_heatmap_code/
│
├── rdith/
│   ├── __init__.py
│   ├── types.py
│   ├── heatmap_adapter.py
│   ├── residual.py
│   ├── blob_extraction.py
│   ├── roi_features.py
│   ├── geometry.py
│   ├── transforms.py
│   ├── tracking.py
│   ├── visualization.py
│   ├── config.py
│   └── pipeline.py
│
├── experiments/
│   ├── run_rdith.py
│   ├── evaluate_rdith.py
│   └── benchmark.py
│
├── models/
│   └── roi_model/
│
└── docs/
    └── rdith_design.md
```

---

# 6. Core Data Definitions

---

## 6.1 HeatmapTensor

Represents existing Doppler heatmap output.

```python
@dataclass
class HeatmapTensor:
    energy: np.ndarray
    time_s: np.ndarray
    doppler_hz: np.ndarray
    tof_s: Optional[np.ndarray]
    aoa_rad: Optional[np.ndarray]
```

---

### Meaning

Represents:

```text
Energy(time, doppler, tof, aoa)
```

This is the direct output from the existing heatmap generator.

---

### Field Definitions

| Field | Meaning |
|---|---|
| energy | Heatmap tensor |
| time_s | Time axis |
| doppler_hz | Doppler axis |
| tof_s | ToF axis |
| aoa_rad | AoA axis |

---

# 6.2 Pose6DoF

```python
@dataclass
class Pose6DoF:
    timestamp_s: float
    position_xyz: np.ndarray
    rotation_matrix: np.ndarray
    linear_velocity_xyz: np.ndarray
    angular_velocity_xyz: np.ndarray
```

---

### Meaning

Represents headset state.

Includes:

- position
- orientation
- linear velocity
- angular velocity

---

# 6.3 CandidateROI

```python
@dataclass
class CandidateROI:
    roi_id: int
    center_xyz: np.ndarray
    bbox_extent_xyz: np.ndarray
    visibility_score: float
    occlusion_score: float
```

---

### Meaning

Represents ROI candidates after:

```text
Frustum Culling
+ Occlusion Culling
```

---

# 6.4 ResidualHeatmap

```python
@dataclass
class ResidualHeatmap:
    residual_energy: np.ndarray
    residual_velocity: np.ndarray
    confidence: np.ndarray
```

---

### Meaning

Represents:

```text
Motion not explained by headset motion
```

This is the core RDITH representation.

---

# 6.5 RFBlob

```python
@dataclass
class RFBlob:
    centroid_xyz: np.ndarray
    velocity_xyz: np.ndarray
    energy: float
    doppler_bandwidth: float
    confidence: float
    lifetime_frames: int
```

---

### Meaning

Represents sparse motion regions extracted from RF heatmap.

Equivalent to:

```text
Motion object hypothesis
```

---

# 6.6 ROIFeatureVector

```python
@dataclass
class ROIFeatureVector:
    roi_id: int
    features: np.ndarray
```

---

### Meaning

Final feature vector fed into ML model.

---

# 7. Function-Level Design

This section defines all required functions.

The implementation MUST follow these interfaces.

---

# 7.1 Heatmap Adapter

File:

```text
rdith/heatmap_adapter.py
```

---

## Function

```python
def wrap_existing_heatmap(
    energy: np.ndarray,
    time_s: np.ndarray,
    doppler_hz: np.ndarray,
    tof_s: Optional[np.ndarray] = None,
    aoa_rad: Optional[np.ndarray] = None,
) -> HeatmapTensor:
```

---

### Input

| Input | Meaning |
|---|---|
| energy | Existing heatmap tensor |
| time_s | Time axis |
| doppler_hz | Doppler bins |
| tof_s | ToF bins |
| aoa_rad | AoA bins |

---

### Output

```python
HeatmapTensor
```

---

### Purpose

Converts existing codebase outputs into unified RDITH format.

This function should NOT modify existing heatmap logic.

It is only an adapter layer.

---

# 7.2 6DoF Residual Estimation

File:

```text
rdith/residual.py
```

---

## Function

```python
def estimate_expected_velocity_field(
    pose: Pose6DoF,
    voxel_positions_xyz: np.ndarray,
) -> np.ndarray:
```

---

### Input

| Input | Meaning |
|---|---|
| pose | Current headset pose |
| voxel_positions_xyz | 3D voxel coordinates |

---

### Output

```python
expected_velocity_xyz
```

Shape:

```text
[N_voxel, 3]
```

---

### Purpose

Estimates motion field caused ONLY by headset rigid-body motion.

Formula:

```text
v_expected = v_head + omega × r
```

---

## Function

```python
def compute_residual_heatmap(
    heatmap: HeatmapTensor,
    expected_velocity_xyz: np.ndarray,
) -> ResidualHeatmap:
```

---

### Input

| Input | Meaning |
|---|---|
| heatmap | Existing Doppler heatmap |
| expected_velocity_xyz | Motion predicted by headset |

---

### Output

```python
ResidualHeatmap
```

---

### Purpose

Computes:

```text
RF motion - 6DoF motion
```

This is the core novelty.

---

# 7.3 RF Blob Extraction

File:

```text
rdith/blob_extraction.py
```

---

## Function

```python
def threshold_residual_heatmap(
    residual_heatmap: ResidualHeatmap,
    threshold: float,
) -> np.ndarray:
```

---

### Output

Binary mask.

---

### Purpose

Suppress weak/noisy regions.

---

## Function

```python
def extract_rf_blobs(
    residual_heatmap: ResidualHeatmap,
    binary_mask: np.ndarray,
) -> List[RFBlob]:
```

---

### Purpose

Converts dense heatmap into sparse motion hypotheses.

Recommended methods:

- connected components
- DBSCAN
- mean shift

---

## Function

```python
def track_rf_blobs(
    blobs: List[RFBlob],
    previous_blobs: List[RFBlob],
) -> List[RFBlob]:
```

---

### Purpose

Maintains temporal consistency.

Recommended:

- Kalman filter
- Hungarian matching
- particle filter

---

# 7.4 ROI Feature Extraction

File:

```text
rdith/roi_features.py
```

---

## Function

```python
def compute_roi_motion_energy(
    roi: CandidateROI,
    residual_heatmap: ResidualHeatmap,
) -> float:
```

---

### Meaning

Measures motion energy near ROI.

---

## Function

```python
def compute_motion_to_roi_alignment(
    roi: CandidateROI,
    blob: RFBlob,
) -> float:
```

---

### Meaning

Measures whether motion is moving toward ROI.

---

## Function

```python
def compute_time_to_contact(
    roi: CandidateROI,
    blob: RFBlob,
) -> float:
```

---

### Meaning

Predicts interaction arrival time.

---

## Function

```python
def compute_microdoppler_bandwidth(
    heatmap: HeatmapTensor,
    roi: CandidateROI,
) -> float:
```

---

### Meaning

Measures motion articulation complexity.

Useful for:

- hand movement
- finger movement
- interaction intent

---

## Function

```python
def compute_visibility_conflict_score(
    roi: CandidateROI,
    residual_energy: float,
) -> float:
```

---

### Meaning

Measures:

```text
Strong RF motion
BUT
low visual visibility
```

This is useful for:

- prefetching
- future ROI prediction

---

## Function

```python
def build_roi_feature_vector(
    roi: CandidateROI,
    blobs: List[RFBlob],
    residual_heatmap: ResidualHeatmap,
    heatmap: HeatmapTensor,
) -> ROIFeatureVector:
```

---

### Purpose

Constructs final ML feature vector.

---

### Suggested Features

```text
[
    RF motion energy,
    residual energy,
    motion-to-ROI alignment,
    time-to-contact,
    Doppler entropy,
    micro-Doppler bandwidth,
    confidence,
    visibility conflict,
    temporal growth,
]
```

---

# 7.5 Geometry Utilities

File:

```text
rdith/geometry.py
```

---

## Function

```python
def compute_distance_point_to_roi(
    point_xyz: np.ndarray,
    roi: CandidateROI,
) -> float:
```

---

## Function

```python
def compute_roi_direction_vector(
    source_xyz: np.ndarray,
    roi: CandidateROI,
) -> np.ndarray:
```

---

## Function

```python
def is_point_inside_roi(
    point_xyz: np.ndarray,
    roi: CandidateROI,
) -> bool:
```

---

# 7.6 Visualization

File:

```text
rdith/visualization.py
```

---

## Function

```python
def visualize_residual_heatmap(
    residual_heatmap: ResidualHeatmap,
):
```

---

## Function

```python
def visualize_rf_blobs(
    blobs: List[RFBlob],
):
```

---

## Function

```python
def visualize_roi_features(
    roi_features: List[ROIFeatureVector],
):
```

---

# 7.7 Main Pipeline

File:

```text
rdith/pipeline.py
```

---

## Function

```python
def run_rdith_pipeline(
    heatmap: HeatmapTensor,
    pose: Pose6DoF,
    candidate_rois: List[CandidateROI],
    previous_blobs: Optional[List[RFBlob]] = None,
):
```

---

### Pipeline

```text
1. Estimate expected motion from 6DoF
2. Compute residual heatmap
3. Threshold residual heatmap
4. Extract RF blobs
5. Track RF blobs
6. Extract ROI features
7. Return ROI feature vectors
```

---

### Output

```python
{
    "residual_heatmap": ResidualHeatmap,
    "tracked_blobs": List[RFBlob],
    "roi_features": List[ROIFeatureVector],
}
```

---

# 8. Important Engineering Constraints

---

# 8.1 DO NOT Modify Existing Doppler Generator Core

The current heatmap generator is already a correct standard sensing pipeline.

Do NOT:

- rewrite MUSIC
- rewrite CSI preprocessing
- rewrite ToF/Doppler estimation
- rewrite AoA estimation

RDITH should be implemented as:

```text
Downstream extension layer
```

NOT:

```text
Replacement
```

---

# 8.2 Residualization is the Core Contribution

The MOST important part is:

```text
RF motion
MINUS
6DoF-explained motion
```

This is the primary novelty.

Everything should preserve this design philosophy.

---

# 8.3 Avoid Raw Heatmap ML Input

DO NOT directly feed:

```text
Raw Doppler tensor
```

into ML model.

Reason:

- too large
- noisy
- difficult to interpret
- poor explainability
- unstable across environments

Instead:

```text
Heatmap
→ sparse motion representation
→ ROI-aligned features
→ ML model
```

---

# 8.4 Sparse Representation is Important

The system should eventually operate in real time.

Therefore:

DO NOT process full dense voxel grids unnecessarily.

Preferred pipeline:

```text
Thresholding
→ sparse blob extraction
→ tracking
→ ROI pooling
```

---

# 8.5 Coordinate System Consistency is Critical

All modules MUST use consistent coordinate systems.

Recommended:

```text
World Coordinate System
```

with:

```text
Headset pose as transform
```

Avoid mixing:

- local RF coordinates
- camera coordinates
- headset coordinates
- world coordinates

without explicit transforms.

---

# 8.6 Visualization is Extremely Important

The project MUST include visualization tools.

Reason:

Residual motion interpretation is difficult to debug numerically.

Required visualizations:

- raw heatmap
- residual heatmap
- extracted blobs
- ROI overlays
- motion vectors
- temporal tracking

---

# 8.7 The Goal is ROI Prediction, NOT Human Sensing Alone

This project is NOT only:

- HAR
- localization
- pose estimation
- RF imaging

Those are upstream sensing tasks.

The actual downstream target is:

```text
ROI selection improvement
```

All feature design should optimize:

```text
Future attention / future interaction prediction
```

NOT merely sensing accuracy.

---

# 9. Suggested Experimental Stages

---

# Stage 1

Validate:

```text
Existing heatmap generator works correctly
```

---

# Stage 2

Implement:

```text
Heatmap adapter
+ residual heatmap
```

Validate:

```text
Head motion is properly removed
```

---

# Stage 3

Implement:

```text
RF blob extraction
```

Validate:

```text
Motion regions are stable
```

---

# Stage 4

Implement:

```text
ROI-aligned features
```

Validate:

```text
Interaction intent correlates with ROI
```

---

# Stage 5

Train:

```text
ROI prediction model
```

---

# 10. Suggested Ablation Study

| Model | Inputs |
|---|---|
| Baseline A | 6DoF only |
| Baseline B | 6DoF + frustum |
| Baseline C | 6DoF + raw Doppler |
| Baseline D | 6DoF + normal Doppler heatmap |
| Proposed | 6DoF + residual RF ROI features |

---

# 11. Final Design Philosophy

This project should always follow:

```text
Sensing
→ residualization
→ sparse semantic motion
→ ROI-oriented representation
→ prediction
```

NOT:

```text
Raw RF tensor
→ giant black-box model
```

The project must remain:

- interpretable
- physically meaningful
- geometrically consistent
- ROI-oriented
- extensible
- real-time feasible

---

# 12. Final One-Sentence Summary

RDITH transforms standard Doppler heatmaps into a physically explainable motion-intent prior for VR ROI selection by removing 6DoF-explained motion and extracting ROI-oriented RF interaction features.

