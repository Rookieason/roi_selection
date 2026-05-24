import numpy as np

from .types import ROIFeatureVector


def validate_feature_vector(feature_vector: ROIFeatureVector) -> None:
    features = np.asarray(feature_vector.features, dtype=float)
    if features.ndim != 1:
        raise ValueError("ROI features must be a 1D vector")
    if len(feature_vector.feature_names) != features.size:
        raise ValueError("feature_names length must match features length")
    if not np.all(np.isfinite(features)):
        raise ValueError("ROI feature vector contains NaN or Inf")
