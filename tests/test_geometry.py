import numpy as np

from rdith.geometry import compute_distance_point_to_roi, is_point_inside_roi
from rdith.types import CandidateROI


def test_point_to_roi_distance_uses_bbox_extent():
    roi = CandidateROI(1, np.zeros(3), np.array([2.0, 2.0, 2.0]), 1.0, 0.0)
    assert compute_distance_point_to_roi(np.array([0.5, 0.0, 0.0]), roi) == 0.0
    assert np.isclose(compute_distance_point_to_roi(np.array([2.0, 0.0, 0.0]), roi), 1.0)


def test_point_inside_roi():
    roi = CandidateROI(1, np.zeros(3), np.array([2.0, 2.0, 2.0]), 1.0, 0.0)
    assert is_point_inside_roi(np.array([0.9, 0.0, 0.0]), roi)
    assert not is_point_inside_roi(np.array([1.1, 0.0, 0.0]), roi)
