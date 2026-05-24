import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ROI prediction scores from RDITH features.")
    parser.add_argument("--features_npz", required=True)
    parser.add_argument("--labels_npz", required=True, help="NPZ with labels shaped like features frame/ROI layout")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--top_k", type=int, default=1)
    args = parser.parse_args()

    feature_data = np.load(args.features_npz, allow_pickle=True)
    label_data = np.load(args.labels_npz, allow_pickle=True)
    feature_names = list(feature_data["feature_names"])
    labels = label_data["labels"]

    metrics = {
        "rdith": _evaluate_scores(_score_rdith(feature_data["features"], feature_names), labels, args.top_k),
        "baseline_visibility": _evaluate_scores(_score_visibility(feature_data["features"], feature_names), labels, args.top_k),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def _score_rdith(feature_frames: np.ndarray, feature_names: list[str]) -> list[np.ndarray]:
    energy_id = feature_names.index("roi_motion_energy")
    alignment_id = feature_names.index("motion_to_roi_alignment")
    ttc_score_id = feature_names.index("time_to_contact_score")
    conflict_id = feature_names.index("visibility_conflict_score")
    return [
        frame[:, energy_id] + 0.5 * frame[:, alignment_id] + frame[:, ttc_score_id] + 0.25 * frame[:, conflict_id]
        for frame in feature_frames
    ]


def _score_visibility(feature_frames: np.ndarray, feature_names: list[str]) -> list[np.ndarray]:
    conflict_id = feature_names.index("visibility_conflict_score")
    return [-frame[:, conflict_id] for frame in feature_frames]


def _evaluate_scores(score_frames: list[np.ndarray], label_frames: np.ndarray, top_k: int) -> dict[str, float]:
    hits = 0
    positives = 0
    false_high_priority = 0
    selected = 0
    for scores, labels in zip(score_frames, label_frames):
        labels = np.asarray(labels, dtype=int)
        if scores.size == 0 or labels.size == 0:
            continue
        k = min(top_k, scores.size)
        top_ids = np.argsort(scores)[-k:]
        hits += int(np.any(labels[top_ids] > 0))
        positives += int(np.any(labels > 0))
        false_high_priority += int(np.sum(labels[top_ids] == 0))
        selected += k
    recall = hits / positives if positives else 0.0
    false_high_priority_ratio = false_high_priority / selected if selected else 0.0
    return {
        "top_k": float(top_k),
        "top_k_recall": float(recall),
        "false_high_priority_roi_ratio": float(false_high_priority_ratio),
    }


if __name__ == "__main__":
    main()
