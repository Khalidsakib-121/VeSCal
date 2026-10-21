"""Fast structural/numerical checks for Phase 3. No manuscript training is performed."""
from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np

from evaluation import compute_ece
from calibration import fit_temperature_scaling
from spatial_data import load_and_prepare_data
from comparisons import MODALITY_INDICES, _metric_rows

BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "vegetation_reference.csv"


def test_ece_inclusive_one():
    y = np.array([0, 1])
    p = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)
    ece = compute_ece(y, p, n_bins=10)
    assert abs(ece - 0.5) < 1e-12, f"ECE final-bin correction failed: {ece}"


def test_temperature_can_go_below_point_five():
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([
        [0.60, 0.40], [0.40, 0.60], [0.58, 0.42],
        [0.42, 0.58], [0.62, 0.38], [0.38, 0.62],
    ], dtype=float)
    info = fit_temperature_scaling(p, y, temperature_bounds=(0.05, 10.0))
    assert info["temperature"] < 0.5, info
    assert info["search_method"] in {"bounded_log_temperature", "geometric_grid_fallback"}


def test_strict_spatial_split_and_modalities():
    with tempfile.TemporaryDirectory() as td:
        data = load_and_prepare_data(
            str(DATA), output_dir=td, random_state=42,
            split_mode="spatial_block", spatial_fold=0,
            n_spatial_folds=5, n_grid=5,
        )
        audit = data["split_audit"]
        assert not audit["block_overlap_train_test"]
        assert not audit["block_overlap_val_test"]
        assert not audit["block_overlap_train_val"]
        assert data["X_train_raw"].shape[1] == 8
        assert MODALITY_INDICES["lidar_derived"].tolist() == [0,1,2,3,4,5]
        assert MODALITY_INDICES["spectral"].tolist() == [6,7]
        assert MODALITY_INDICES["fused"].tolist() == list(range(8))


def test_supported_only_fusion_uses_the_baseline_metric_definition():
    # Class 2 is absent from training, but present in the geographic test fold.
    y_test = np.array([0, 1, 2, 0])
    raw = np.array([[.85,.10,.05], [.10,.80,.10], [.10,.80,.10], [.80,.10,.10]])
    from evaluation import compute_all_metrics
    rows, audit = _metric_rows("proposed_fusion", y_test, raw, raw, 3,
                               {"temperature": 1.0}, [0, 1])
    assert audit["unsupported_test_class_indices"] == [2]
    assert rows[0]["n_test"] == 4 and rows[0]["n_test_supported"] == 3
    assert rows[0]["n_unsupported_test_classes"] == 1
    mask = y_test != 2
    expected = compute_all_metrics(y_test[mask], raw.argmax(axis=1)[mask], raw[mask], 3)
    for k, ek in [("macro_f1", "macro_f1"), ("ece", "ece"), ("balanced_accuracy", "balanced_acc")]:
        assert abs(rows[0]["supported_only_"+k] - expected[ek]) < 1e-12


def main():
    test_ece_inclusive_one()
    test_temperature_can_go_below_point_five()
    test_strict_spatial_split_and_modalities()
    test_supported_only_fusion_uses_the_baseline_metric_definition()
    print("PHASE 3 SELF-TEST: PASS")


if __name__ == "__main__":
    main()
