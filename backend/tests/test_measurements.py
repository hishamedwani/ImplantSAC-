import sys
import os
import numpy as np
import scipy.ndimage as ndi

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline.measurements import compute_measurements


def test_measurements():
    spacing = (0.4, 0.4, 0.4)

    # Build a synthetic 60x60x60 volume
    volume = np.full((60, 60, 60), -1000.0, dtype=np.float32)

    # Add bone ring around center (HU > 200)
    volume[20:40, 20:40, 10:50] = 700.0

    # Add a segmentation with two tooth blobs
    segmentation = np.zeros((60, 60, 60), dtype=np.uint8)
    segmentation[25:35, 22:28, 15:25] = 1  # tooth 1
    segmentation[25:35, 22:28, 35:45] = 1  # tooth 2

    results = compute_measurements(
        local_volume=volume,
        local_seg=segmentation,
        spacing=spacing,
        is_molar=True
    )

    print("\n--- Measurement Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")

    # Assertions
    assert results["apical_bone_mm"] >= 0, "Apical bone must be >= 0"
    assert results["buccal_wall_mm"] >= 0, "Buccal wall must be >= 0"
    assert results["ridge_width_mm"] >= 0, "Ridge width must be >= 0"
    assert results["septum_width_mm"] is not None, "Septum must be computed for molar"
    assert results["septum_width_mm"] >= 0, "Septum width must be >= 0"
    assert isinstance(results["lesion_detected"], bool), "Lesion must be bool"

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    test_measurements()
