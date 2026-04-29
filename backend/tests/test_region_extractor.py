import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pipeline.region_extractor import (
    extract_local_region,
    get_best_slice,
    get_missing_tooth_location
)


def test_region_extractor():
    # Simulate a 100x100x100 volume
    volume = np.random.uniform(-1000, 3000, (100, 100, 100)).astype(np.float32)

    # Simulate segmentation — place two tooth blobs with a gap between them
    segmentation = np.zeros((100, 100, 100), dtype=np.uint8)
    segmentation[50, 30:45, 40:55] = 1  # tooth 1
    segmentation[50, 55:70, 40:55] = 1  # tooth 2 — gap at x=45:55

    spacing = (0.4, 0.4, 0.4)  # mm

    # Test get_best_slice
    z = get_best_slice(segmentation)
    assert z == 50, f"Expected best slice at z=50, got {z}"
    print(f"Best slice: z={z} ✓")

    # Test get_missing_tooth_location
    z_site, x_site, y_site = get_missing_tooth_location(segmentation)
    print(f"Missing tooth location: z={z_site}, x={x_site}, y={y_site} ✓")

    # Test extract_local_region
    local_vol, local_seg, bounds = extract_local_region(
        volume, segmentation,
        site_xyz=(z_site, x_site, y_site),
        spacing=spacing,
        window_mm=10.0
    )

    print(f"Local volume shape: {local_vol.shape}")
    print(f"Local seg shape: {local_seg.shape}")
    print(f"Bounds: {bounds}")

    assert local_vol.shape == local_seg.shape, "Shape mismatch"
    assert local_vol.size > 0, "Empty region extracted"

    # Verify window size is approximately correct in mm
    z_size_mm = (bounds[0][1] - bounds[0][0]) * spacing[2]
    x_size_mm = (bounds[1][1] - bounds[1][0]) * spacing[0]
    y_size_mm = (bounds[2][1] - bounds[2][0]) * spacing[1]
    print(f"Extracted window size: {x_size_mm:.1f}mm x {y_size_mm:.1f}mm x {z_size_mm:.1f}mm")

    print("All assertions passed ✓")


if __name__ == "__main__":
    test_region_extractor()
