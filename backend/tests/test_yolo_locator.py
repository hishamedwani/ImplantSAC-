import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.cbct_loader import load_cbct
from app.pipeline.yolo_locator import locate_missing_tooth

# Set this to your test scan path
FILE_PATH = os.environ.get("TEST_SCAN_PATH", r"C:\path\to\your\scan.nii.gz")


def test_yolo_locator():
    print(f"\nLoading scan: {FILE_PATH}")
    volume, spacing = load_cbct(FILE_PATH)

    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")

    print("\nRunning YOLO localization...")
    result = locate_missing_tooth(volume, spacing, device="cpu")

    print("\n--- YOLO Result ---")
    for k, v in result.items():
        print(f"  {k}: {v}")

    assert result["z"] >= 0, "z must be non-negative"
    assert result["cx"] >= 0, "cx must be non-negative"
    assert result["cy"] >= 0, "cy must be non-negative"
    assert 0.0 <= result["conf"] <= 1.0, "conf must be between 0 and 1"

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    test_yolo_locator()
