import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.cbct_loader import load_cbct
from app.pipeline.yolo_locator import locate_missing_tooth
from app.pipeline.toothseg import run_toothseg, determine_is_molar

FILE_PATH = os.environ.get(
    "TEST_SCAN_PATH",
    r"C:\Users\t480s\OneDrive\Desktop\Input_scan_seg\1001470284_20180115.nii.gz"
)


def test_toothseg():
    print(f"\nLoading scan: {FILE_PATH}")
    volume, spacing = load_cbct(FILE_PATH)
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing: {spacing}")

    print("\nRunning YOLO localization...")
    yolo = locate_missing_tooth(volume, spacing, device="cpu")
    z, cx, cy = yolo["z"], yolo["cx"], yolo["cy"]
    print(f"YOLO site: z={z}, cx={cx}, cy={cy}")
    print(f"Scanner: {yolo['scanner']}, conf={yolo['conf']}")

    is_molar = determine_is_molar(cx, volume.shape[2])
    print(f"Is molar: {is_molar}")

    print("\nRunning ToothSeg on crop...")
    segmentation = run_toothseg(
        volume=volume,
        spacing=spacing,
        z=z,
        cx=cx,
        cy=cy,
        window=50
    )

    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Unique classes: {sorted(set(segmentation.flatten().tolist()))}")
    print(f"Teeth voxels:  {int((segmentation == 1).sum())}")
    print(f"Bone voxels:   {int((segmentation == 2).sum())}")
    print(f"Implant voxels:{int((segmentation == 3).sum())}")

    assert segmentation.shape == volume.shape, "Shape mismatch"
    assert segmentation.max() <= 3, "Unexpected class value"
    assert (segmentation == 1).sum() > 0 or (segmentation == 2).sum() > 0, \
        "No teeth or bone found — check YOLO site or ToothSeg output"

    print("\nAll assertions passed ✓")


if __name__ == "__main__":
    test_toothseg()