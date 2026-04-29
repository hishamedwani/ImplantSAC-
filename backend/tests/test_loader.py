import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.cbct_loader import load_cbct

# Set the path to your .nii.gz file via environment variable or edit this line locally
FILE_PATH = os.environ.get("TEST_SCAN_PATH", r"C:\path\to\your\file.nii.gz")

def test_load():
    volume, spacing = load_cbct(FILE_PATH)
    
    print(f"Volume shape: {volume.shape}")
    print(f"Spacing (mm): x={spacing[0]:.4f}, y={spacing[1]:.4f}, z={spacing[2]:.4f}")
    print(f"Intensity range: min={volume.min():.1f}, max={volume.max():.1f}")
    
    assert len(volume.shape) == 3, "Volume must be 3D"
    assert all(s > 0 for s in spacing), "Spacing must be positive"
    print("All assertions passed.")

if __name__ == "__main__":
    test_load()
