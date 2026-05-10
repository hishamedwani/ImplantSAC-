import time
import numpy as np
import SimpleITK as sitk
import os
import subprocess

INPUT_DIR  = r"C:\Users\t480s\OneDrive\Desktop\weights\test_crop\imagesTs"
OUTPUT_DIR = r"C:\Users\t480s\OneDrive\Desktop\weights\test_crop\preds"

os.makedirs(INPUT_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a fake 64x64x64 crop
fake_vol = np.random.randint(-1000, 3000, (64, 64, 64)).astype(np.int16)
img      = sitk.GetImageFromArray(fake_vol)
img.SetSpacing((0.4, 0.4, 0.4))
out_path = os.path.join(INPUT_DIR, "crop_0000.mha")
sitk.WriteImage(img, out_path)

# Confirm file exists before running
assert os.path.exists(out_path), f"Input file not written: {out_path}"
print(f"Input file written: {out_path}")
print(f"File size: {os.path.getsize(out_path)} bytes")

print("\nRunning ToothSeg on 64x64x64 crop locally...")
start = time.time()

# Force CPU with device flag
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""

result = subprocess.run([
    "nnUNetv2_predict",
    "-i", INPUT_DIR,
    "-o", OUTPUT_DIR,
    "-d", "112",
    "-c", "3d_fullres",
    "-f", "0",
    "-chk", "checkpoint_best.pth",
    "-device", "cpu"
], capture_output=True, text=True, env=env)

elapsed = time.time() - start
print(f"Inference time: {elapsed:.1f} seconds")
print(f"Return code: {result.returncode}")
if result.returncode != 0:
    print("STDERR:", result.stderr[-1000:])
else:
    print("SUCCESS")
    print(result.stdout[-500:])