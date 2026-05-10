import os
import uuid
import subprocess
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _get_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Environment variable '{key}' is not set. "
            f"Add it to your .env file."
        )
    return val


def run_toothseg(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    z: int,
    cx: int,
    cy: int,
    window: int = 50
) -> np.ndarray:
    """
    Run ToothSeg (nnU-Net) on a cropped 3D sub-volume around the YOLO site.

    Args:
        volume:  Full CBCT volume (z, y, x) from cbct_loader
        spacing: (sx, sy, sz) voxel spacing in mm
        z:       YOLO best axial slice index
        cx:      YOLO centroid column (x-axis)
        cy:      YOLO centroid row (y-axis)
        window:  Half-window size in voxels for the crop (default 50 = 100 voxels)

    Returns:
        segmentation: 3D numpy array same shape as volume
                      with classes: 0=background, 1=teeth, 2=bone, 3=implant
                      Cropped region is filled with ToothSeg output.
                      Rest of volume is 0 (background).
    """
    results_dir      = _get_env("TOOTHSEG_RESULTS")
    raw_dir          = _get_env("TOOTHSEG_RAW")
    preprocessed_dir = _get_env("TOOTHSEG_PREPROCESSED")

    # Set nnU-Net environment variables
    env = os.environ.copy()
    env["nnUNet_results"]       = results_dir
    env["nnUNet_raw"]           = raw_dir
    env["nnUNet_preprocessed"]  = preprocessed_dir
    env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

    # Create unique temp dirs for this inference run
    run_id    = str(uuid.uuid4())[:8]
    input_dir  = Path(raw_dir) / f"input_{run_id}"
    output_dir = Path(raw_dir) / f"output_{run_id}"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Compute crop bounds
        nz, ny, nx = volume.shape
        z0  = max(0, z  - window);  z1  = min(nz, z  + window)
        y0  = max(0, cy - window);  y1  = min(ny, cy + window)
        x0  = max(0, cx - window);  x1  = min(nx, cx + window)

        # Extract crop
        crop = volume[z0:z1, y0:y1, x0:x1].astype(np.int16)

        assert crop.size > 0, \
            f"Crop is empty. Check YOLO site coordinates: z={z}, cx={cx}, cy={cy}"

        # Save crop as nnU-Net input
        crop_img = sitk.GetImageFromArray(crop)
        crop_img.SetSpacing((float(spacing[0]),
                             float(spacing[1]),
                             float(spacing[2])))
        input_path = str(input_dir / "crop_0000.mha")
        sitk.WriteImage(crop_img, input_path)

        # Run nnU-Net inference
        result = subprocess.run([
            "nnUNetv2_predict",
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-d", "112",
            "-c", "3d_fullres",
            "-f", "0",
            "-chk", "checkpoint_best.pth",
            "-device", "cpu"
        ], capture_output=True, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(
                f"ToothSeg inference failed:\n{result.stderr[-500:]}"
            )

        # Load segmentation output
        pred_path = output_dir / "crop.mha"
        if not pred_path.exists():
            raise FileNotFoundError(
                f"ToothSeg output not found at {pred_path}. "
                f"Files in output dir: {list(output_dir.iterdir())}"
            )

        pred_img  = sitk.ReadImage(str(pred_path))
        pred_crop = sitk.GetArrayFromImage(pred_img).astype(np.uint8)

        assert pred_crop.shape == crop.shape, \
            f"Segmentation shape {pred_crop.shape} != crop shape {crop.shape}"

        # Place crop segmentation back into full volume shape
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        segmentation[z0:z1, y0:y1, x0:x1] = pred_crop

        return segmentation

    finally:
        # Clean up temp dirs
        import shutil
        shutil.rmtree(str(input_dir),  ignore_errors=True)
        shutil.rmtree(str(output_dir), ignore_errors=True)


def determine_is_molar(cx: int, img_width: int) -> bool:
    """
    Determine if the implant site is a molar based on YOLO centroid x position.
    Molars are at the lateral edges of the arch (outer 20% on each side).
    Anterior teeth are near the center.

    Args:
        cx:        YOLO centroid column (x-axis) in pixel coordinates
        img_width: Width of the axial image in pixels

    Returns:
        True if molar, False if anterior
    """
    relative_x = cx / img_width
    return relative_x < 0.25 or relative_x > 0.75