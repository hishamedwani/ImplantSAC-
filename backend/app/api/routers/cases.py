import os
import uuid
import shutil
import numpy as np
import scipy.ndimage as ndi
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.core.cbct_loader import load_cbct
from app.pipeline.measurements import compute_measurements
from app.classification.sac_classifier import classify_sac

router = APIRouter(tags=["cases"])

# Temporary in-memory store — will be replaced with PostgreSQL in Week 3
cases_store = {}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_best_slice_and_site(segmentation: np.ndarray) -> tuple[int, int, int]:
    """
    Find best axial slice and missing tooth location.
    Replicates notebook logic exactly — works in 2D on best slice.
    """
    teeth_vol = (segmentation == 1)
    teeth_per_slice = teeth_vol.sum(axis=(1, 2))
    z = int(np.argmax(teeth_per_slice))

    teeth_2d = teeth_vol[z]
    labeled, num = ndi.label(teeth_2d)

    assert num >= 2, "Need at least 2 tooth regions to detect a gap"

    centroids = []
    for i in range(1, num + 1):
        coords = np.argwhere(labeled == i)
        centroids.append(coords.mean(axis=0))

    centroids = np.array(centroids)
    centroids = centroids[np.argsort(centroids[:, 1])]

    distances = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
    gap_idx = int(np.argmax(distances))

    missing = (centroids[gap_idx] + centroids[gap_idx + 1]) / 2
    x, y = int(missing[0]), int(missing[1])

    return z, x, y


def extract_2d_patch(
    volume: np.ndarray,
    segmentation: np.ndarray,
    z: int, x: int, y: int,
    spacing: tuple[float, float, float],
    window_mm: float = 20.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D patch around implant site on the best slice.
    Window size in mm converted to pixels using real spacing.
    """
    sx, sy, _ = spacing
    wx = int(round(window_mm / sx))
    wy = int(round(window_mm / sy))

    x0 = max(0, x - wx)
    x1 = min(volume.shape[1], x + wx)
    y0 = max(0, y - wy)
    y1 = min(volume.shape[2], y + wy)

    patch_vol = volume[z, x0:x1, y0:y1]
    patch_seg = segmentation[z, x0:x1, y0:y1]

    return patch_vol, patch_seg


@router.post("/upload")
async def upload_case(
    file: UploadFile = File(...),
    segmentation_file: UploadFile = File(...),
    patient_id: str = Form(default="anonymous"),
    is_molar: bool = Form(default=False)
):
    """
    Accept a CBCT scan and pre-computed segmentation.
    Runs full measurement and SAC classification pipeline.
    """
    allowed_extensions = [".nii", ".nii.gz", ".mha"]
    for f in [file, segmentation_file]:
        fname = f.filename or ""
        if not any(fname.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {fname}. Allowed: {allowed_extensions}"
            )

    case_id = str(uuid.uuid4())
    cbct_path = os.path.join(UPLOAD_DIR, f"{case_id}_cbct_{file.filename}")
    seg_path = os.path.join(UPLOAD_DIR, f"{case_id}_seg_{segmentation_file.filename}")

    with open(cbct_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    with open(seg_path, "wb") as f_out:
        shutil.copyfileobj(segmentation_file.file, f_out)

    try:
        # Step 1: Load CBCT and segmentation
        volume, spacing = load_cbct(cbct_path)
        segmentation, _ = load_cbct(seg_path)
        segmentation = segmentation.astype(np.uint8)

        assert volume.shape == segmentation.shape, \
            f"Shape mismatch: CBCT {volume.shape} vs seg {segmentation.shape}"

        # Step 2: Find best slice and missing tooth site
        z, x, y = get_best_slice_and_site(segmentation)


        # Step 3: Extract 2D patch around site
        patch_vol, patch_seg = extract_2d_patch(
            volume, segmentation,
            z=z, x=x, y=y,
            spacing=spacing,
            window_mm=20.0
        )


        # Step 4: Compute measurements on 2D patch
        # Wrap patch as single-slice 3D for measurements module
        measurements = compute_measurements(
            local_volume=patch_vol[np.newaxis, :, :],
            local_seg=patch_seg[np.newaxis, :, :],
            spacing=spacing,
            is_molar=is_molar
        )

        # Step 5: SAC classification
        result = classify_sac(measurements)

        cases_store[case_id] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "filename": file.filename,
            "spacing_mm": spacing,
            "result": result
        }

        return JSONResponse(content={
            "case_id": case_id,
            "patient_id": patient_id,
            "result": result
        })

    except AssertionError as e:
        raise HTTPException(status_code=422, detail=f"Pipeline assertion failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/{case_id}")
def get_case(case_id: str):
    if case_id not in cases_store:
        raise HTTPException(status_code=404, detail="Case not found")
    return cases_store[case_id]


@router.get("/")
def get_all_cases():
    return list(cases_store.values())