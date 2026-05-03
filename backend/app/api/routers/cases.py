import os
import uuid
import shutil
import numpy as np
import scipy.ndimage as ndi
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.cbct_loader import load_cbct
from app.pipeline.measurements import compute_measurements
from app.classification.sac_classifier import classify_sac
from app.db.database import get_db
from app.db.models import Case

router = APIRouter(tags=["cases"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_best_slice_and_site(segmentation: np.ndarray) -> tuple[int, int, int]:
    """
    Find best axial slice and missing tooth location.
    Placeholder until YOLO model is ready.
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
    """Extract 2D patch around implant site using real mm spacing."""
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
    is_molar: bool = Form(default=False),
    db: Session = Depends(get_db)
):
    """
    Accept a CBCT scan and pre-computed segmentation.
    Runs full measurement and SAC classification pipeline.
    Saves result to PostgreSQL.
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

        # Step 4: Compute measurements
        measurements = compute_measurements(
            local_volume=patch_vol[np.newaxis, :, :],
            local_seg=patch_seg[np.newaxis, :, :],
            spacing=spacing,
            is_molar=is_molar
        )

        # Step 5: SAC classification
        result = classify_sac(measurements)

        # Step 6: Save to PostgreSQL
        factors = result["factors"]
        case = Case(
            id=case_id,
            patient_id=patient_id,
            filename=file.filename,
            spacing_x=spacing[0],
            spacing_y=spacing[1],
            spacing_z=spacing[2],
            apical_bone_mm=factors["apical_bone"]["measurement_mm"],
            buccal_wall_mm=factors["buccal_wall"]["measurement_mm"],
            ridge_width_mm=factors["ridge_width"]["measurement_mm"],
            septum_width_mm=factors["septum_width"]["measurement_mm"],
            lesion_detected=factors["periapical_lesion"]["lesion_detected"],
            lesion_size_mm3=factors["periapical_lesion"]["lesion_size_mm3"],
            apical_risk=factors["apical_bone"]["risk"],
            buccal_risk=factors["buccal_wall"]["risk"],
            ridge_risk=factors["ridge_width"]["risk"],
            septum_risk=factors["septum_width"]["risk"],
            lesion_risk=factors["periapical_lesion"]["risk"],
            classification=result["classification"],
            reasoning=result["reasoning"],
            full_result=result
        )
        db.add(case)
        db.commit()
        db.refresh(case)

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
def get_case(case_id: str, db: Session = Depends(get_db)):
    """Retrieve full results for a specific case."""
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case.full_result


@router.get("/")
def get_all_cases(db: Session = Depends(get_db)):
    """Retrieve all analyzed cases."""
    cases = db.query(Case).order_by(Case.created_at.desc()).all()
    return [
        {
            "case_id": c.id,
            "patient_id": c.patient_id,
            "filename": c.filename,
            "classification": c.classification,
            "created_at": c.created_at.isoformat()
        }
        for c in cases
    ]