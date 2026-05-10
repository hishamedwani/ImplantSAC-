import os
import uuid
import shutil
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.cbct_loader import load_cbct
from app.pipeline.yolo_locator import locate_missing_tooth
from app.pipeline.measurements import compute_measurements
from app.classification.sac_classifier import classify_sac
from app.db.database import get_db
from app.db.models import Case

router = APIRouter(tags=["cases"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    Runs YOLO localization, extracts three orthogonal views,
    computes measurements, and produces SAC classification.
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

    case_id  = str(uuid.uuid4())
    cbct_path = os.path.join(UPLOAD_DIR, f"{case_id}_cbct_{file.filename}")
    seg_path  = os.path.join(UPLOAD_DIR, f"{case_id}_seg_{segmentation_file.filename}")

    with open(cbct_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    with open(seg_path, "wb") as f_out:
        shutil.copyfileobj(segmentation_file.file, f_out)

    try:
        # Step 1: Load CBCT and segmentation
        volume, spacing      = load_cbct(cbct_path)
        segmentation, _      = load_cbct(seg_path)
        segmentation         = segmentation.astype(np.uint8)

        assert volume.shape == segmentation.shape, \
            f"Shape mismatch: CBCT {volume.shape} vs seg {segmentation.shape}"

        # Step 2: YOLO localization — find missing tooth site
        yolo_result = locate_missing_tooth(
            volume=volume,
            spacing=spacing,
            device="cpu"
        )
        z  = yolo_result["z"]
        cx = yolo_result["cx"]
        cy = yolo_result["cy"]

        # Step 3: Compute measurements using three orthogonal views
        measurements = compute_measurements(
            volume=volume,
            segmentation=segmentation,
            z=z,
            cx=cx,
            cy=cy,
            spacing=spacing,
            is_molar=is_molar
        )

        # Step 4: SAC classification
        result = classify_sac(measurements)

        # Step 5: Save to PostgreSQL
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
            "case_id":     case_id,
            "patient_id":  patient_id,
            "yolo": {
                "z":       z,
                "cx":      cx,
                "cy":      cy,
                "conf":    yolo_result["conf"],
                "score":   yolo_result["score"],
                "z_range": yolo_result["z_range"],
                "scanner": yolo_result["scanner"],
            },
            "result": result
        })

    except AssertionError as e:
        raise HTTPException(status_code=422, detail=f"Pipeline assertion failed: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
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
            "case_id":        c.id,
            "patient_id":     c.patient_id,
            "filename":       c.filename,
            "classification": c.classification,
            "created_at":     c.created_at.isoformat()
        }
        for c in cases
    ]