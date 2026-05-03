import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.core.cbct_loader import load_cbct
from app.pipeline.region_extractor import get_missing_tooth_location, extract_local_region
from app.pipeline.measurements import compute_measurements
from app.classification.sac_classifier import classify_sac

router = APIRouter(tags=["cases"])

# Temporary in-memory store — will be replaced with PostgreSQL in Week 3
cases_store = {}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_case(
    file: UploadFile = File(...),
    patient_id: str = Form(default="anonymous")
):
    """
    Accept a CBCT scan file and run the full pipeline.
    Returns case_id and full SAC classification result.
    """
    # Validate file type
    allowed_extensions = [".nii", ".nii.gz", ".mha"]
    filename = file.filename or ""
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )

    # Save uploaded file
    case_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{case_id}_{filename}")

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Step 1: Load CBCT
        volume, spacing = load_cbct(save_path)

        # Step 2: Placeholder segmentation — ToothSeg integration coming in Week 3
        # For now we use a mock segmentation to test the endpoint flow
        import numpy as np
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        mid = [s // 2 for s in volume.shape]
        segmentation[
            mid[0]-5:mid[0]+5,
            mid[1]-10:mid[1],
            mid[2]-10:mid[2]
        ] = 1
        segmentation[
            mid[0]-5:mid[0]+5,
            mid[1]-10:mid[1],
            mid[2]:mid[2]+10
        ] = 1

        # Step 3: Get implant site location
        z, x, y = get_missing_tooth_location(segmentation)

        # Step 4: Extract local region
        local_vol, local_seg, bounds = extract_local_region(
            volume, segmentation,
            site_xyz=(z, x, y),
            spacing=spacing,
            window_mm=20.0
        )

        # Step 5: Compute measurements
        measurements = compute_measurements(
            local_volume=local_vol,
            local_seg=local_seg,
            spacing=spacing,
            is_molar=False
        )

        # Step 6: SAC classification
        result = classify_sac(measurements)

        # Store result
        cases_store[case_id] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "filename": filename,
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
    """
    Retrieve full results for a specific case.
    """
    if case_id not in cases_store:
        raise HTTPException(status_code=404, detail="Case not found")
    return cases_store[case_id]


@router.get("/")
def get_all_cases():
    """
    Retrieve all analyzed cases.
    """
    return list(cases_store.values())
