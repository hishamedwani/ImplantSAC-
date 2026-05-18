import os
import io
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session
from PIL import Image

from app.core.cbct_loader import load_cbct
from app.db.database import get_db
from app.db.models import Case

router = APIRouter(tags=["viewer"])

SEG_COLORS = {
    1: (0,   180, 216, 140),
    2: (255, 255, 255,  60),
    3: (251, 191,  36, 180),
}


def normalize_slice(sl: np.ndarray) -> np.ndarray:
    sl   = sl.astype(np.float32)
    lo   = np.percentile(sl, 1)
    hi   = np.percentile(sl, 99)
    sl   = np.clip(sl, lo, hi)
    sl   = ((sl - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return sl


def render_slice_png(
    img_slice: np.ndarray,
    seg_slice: np.ndarray,
    crosshair: tuple | None = None,
) -> bytes:
    gray    = normalize_slice(img_slice)
    h, w    = gray.shape
    base    = Image.fromarray(gray, mode='L').convert('RGBA')
    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    pix     = overlay.load()

    for class_id, color in SEG_COLORS.items():
        mask = (seg_slice == class_id)
        rows, cols = np.where(mask)
        for r, c in zip(rows, cols):
            pix[c, r] = color

    result = Image.alpha_composite(base, overlay).convert('RGB')

    if crosshair:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(result)
        cr, cc = crosshair
        draw.line([(max(0, cc - 20), cr), (min(w-1, cc + 20), cr)], fill=(0, 180, 216), width=1)
        draw.line([(cc, max(0, cr - 20)), (cc, min(h-1, cr + 20))], fill=(0, 180, 216), width=1)

    buf = io.BytesIO()
    result.save(buf, format='PNG')
    return buf.getvalue()


@router.get("/{case_id}/volume-info")
def get_volume_info(case_id: str, db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.cbct_path or not os.path.exists(case.cbct_path):
        raise HTTPException(status_code=404, detail="CBCT file not found on disk")

    volume, spacing = load_cbct(case.cbct_path)

    return {
        "shape":   list(volume.shape),
        "spacing": [case.spacing_x, case.spacing_y, case.spacing_z],
        "yolo": {
            "z":  case.yolo_z,
            "cx": case.yolo_cx,
            "cy": case.yolo_cy,
        }
    }


@router.get("/{case_id}/slice")
def get_slice(
    case_id: str,
    view:    str = Query(..., description="axial | coronal | sagittal"),
    index:   int = Query(..., description="Slice index"),
    db:      Session = Depends(get_db)
):
    case = db.query(Case).filter(Case.id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not case.cbct_path or not os.path.exists(case.cbct_path):
        raise HTTPException(status_code=404, detail="CBCT file not found")
    if not case.segmentation_path or not os.path.exists(case.segmentation_path):
        raise HTTPException(status_code=404, detail="Segmentation file not found")

    volume, _    = load_cbct(case.cbct_path)
    segmentation = np.load(case.segmentation_path)

    nz, ny, nx = volume.shape

    if view == "axial":
        index     = max(0, min(index, nz - 1))
        img_slice = volume[index, :, :]
        seg_slice = segmentation[index, :, :]
        crosshair = (case.yolo_cy, case.yolo_cx) if index == case.yolo_z else None
    elif view == "coronal":
        index     = max(0, min(index, ny - 1))
        img_slice = volume[:, index, :]
        seg_slice = segmentation[:, index, :]
        crosshair = (case.yolo_z, case.yolo_cx) if index == case.yolo_cy else None
    elif view == "sagittal":
        index     = max(0, min(index, nx - 1))
        img_slice = volume[:, :, index]
        seg_slice = segmentation[:, :, index]
        crosshair = (case.yolo_z, case.yolo_cy) if index == case.yolo_cx else None
    else:
        raise HTTPException(status_code=400, detail="view must be axial, coronal, or sagittal")

    png_bytes = render_slice_png(img_slice, seg_slice, crosshair=crosshair)
    return Response(content=png_bytes, media_type="image/png")