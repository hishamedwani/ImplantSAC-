import os
import cv2
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# ── Constants (matching Alaa's notebook exactly) ──────────────────────────
CONF_THR   = 0.35
MIN_SLICES = 5
Z_GAP      = 5
IOU_MERGE  = 0.3
MERGE_DIST = 60
TOP_N      = 6
ARCH_X     = (0.20, 0.80)
ARCH_Y     = (0.10, 0.75)
IMG_SIZE   = 640

# ── Model loader ──────────────────────────────────────────────────────────
_model = None

def get_model() -> YOLO:
    """Load YOLO model once and cache it."""
    global _model
    if _model is None:
        weights_path = os.getenv("YOLO_WEIGHTS_PATH")
        if not weights_path or not Path(weights_path).exists():
            raise FileNotFoundError(
                f"YOLO weights not found at: {weights_path}. "
                "Set YOLO_WEIGHTS_PATH in your .env file."
            )
        _model = YOLO(weights_path)
    return _model


# ── Scanner z-range detection (from Alaa's notebook exactly) ──────────────
def detect_scanner(nz: int, spacing: tuple) -> tuple[str, tuple[float, float]]:
    sp = spacing[0] if spacing else 0.0
    if 430 <= nz <= 460:                return "Kavo",         (0.36, 0.75)
    if 490 <= nz <= 520:                return "Newtom",       (0.28, 0.65)
    if 300 <= nz <= 340:                return "Meyer",        (0.31, 0.80)
    if 250 <= nz <= 310 and sp <= 0.32: return "ToothFairy2",  (0.45, 0.82)
    if 260 <= nz <= 300 and sp >= 0.38: return "FullSkullNII", (0.28, 0.55)
    if 220 <= nz <= 260 and sp >= 0.38: return "FullSkullNII", (0.25, 0.55)
    return "Unknown", (0.25, 0.60)


# ── Slice normalisation (from Alaa's notebook exactly) ────────────────────
def norm_slice(sl: np.ndarray) -> np.ndarray:
    sl  = sl.astype(np.float32)
    lo  = np.percentile(sl, 1)
    hi  = np.percentile(sl, 99)
    sl  = np.clip(sl, lo, hi)
    sl  = ((sl - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return cv2.cvtColor(sl, cv2.COLOR_GRAY2BGR)


# ── IoU (from Alaa's notebook exactly) ────────────────────────────────────
def box_iou(a: tuple, b: tuple) -> float:
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter    = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) +
                    (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)


# ── Post-processing (from Alaa's notebook exactly) ────────────────────────
def merge_nearby(sites: list, img_w: int, img_h: int, dist: int = 60) -> list:
    merged, used = [], set()
    for i, g1 in enumerate(sites):
        if i in used: continue
        boxes1 = [d["box"] for d in g1]
        cx1 = np.mean([(b[0]+b[2])/2 for b in boxes1])
        cy1 = np.mean([(b[1]+b[3])/2 for b in boxes1])
        grp = list(g1); used.add(i)
        for j, g2 in enumerate(sites):
            if j <= i or j in used: continue
            boxes2 = [d["box"] for d in g2]
            cx2 = np.mean([(b[0]+b[2])/2 for b in boxes2])
            cy2 = np.mean([(b[1]+b[3])/2 for b in boxes2])
            if np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2) < dist:
                grp.extend(g2); used.add(j)
        merged.append(grp)
    return merged


def filter_arch(sites: list, img_w: int, img_h: int) -> list:
    kept = []
    for g in sites:
        boxes = [d["box"] for d in g]
        cx = np.mean([(b[0]+b[2])/2 for b in boxes]) / img_w
        cy = np.mean([(b[1]+b[3])/2 for b in boxes]) / img_h
        if ARCH_X[0] < cx < ARCH_X[1] and ARCH_Y[0] < cy < ARCH_Y[1]:
            kept.append(g)
    return kept


def score_rank(sites: list, top_n: int) -> list:
    return sorted(
        sites,
        key=lambda g: np.mean([d["conf"] for d in g]) * len(g),
        reverse=True
    )[:top_n]


# ── Main localization function ────────────────────────────────────────────
def locate_missing_tooth(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    z_override: tuple[float, float] = None,
    device: str = "cpu"
) -> dict:
    """
    Run YOLO inference on a CBCT volume to locate the missing tooth.

    Volume must be in (z, y, x) axis ordering — consistent with our
    cbct_loader.py which always uses SimpleITK.

    Args:
        volume:     3D numpy array (z, y, x)
        spacing:    (sx, sy, sz) voxel spacing in mm
        z_override: optional (frac_start, frac_end) to manually set z range
        device:     "cpu" or "0" for GPU

    Returns:
        dict with keys:
            z       — best axial slice index
            cx      — centroid x in pixel coords of axial slice (column)
            cy      — centroid y in pixel coords of axial slice (row)
            conf    — mean confidence of top site
            score   — conf * n_slices
            z_range — (z_min, z_max) of the detected site
            scanner — detected scanner type
    """
    model = get_model()
    nz, img_h, img_w = volume.shape

    scanner, z_frac = detect_scanner(nz, spacing)
    if z_override:
        z_frac = z_override
    z0, z1 = int(nz * z_frac[0]), int(nz * z_frac[1])

    # Raw inference slice by slice
    raw = []
    for z in range(z0, z1):
        sl  = norm_slice(volume[z])
        res = model.predict(sl, imgsz=IMG_SIZE, conf=CONF_THR,
                            device=device, verbose=False)[0]
        for box in res.boxes:
            raw.append({
                "z":    z,
                "conf": float(box.conf),
                "box":  tuple(box.xyxy[0].tolist()),
            })

    if not raw:
        raise ValueError(
            "YOLO found no detections. "
            "Check z_override or scan quality."
        )

    # Consensus grouping
    raw.sort(key=lambda d: d["z"])
    sites, used = [], set()
    for i, d in enumerate(raw):
        if i in used: continue
        g = [d]; used.add(i)
        for j, d2 in enumerate(raw):
            if j in used or d2["z"] - g[-1]["z"] > Z_GAP: continue
            if box_iou(g[-1]["box"], d2["box"]) >= IOU_MERGE:
                g.append(d2); used.add(j)
        if len(g) >= MIN_SLICES:
            sites.append(g)

    if not sites:
        raise ValueError(
            f"YOLO found {len(raw)} raw detections but none passed "
            f"the MIN_SLICES={MIN_SLICES} threshold."
        )

    sites = merge_nearby(sites, img_w, img_h, MERGE_DIST)
    sites = filter_arch(sites, img_w, img_h)
    sites = score_rank(sites, TOP_N)

    if not sites:
        raise ValueError(
            "No sites survived arch filtering. "
            "Try z_override to manually set the z range."
        )

    # Take the top-scoring site
    top = sites[0]
    zs    = [d["z"]   for d in top]
    confs = [d["conf"] for d in top]
    boxes = [d["box"]  for d in top]

    # Best z = slice with highest confidence in top site
    best  = max(top, key=lambda d: d["conf"])
    z_best = best["z"]

    # Centroid in pixel coordinates of the axial slice
    # cx = column (x-axis), cy = row (y-axis)
    cx = int(np.mean([(b[0]+b[2])/2 for b in boxes]))
    cy = int(np.mean([(b[1]+b[3])/2 for b in boxes]))

    return {
        "z":       z_best,
        "cx":      cx,
        "cy":      cy,
        "conf":    round(float(np.mean(confs)), 3),
        "score":   round(float(np.mean(confs) * len(top)), 2),
        "z_range": (min(zs), max(zs)),
        "scanner": scanner,
    }
