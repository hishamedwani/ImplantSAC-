import numpy as np
import scipy.ndimage as ndi


def extract_views(
    volume: np.ndarray,
    segmentation: np.ndarray,
    z: int,
    cx: int,
    cy: int
) -> dict:
    """
    Extract three orthogonal 2D views at the YOLO-detected implant site.

    Volume and segmentation are in (z, y, x) axis ordering.
    YOLO output: z = axial slice, cx = column (x-axis), cy = row (y-axis)

    Returns:
        dict with keys: axial, coronal, sagittal
        Each value is a dict with: img (HU), seg (segmentation mask)
    """
    assert volume.shape == segmentation.shape, \
        "Volume and segmentation shape mismatch"

    return {
        "axial": {
            "img": volume[z, :, :],
            "seg": segmentation[z, :, :],
        },
        "coronal": {
            "img": volume[:, cy, :],
            "seg": segmentation[:, cy, :],
        },
        "sagittal": {
            "img": volume[:, :, cx],
            "seg": segmentation[:, :, cx],
        },
    }


def compute_measurements(
    volume: np.ndarray,
    segmentation: np.ndarray,
    z: int,
    cx: int,
    cy: int,
    spacing: tuple[float, float, float],
    is_molar: bool = False
) -> dict:
    """
    Compute all 5 clinical measurements using three orthogonal views.

    Measurement to view mapping (agreed):
        Apical Bone          → Sagittal
        Buccal Wall          → Coronal
        Ridge Width          → Coronal
        Interradicular Sep.  → Axial
        Periapical Lesion    → Sagittal

    Args:
        volume:       Full CBCT volume (z, y, x)
        segmentation: Full segmentation volume (z, y, x)
        z:            YOLO best axial slice index
        cx:           YOLO centroid column (x-axis)
        cy:           YOLO centroid row (y-axis)
        spacing:      (sx, sy, sz) voxel spacing in mm
        is_molar:     Whether the implant site is a molar

    Returns:
        Dictionary of measurements in mm
    """
    assert volume.shape == segmentation.shape, \
        "Volume and segmentation shape mismatch"
    assert all(s > 0 for s in spacing), \
        "Spacing must be positive"

    sx, sy, sz = spacing
    views = extract_views(volume, segmentation, z, cx, cy)

    results = {}

    # ------------------------------------------------------------------
    # 1. APICAL BONE AVAILABILITY — Sagittal view (z, y)
    # Measure continuous bone below implant apex along z-axis
    # Anchor point: (z, cy) in sagittal view
    # ------------------------------------------------------------------
    sag_img = views["sagittal"]["img"]   # shape (z_total, y_total)
    sag_seg = views["sagittal"]["seg"]

    sag_bone  = sag_img > 200
    sag_teeth = (sag_seg == 1)
    sag_zone  = ndi.binary_dilation(sag_teeth, iterations=4)
    sag_bone  = sag_bone & sag_zone

    # Column at cy in sagittal view, look downward from z
    col = sag_bone[z:, cy] if cy < sag_bone.shape[1] else sag_bone[z:, sag_bone.shape[1]//2]
    indices = np.where(col > 0)[0]
    apical_px = int(indices[-1]) if len(indices) > 0 else 0
    apical_mm = float(apical_px) * sz
    assert apical_mm >= 0, "Apical bone cannot be negative"
    results["apical_bone_mm"] = round(apical_mm, 2)

    # ------------------------------------------------------------------
    # 2. BUCCAL WALL THICKNESS — Coronal view (z, x)
    # Measure bone on the buccal (outer) side at implant level
    # Anchor point: (z, cx) in coronal view
    # ------------------------------------------------------------------
    cor_img = views["coronal"]["img"]    # shape (z_total, x_total)
    cor_seg = views["coronal"]["seg"]

    cor_bone  = cor_img > 200
    cor_teeth = (cor_seg == 1)
    cor_zone  = ndi.binary_dilation(cor_teeth, iterations=4)
    cor_bone  = cor_bone & cor_zone

    # Row at z in coronal view, measure bone to the left of cx (buccal side)
    row = cor_bone[z, :cx] if cx < cor_bone.shape[1] else cor_bone[z, :]
    indices = np.where(row > 0)[0]
    buccal_px = len(indices) if len(indices) > 0 else 0
    buccal_mm = float(buccal_px) * sx
    assert buccal_mm >= 0, "Buccal thickness cannot be negative"
    results["buccal_wall_mm"] = round(buccal_mm, 2)

    # ------------------------------------------------------------------
    # 3. BUCCOLINGUAL RIDGE WIDTH — Coronal view (z, x)
    # Measure total horizontal bone span at crest level
    # Anchor point: row z in coronal view
    # ------------------------------------------------------------------
    row = cor_bone[z, :]
    indices = np.where(row > 0)[0]
    ridge_px = int(indices[-1] - indices[0]) if len(indices) > 1 else 0
    ridge_mm = float(ridge_px) * sx
    assert ridge_mm >= 0, "Ridge width cannot be negative"
    results["ridge_width_mm"] = round(ridge_mm, 2)

    # ------------------------------------------------------------------
    # 4. INTERRADICULAR SEPTUM WIDTH — Axial view (y, x)
    # Measure minimum distance between root centroids
    # Anchor point: axial slice z
    # ------------------------------------------------------------------
    if is_molar:
        ax_seg   = views["axial"]["seg"]    # shape (y_total, x_total)
        ax_teeth = (ax_seg == 1)
        labeled_teeth, num_teeth = ndi.label(ax_teeth)
        if num_teeth >= 2:
            centers = []
            for i in range(1, num_teeth + 1):
                coords = np.argwhere(labeled_teeth == i)
                centers.append(coords.mean(axis=0))
            centers = np.array(centers)
            dists = []
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dist_mm = np.linalg.norm(
                        (centers[i] - centers[j]) * np.array([sy, sx])
                    )
                    dists.append(dist_mm)
            septum_mm = float(min(dists))
        else:
            septum_mm = 0.0
        assert septum_mm >= 0, "Septum width cannot be negative"
        results["septum_width_mm"] = round(septum_mm, 2)
    else:
        results["septum_width_mm"] = None

    # ------------------------------------------------------------------
    # 5. PERIAPICAL LESION STATUS — Sagittal view (z, y)
    # Detect low-density regions inside bone near apex
    # ------------------------------------------------------------------
    sag_bone_raw = sag_img > 200
    low_density  = sag_img < 100
    bone_eroded  = ndi.binary_erosion(sag_bone_raw, iterations=1)
    candidate    = low_density & bone_eroded
    labeled, num = ndi.label(candidate)

    lesion_detected  = False
    lesion_size_mm3  = 0.0
    if num > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        max_size_px  = max(sizes)
        voxel_area   = sy * sz
        max_size_mm2 = max_size_px * voxel_area
        threshold    = 80 * (0.4 ** 2)
        if max_size_mm2 > threshold:
            lesion_detected = True
            lesion_size_mm3 = round(max_size_mm2, 2)

    results["lesion_detected"] = lesion_detected
    results["lesion_size_mm3"] = lesion_size_mm3

    return results