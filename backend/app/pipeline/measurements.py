import numpy as np
import scipy.ndimage as ndi


def compute_measurements(
    local_volume: np.ndarray,
    local_seg: np.ndarray,
    spacing: tuple[float, float, float],
    is_molar: bool = False
) -> dict:
    """
    Compute all 5 clinical measurements from the local region.
    Works on the best 2D axial slice within the local region,
    consistent with how ToothSeg notebook validated measurements.

    Args:
        local_volume: Cropped HU volume around implant site
        local_seg:    Cropped segmentation around implant site
        spacing:      (sx, sy, sz) real voxel spacing in mm
        is_molar:     Whether the implant site is a molar

    Returns:
        Dictionary of measurements in mm
    """
    assert local_volume.shape == local_seg.shape, \
        "Volume and segmentation shape mismatch"
    assert all(s > 0 for s in spacing), \
        "Spacing must be positive"

    sx, sy, sz = spacing

    # Find the best 2D slice — most teeth voxels
    teeth_vol = (local_seg == 1)
    teeth_per_slice = teeth_vol.sum(axis=(1, 2))

    if teeth_per_slice.max() == 0:
        # No teeth found — use center slice
        z = local_volume.shape[0] // 2
    else:
        z = int(np.argmax(teeth_per_slice))

    # Work on the best 2D slice
    img_2d = local_volume[z]
    seg_2d = local_seg[z]

    cx = img_2d.shape[0] // 2
    cy = img_2d.shape[1] // 2

    # Bone mask from HU threshold
    bone_2d = img_2d > 200

    # Teeth mask from segmentation
    teeth_2d = (seg_2d == 1)
    teeth_zone = ndi.binary_dilation(teeth_2d, iterations=4)

    # Restrict bone to near teeth
    bone = bone_2d & teeth_zone

    results = {}

    # ------------------------------------------------------------------
    # 1. APICAL BONE AVAILABILITY
    # Measure continuous bone below implant apex along vertical axis
    # ------------------------------------------------------------------
    col = bone[:, cy]
    indices = np.where(col > 0)[0]
    if len(indices) > 0:
        apical_px = max(0, indices[-1] - cx)
    else:
        apical_px = 0
    apical_mm = float(apical_px) * sx
    assert apical_mm >= 0, "Apical bone cannot be negative"
    results["apical_bone_mm"] = round(apical_mm, 2)

    # ------------------------------------------------------------------
    # 2. BUCCAL WALL THICKNESS
    # Measure bone thickness on the buccal (outer) side
    # ------------------------------------------------------------------
    front = bone[cx, :cy]
    indices = np.where(front > 0)[0]
    buccal_px = len(indices) if len(indices) > 0 else 0
    buccal_mm = float(buccal_px) * sy
    assert buccal_mm >= 0, "Buccal thickness cannot be negative"
    results["buccal_wall_mm"] = round(buccal_mm, 2)

    # ------------------------------------------------------------------
    # 3. BUCCOLINGUAL RIDGE WIDTH
    # Measure total horizontal bone span at crest level
    # ------------------------------------------------------------------
    row = bone[cx, :]
    indices = np.where(row > 0)[0]
    if len(indices) > 1:
        ridge_px = indices[-1] - indices[0]
    else:
        ridge_px = 0
    ridge_mm = float(ridge_px) * sy
    assert ridge_mm >= 0, "Ridge width cannot be negative"
    results["ridge_width_mm"] = round(ridge_mm, 2)

    # ------------------------------------------------------------------
    # 4. INTERRADICULAR SEPTUM WIDTH (molars only)
    # Measure minimum distance between root centroids in mm
    # ------------------------------------------------------------------
    if is_molar:
        labeled_teeth, num_teeth = ndi.label(teeth_2d)
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
                        (centers[i] - centers[j]) * np.array([sx, sy])
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
    # 5. PERIAPICAL LESION STATUS
    # Detect low-density regions inside bone near apex
    # ------------------------------------------------------------------
    low_density = img_2d < 100
    bone_eroded = ndi.binary_erosion(bone_2d, iterations=1)
    candidate = low_density & bone_eroded
    labeled, num = ndi.label(candidate)
    lesion_detected = False
    lesion_size_mm3 = 0.0
    if num > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        max_size_px = max(sizes)
        voxel_area_mm2 = sx * sy
        max_size_mm2 = max_size_px * voxel_area_mm2
        threshold_mm2 = 80 * (0.4 ** 2)
        if max_size_mm2 > threshold_mm2:
            lesion_detected = True
            lesion_size_mm3 = round(max_size_mm2, 2)

    results["lesion_detected"] = lesion_detected
    results["lesion_size_mm3"] = lesion_size_mm3

    return results