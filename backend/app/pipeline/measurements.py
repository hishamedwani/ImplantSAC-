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

    Args:
        local_volume: Cropped HU volume around implant site
        local_seg:    Cropped segmentation around implant site
        spacing:      (sx, sy, sz) real voxel spacing in mm
        is_molar:     Whether the implant site is a molar (enables septum)

    Returns:
        Dictionary of measurements in mm and raw values
    """
    assert local_volume.shape == local_seg.shape, \
        "Volume and segmentation shape mismatch"
    assert all(s > 0 for s in spacing), \
        "Spacing must be positive"

    sx, sy, sz = spacing
    cx = local_volume.shape[1] // 2
    cy = local_volume.shape[2] // 2
    cz = local_volume.shape[0] // 2

    # Bone mask from HU threshold
    bone_mask = local_volume > 200

    # Teeth mask from segmentation
    teeth_mask = (local_seg == 1)
    teeth_zone = ndi.binary_dilation(teeth_mask, iterations=4)
    bone_near_teeth = bone_mask & teeth_zone

    results = {}

    # ------------------------------------------------------------------
    # 1. APICAL BONE AVAILABILITY
    # Measure continuous bone below implant apex along vertical axis
    # ------------------------------------------------------------------
    col = bone_near_teeth[:, cx, cy]
    indices = np.where(col > 0)[0]
    if len(indices) > 0:
        apical_px = indices[-1] - cz
        apical_px = max(0, apical_px)
    else:
        apical_px = 0
    apical_mm = float(apical_px) * sz
    assert apical_mm >= 0, "Apical bone cannot be negative"
    results["apical_bone_mm"] = round(apical_mm, 2)

    # ------------------------------------------------------------------
    # 2. BUCCAL WALL THICKNESS
    # Measure bone thickness on the buccal (outer) side
    # ------------------------------------------------------------------
    row = bone_near_teeth[cz, cx, :cy]
    indices = np.where(row > 0)[0]
    buccal_px = len(indices) if len(indices) > 0 else 0
    buccal_mm = float(buccal_px) * sy
    assert buccal_mm >= 0, "Buccal thickness cannot be negative"
    results["buccal_wall_mm"] = round(buccal_mm, 2)

    # ------------------------------------------------------------------
    # 3. BUCCOLINGUAL RIDGE WIDTH
    # Measure total horizontal bone span at crest level
    # ------------------------------------------------------------------
    row = bone_near_teeth[cz, cx, :]
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
    # Measure minimum distance between root centroids
    # ------------------------------------------------------------------
    if is_molar:
        labeled_teeth, num_teeth = ndi.label(teeth_mask[cz])
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
    low_density = local_volume < 100
    bone_eroded = ndi.binary_erosion(bone_mask, iterations=1)
    candidate = low_density & bone_eroded
    labeled, num = ndi.label(candidate)
    lesion_detected = False
    lesion_size_mm3 = 0.0
    if num > 0:
        sizes = [np.sum(labeled == i) for i in range(1, num + 1)]
        max_size_px = max(sizes)
        voxel_volume_mm3 = sx * sy * sz
        max_size_mm3 = max_size_px * voxel_volume_mm3
        # Threshold: lesion must be larger than 80 voxels worth at 0.4mm
        threshold_mm3 = 80 * (0.4 ** 3)
        if max_size_mm3 > threshold_mm3:
            lesion_detected = True
            lesion_size_mm3 = round(max_size_mm3, 2)

    results["lesion_detected"] = lesion_detected
    results["lesion_size_mm3"] = lesion_size_mm3

    return results