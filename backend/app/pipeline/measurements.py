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
    is_molar: bool = False,
    local_window: int = 60
) -> dict:
    """
    Compute all 5 clinical measurements using three orthogonal views.

    Measurement to view mapping:
        Apical Bone          -> Sagittal (z, y)
        Buccal Wall          -> Coronal  (z, x)
        Ridge Width          -> Coronal  (z, x)
        Interradicular Sep.  -> Axial    (y, x)
        Periapical Lesion    -> Sagittal (z, y)
    """
    assert volume.shape == segmentation.shape, \
        "Volume and segmentation shape mismatch"
    assert all(s > 0 for s in spacing), \
        "Spacing must be positive"

    sx, sy, sz = spacing
    views = extract_views(volume, segmentation, z, cx, cy)

    results = {}

    # Pre-compute bone masks and local windows
    sag_img  = views["sagittal"]["img"]
    sag_bone = sag_img > 150
    cor_img  = views["coronal"]["img"]
    cor_bone = cor_img > 150

    y_total  = sag_bone.shape[1]
    y0       = max(0, cy - local_window)
    y1       = min(y_total, cy + local_window)
    x_total  = cor_bone.shape[1]
    x0       = max(0, cx - local_window)
    x1       = min(x_total, cx + local_window)

    # ------------------------------------------------------------------
    # 1. APICAL BONE AVAILABILITY — Sagittal view
    # Anchor: lowest teeth voxel near implant site
    # Then measure bone after the edentulous gap
    # ------------------------------------------------------------------
    sag_seg   = views["sagittal"]["seg"]
    sag_teeth = (sag_seg == 1)

    teeth_in_window = sag_teeth[:, y0:y1]
    teeth_z_indices = np.where(teeth_in_window.any(axis=1))[0]
    teeth_near_site = teeth_z_indices[teeth_z_indices <= z + 10] \
        if len(teeth_z_indices) > 0 else []

    apex_z     = int(teeth_near_site[-1]) if len(teeth_near_site) > 0 else z
    cy_clamped = cy if cy < sag_bone.shape[1] else sag_bone.shape[1] // 2

    # Scan downward from apex — max 120 voxels (~48mm)
    max_scan = 120
    z_end    = min(volume.shape[0], apex_z + max_scan)
    col_hu   = sag_img[apex_z:z_end, cy_clamped].astype(np.float32)
    col_bone = col_hu > 150

    # Step 1: Find first low-density gap (>= 3 consecutive voxels < 150 HU)
    gap_start = None
    gap_count = 0
    for i, is_bone in enumerate(col_bone):
        if not is_bone:
            gap_count += 1
            if gap_count >= 3 and gap_start is None:
                gap_start = i - gap_count + 1
        else:
            gap_count = 0

    if gap_start is None:
        results["apical_bone_mm"] = 0.0
    else:
        # Step 2: After the gap, find where bone resumes
        bone_resume = None
        for i in range(gap_start, len(col_bone)):
            if col_bone[i]:
                bone_resume = i
                break

        if bone_resume is None:
            results["apical_bone_mm"] = 0.0
        else:
            # Step 3: Measure continuous bone until next barrier
            # Use stricter threshold for second barrier detection
            col_bone_strict = col_hu > 200
            next_barrier    = None
            barrier_count   = 0
            for i in range(bone_resume, len(col_bone_strict)):
                if not col_bone_strict[i]:
                    barrier_count += 1
                    if barrier_count >= 1:
                        next_barrier = i - barrier_count + 1
                        break
                else:
                    barrier_count = 0

            if next_barrier is None:
                max_apical_voxels = int(25.0 / sz)
                apical_px = min(len(col_bone) - bone_resume, max_apical_voxels)
            else:
                apical_px = next_barrier - bone_resume

            apical_mm = float(apical_px) * sz
            assert apical_mm >= 0, "Apical bone cannot be negative"
            results["apical_bone_mm"] = round(apical_mm, 2)

    # ------------------------------------------------------------------
    # 2. BUCCAL WALL THICKNESS — Coronal view
    # Measure bone to the left of cx (buccal side) at implant level
    # ------------------------------------------------------------------
    row       = cor_bone[z, x0:cx]
    indices   = np.where(row > 0)[0]
    buccal_px = len(indices) if len(indices) > 0 else 0
    buccal_mm = float(buccal_px) * sx
    assert buccal_mm >= 0, "Buccal thickness cannot be negative"
    results["buccal_wall_mm"] = round(buccal_mm, 2)

    # ------------------------------------------------------------------
    # 3. BUCCOLINGUAL RIDGE WIDTH — Coronal view
    # Measure total horizontal bone span at crest level
    # ------------------------------------------------------------------
    row     = cor_bone[z, x0:x1]
    indices = np.where(row > 0)[0]
    if len(indices) > 1:
        ridge_px = int(indices[-1] - indices[0])
    else:
        ridge_px = 0
    ridge_mm = float(ridge_px) * sx
    assert ridge_mm >= 0, "Ridge width cannot be negative"
    results["ridge_width_mm"] = round(ridge_mm, 2)

    # ------------------------------------------------------------------
    # 4. INTERRADICULAR SEPTUM WIDTH — Axial view (molars only)
    # ------------------------------------------------------------------
    if is_molar:
        ax_seg        = views["axial"]["seg"]
        ax_teeth      = (ax_seg == 1)
        labeled_teeth, num_teeth = ndi.label(ax_teeth)
        if num_teeth >= 2:
            centers = []
            for i in range(1, num_teeth + 1):
                coords = np.argwhere(labeled_teeth == i)
                centers.append(coords.mean(axis=0))
            centers = np.array(centers)
            dists   = []
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
    # 5. PERIAPICAL LESION STATUS — Sagittal view
    # Detect low-density regions inside bone near apex
    # ------------------------------------------------------------------
    sag_bone_local = np.zeros_like(sag_bone)
    sag_bone_local[max(0, z - local_window):z + local_window, y0:y1] = \
        sag_bone[max(0, z - local_window):z + local_window, y0:y1]

    low_density  = sag_img < 100
    bone_eroded  = ndi.binary_erosion(sag_bone_local, iterations=1)
    candidate    = low_density & bone_eroded
    labeled, num = ndi.label(candidate)

    lesion_detected = False
    lesion_size_mm3 = 0.0
    if num > 0:
        sizes        = [np.sum(labeled == i) for i in range(1, num + 1)]
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