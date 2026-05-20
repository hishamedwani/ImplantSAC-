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


def measure_cortical_thickness(bone_row: np.ndarray, spacing_mm: float) -> float:
    """
    Measure the thickness of the outermost continuous bone layer.
    Finds the first bone region from the edge and measures its width.
    Max capped at 5mm which is the clinical maximum for cortical plate.
    """
    indices = np.where(bone_row)[0]
    if len(indices) == 0:
        return 0.0

    # Find the outermost continuous bone strip
    # Start from the first bone pixel and count continuous pixels
    start = indices[0]
    count = 1
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            count += 1
        else:
            break

    thickness_mm = count * spacing_mm
    # Cap at 5mm — buccal cortical plate cannot exceed this clinically
    return float(round(min(thickness_mm, 5.0), 2))


def compute_measurements(
    volume: np.ndarray,
    segmentation: np.ndarray,
    z: int,
    cx: int,
    cy: int,
    spacing: tuple,
    is_molar: bool = False,
    local_window: int = 60
) -> dict:
    """
    Compute all 5 clinical measurements using three orthogonal views.
    spacing = (spacing_x, spacing_y, spacing_z) in mm
    Volume axes: (z, y, x)
    """
    assert volume.shape == segmentation.shape, \
        "Volume and segmentation shape mismatch"

    # Correct spacing assignment
    # spacing tuple is (x, y, z) but volume is (z, y, x)
    sp_x, sp_y, sp_z = spacing  # mm per voxel in x, y, z directions

    views = extract_views(volume, segmentation, z, cx, cy)
    results = {}

   # ------------------------------------------------------------------
    # 1. APICAL BONE AVAILABILITY — Sagittal view
    # Sample multiple columns around centroid and take maximum
    # ------------------------------------------------------------------
    sag_img = views["sagittal"]["img"]
    y_total = sag_img.shape[1]
    y0      = max(0, cy - local_window // 2)
    y1      = min(y_total, cy + local_window // 2)

    max_scan    = min(150, volume.shape[0] - z - 1)
    best_apical = 0.0

    if max_scan > 0:
        # Sample 5 columns around cy
        sample_cols = [
            max(0, min(cy - 4, y_total - 1)),
            max(0, min(cy - 2, y_total - 1)),
            min(cy, y_total - 1),
            max(0, min(cy + 2, y_total - 1)),
            max(0, min(cy + 4, y_total - 1)),
        ]

        for col_idx in sample_cols:
            col_hu   = sag_img[z:z + max_scan, col_idx].astype(np.float32)
            col_bone = col_hu > 200

            # Find socket end — where bone starts after socket
            socket_end      = 0
            consecutive     = 0
            for i, is_bone in enumerate(col_bone):
                if is_bone:
                    consecutive += 1
                    if consecutive >= 2:
                        socket_end = i - consecutive + 1
                        break
                else:
                    consecutive = 0

            # Measure continuous bone from socket_end
            bone_count   = 0
            prev_bone    = False
            gap_allowed  = 2

            for i in range(socket_end, len(col_hu)):
                hu = col_hu[i]
                # Stop at sinus/nasal floor
                if hu > 900 and i > socket_end + 2:
                    break
                # Stop at IAN canal
                if hu < 50 and prev_bone and i > socket_end + 5:
                    if i + 1 < len(col_hu) and col_hu[i + 1] < 100:
                        break

                if hu > 200:
                    bone_count += 1
                    prev_bone   = True
                    gap_allowed = 2
                else:
                    prev_bone = False
                    if gap_allowed > 0:
                        bone_count  += 1
                        gap_allowed -= 1
                    else:
                        break

            apical_mm = bone_count * sp_z
            apical_mm = min(apical_mm, 20.0)
            if apical_mm > best_apical:
                best_apical = apical_mm

    results["apical_bone_mm"] = round(best_apical, 2)
    # ------------------------------------------------------------------
    # 2. BUCCAL WALL THICKNESS — Coronal view
    # Measure the outermost cortical bone on the buccal (outer) side
    # sp_x = mm per voxel in x direction
    # ------------------------------------------------------------------
    cor_img = views["coronal"]["img"]
    cor_bone = cor_img > 200  # Cortical bone threshold

    # Measure at 1mm apical to crest (slightly below z)
    apical_offset = max(1, int(1.0 / sp_z))
    z_buccal = min(z + apical_offset, cor_bone.shape[0] - 1)

    # Try both sides of cx — buccal could be left or right
    # depending on which arch and which side of the jaw
    window_buccal = int(30.0 / sp_x)  # 30mm search window

    # Left side (buccal for right side of jaw)
    x_left_start = max(0, cx - window_buccal)
    row_left = cor_bone[z_buccal, x_left_start:cx]

    # Right side (buccal for left side of jaw)
    x_right_end = min(cor_bone.shape[1], cx + window_buccal)
    row_right = cor_bone[z_buccal, cx:x_right_end]

    # Measure both and take the thinner one (buccal plate is thinner than lingual)
    thickness_left  = measure_cortical_thickness(row_left,          sp_x)
    thickness_right = measure_cortical_thickness(row_right[::-1],   sp_x)

    # Buccal wall is the minimum of the two sides
    # (buccal cortical plate is always thinner than lingual)
    if thickness_left > 0 and thickness_right > 0:
        buccal_mm = min(thickness_left, thickness_right)
    elif thickness_left > 0:
        buccal_mm = thickness_left
    elif thickness_right > 0:
        buccal_mm = thickness_right
    else:
        buccal_mm = 0.0

    results["buccal_wall_mm"] = float(round(buccal_mm, 2))

    # ------------------------------------------------------------------
    # 3. BUCCOLINGUAL RIDGE WIDTH — Coronal view
    # Clinical: total horizontal ridge width at crest level
    # Find outer edges of cortical bone on both sides of cx
    # ------------------------------------------------------------------
    crest_z  = max(0, z - 3)
    row_full = (cor_img[crest_z, :] > 200).astype(np.uint8)

    # Find the leftmost outer edge of bone to the left of cx
    # Scan left from cx — find where bone STARTS (outer buccal edge)
    left_outer = cx
    in_bone = False
    for i in range(cx, max(0, cx - 80), -1):
        if row_full[i] == 1:
            in_bone = True
            left_outer = i
        elif in_bone:
            # Bone ended — this is the outer edge
            break

    # Find the rightmost outer edge of bone to the right of cx
    right_outer = cx
    in_bone = False
    for i in range(cx, min(cor_img.shape[1], cx + 80)):
        if row_full[i] == 1:
            in_bone = True
            right_outer = i
        elif in_bone:
            break

    ridge_px = right_outer - left_outer
    ridge_mm = ridge_px * sp_x

    # Clinical cap: 12mm maximum for alveolar ridge
    results["ridge_width_mm"] = float(round(min(ridge_mm, 12.0), 2))


# ------------------------------------------------------------------
    # 3. SEPTUM WIDTH — Axial view

    if is_molar:
        # Try axial view first — better for root separation in molars
        ax_seg_molar   = views["axial"]["seg"]
        ax_teeth_molar = (ax_seg_molar == 1)

        # Label connected tooth components in axial view
        labeled_ax, num_ax = ndi.label(ax_teeth_molar)

        if num_ax >= 2:
            # Find closest edges between any two tooth components
            min_gap_mm = float('inf')
            for i in range(1, num_ax + 1):
                for j in range(i + 1, num_ax + 1):
                    coords_i = np.argwhere(labeled_ax == i)
                    coords_j = np.argwhere(labeled_ax == j)
                    # Find minimum distance between edges
                    # Use bounding box approach for speed
                    i_x_max = coords_i[:, 1].max()
                    j_x_min = coords_j[:, 1].min()
                    i_y_max = coords_i[:, 0].max()
                    j_y_min = coords_j[:, 0].min()
                    
                    # Measure gap between inner edges of roots
                    gap_x = max(0, j_x_min - i_x_max) * sp_x
                    gap_y = max(0, j_y_min - i_y_max) * sp_y
                    # Take the smaller axis gap — septum is measured perpendicular
                    if gap_x > 0 and gap_y > 0:
                        gap = min(gap_x, gap_y)
                    else:
                        gap = max(gap_x, gap_y)
                    
                    if 0 < gap < min_gap_mm:
                        min_gap_mm = gap

            if min_gap_mm == float('inf') or min_gap_mm == 0:
                # Roots touching — measure bone between centroids
                centers = []
                for i in range(1, num_ax + 1):
                    coords = np.argwhere(labeled_ax == i)
                    centers.append(coords.mean(axis=0))
                centers = np.array(centers)
                dists = []
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        d = np.linalg.norm(
                            (centers[i] - centers[j]) * np.array([sp_y, sp_x])
                        )
                        dists.append(d)
                min_gap_mm = float(min(dists)) * 0.3  # bone between = ~30% of centroid distance
            
            results["septum_width_mm"] = float(round(min(min_gap_mm, 10.0), 2))
        else:
            # Only one component found — try scanning for gap manually
            teeth_cols = np.where(ax_teeth_molar.any(axis=0))[0]
            if len(teeth_cols) >= 2:
                # Find largest internal gap
                gaps = np.diff(teeth_cols)
                if gaps.max() > 2:
                    sep_px = int(gaps.max())
                    results["septum_width_mm"] = float(round(sep_px * sp_x, 2))
                else:
                    results["septum_width_mm"] = 0.0
            else:
                results["septum_width_mm"] = 0.0
    else:
        results["septum_width_mm"] = None

    # ------------------------------------------------------------------
    # 5. PERIAPICAL LESION STATUS — Sagittal view
    # Clinical hint: measure largest dimension of radiolucency around apex
    # ------------------------------------------------------------------
    sag_bone_mask = sag_img > 150
    local_z0 = max(0, z - local_window // 2)
    local_z1 = min(sag_img.shape[0], z + local_window // 2)

    sag_bone_local = np.zeros_like(sag_bone_mask)
    sag_bone_local[local_z0:local_z1, y0:y1] = \
        sag_bone_mask[local_z0:local_z1, y0:y1]

    # Radiolucency = low density region enclosed within bone
    low_density  = sag_img < 80
    bone_eroded  = ndi.binary_erosion(sag_bone_local, iterations=2)
    candidate    = low_density & bone_eroded
    labeled, num = ndi.label(candidate)

    lesion_detected  = False
    lesion_size_mm3  = 0.0

    if num > 0:
        largest_dim_mm = 0.0
        for i in range(1, num + 1):
            region = np.argwhere(labeled == i)
            if len(region) < 3:
                continue
            # Largest dimension = max extent in any axis
            z_extent = (region[:, 0].max() - region[:, 0].min() + 1) * sp_z
            y_extent = (region[:, 1].max() - region[:, 1].min() + 1) * sp_y
            largest_dim = max(z_extent, y_extent)
            if largest_dim > largest_dim_mm:
                largest_dim_mm = largest_dim

        # Clinical threshold: lesion must be >= 3mm largest dimension
        if largest_dim_mm >= 3.0:
            lesion_detected = True
            lesion_size_mm3 = float(round(largest_dim_mm, 2))

    results["lesion_detected"] = lesion_detected
    results["lesion_size_mm3"] = lesion_size_mm3

    return results