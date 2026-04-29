import numpy as np
import scipy.ndimage as ndi


def extract_local_region(
    volume: np.ndarray,
    segmentation: np.ndarray,
    site_xyz: tuple[int, int, int],
    spacing: tuple[float, float, float],
    window_mm: float = 20.0
) -> tuple[np.ndarray, np.ndarray, tuple]:
    """
    Extract a 3D window around the implant site using real mm-based spacing.

    Args:
        volume:       Raw CBCT volume (HU values)
        segmentation: Segmentation output from ToothSeg (same shape as volume)
        site_xyz:     (z, x, y) voxel coordinate of implant site
        spacing:      (sx, sy, sz) voxel spacing in mm from CBCT loader
        window_mm:    Half-window size in mm (default 20mm = 40mm total box)

    Returns:
        local_volume: Cropped HU volume around implant site
        local_seg:    Cropped segmentation around implant site
        bounds:       ((z0,z1), (x0,x1), (y0,y1)) actual crop indices used
    """
    assert volume.shape == segmentation.shape, \
        "Volume and segmentation must have the same shape"
    assert all(s > 0 for s in spacing), \
        "Spacing values must be positive"

    z, x, y = site_xyz
    sx, sy, sz = spacing

    # Convert mm window to voxels per axis using real spacing
    wx = int(round(window_mm / sx))
    wy = int(round(window_mm / sy))
    wz = int(round(window_mm / sz))

    # Clamp to volume bounds
    z0 = max(0, z - wz)
    z1 = min(volume.shape[0], z + wz)
    x0 = max(0, x - wx)
    x1 = min(volume.shape[1], x + wx)
    y0 = max(0, y - wy)
    y1 = min(volume.shape[2], y + wy)

    local_volume = volume[z0:z1, x0:x1, y0:y1]
    local_seg = segmentation[z0:z1, x0:x1, y0:y1]
    bounds = ((z0, z1), (x0, x1), (y0, y1))

    assert local_volume.size > 0, \
        "Extracted region is empty — site_xyz may be out of bounds"

    return local_volume, local_seg, bounds


def get_best_slice(segmentation: np.ndarray) -> int:
    """
    Find the axial slice with the most tooth voxels.
    This replaces the hardcoded middle slice from the notebook.

    Args:
        segmentation: Full 3D segmentation volume

    Returns:
        z: Index of the best axial slice
    """
    teeth_mask = (segmentation == 1)
    teeth_per_slice = teeth_mask.sum(axis=(1, 2))
    z = int(np.argmax(teeth_per_slice))
    assert teeth_per_slice[z] > 0, \
        "No teeth found in segmentation — check model output"
    return z


def get_missing_tooth_location(
    segmentation: np.ndarray
) -> tuple[int, int, int]:
    """
    Detect the missing tooth location by finding the largest gap
    between tooth centroids on the best axial slice.

    NOTE: This is a placeholder used until YOLO model is ready.
    YOLO will replace this function entirely.

    Args:
        segmentation: Full 3D segmentation volume

    Returns:
        (z, x, y): Voxel coordinate of estimated implant site
    """
    z = get_best_slice(segmentation)
    teeth = (segmentation[z] == 1)

    labeled, num = ndi.label(teeth)
    assert num >= 2, \
        "Need at least 2 tooth regions to detect a gap"

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
