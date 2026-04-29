import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk


def load_cbct(file_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load a CBCT scan from .nii, .nii.gz, .mha, or DICOM directory.
    
    Returns:
        volume: 3D numpy array of voxel intensities (HU values)
        spacing: tuple of (x, y, z) voxel spacing in millimeters
    """
    ext = _get_extension(file_path)

    if ext in [".nii", ".nii.gz"]:
        return _load_nifti(file_path)
    elif ext == ".mha":
        return _load_mha(file_path)
    elif os.path.isdir(file_path):
        return _load_dicom(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _get_extension(file_path: str) -> str:
    """Handle double extensions like .nii.gz"""
    if file_path.endswith(".nii.gz"):
        return ".nii.gz"
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def _load_nifti(file_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.load(file_path)
    volume = np.array(img.get_fdata(), dtype=np.float32)
    zooms = img.header.get_zooms()
    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    assert all(s > 0 for s in spacing), "Invalid spacing values in NIfTI file"
    return volume, spacing


def _load_mha(file_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing_sitk = img.GetSpacing()
    spacing = (float(spacing_sitk[0]), float(spacing_sitk[1]), float(spacing_sitk[2]))
    assert all(s > 0 for s in spacing), "Invalid spacing values in MHA file"
    return volume, spacing


def _load_dicom(folder_path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    if not dicom_names:
        raise ValueError(f"No DICOM files found in directory: {folder_path}")
    reader.SetFileNames(dicom_names)
    img = reader.Execute()
    volume = sitk.GetArrayFromImage(img).astype(np.float32)
    spacing_sitk = img.GetSpacing()
    spacing = (float(spacing_sitk[0]), float(spacing_sitk[1]), float(spacing_sitk[2]))
    assert all(s > 0 for s in spacing), "Invalid spacing values in DICOM files"
    return volume, spacing
