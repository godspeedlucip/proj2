"""SimpleITK NIfTI I/O for Windows paths containing non-ASCII characters."""

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import SimpleITK as sitk


def _nifti_suffix_for_staging(path):
    if os.name == "nt" and not str(path.absolute()).isascii():
        name = path.name.lower()
        return next((ending for ending in (".nii.gz", ".nii") if name.endswith(ending)), None)
    return None


def _staging_path(directory, suffix):
    staged = Path(directory) / ("volume" + suffix)
    if not str(staged).isascii():
        raise OSError("SimpleITK on Windows requires an ASCII TEMP directory to read/write this NIfTI path.")
    return staged


def read_image(path):
    path = Path(path)
    suffix = _nifti_suffix_for_staging(path)
    if suffix is None:
        return sitk.ReadImage(str(path))
    with TemporaryDirectory(prefix="nnunet-tam-") as temporary:
        staged = _staging_path(temporary, suffix)
        shutil.copyfile(path, staged)
        return sitk.ReadImage(str(staged))


def write_image(image, path):
    path = Path(path)
    suffix = _nifti_suffix_for_staging(path)
    if suffix is None:
        sitk.WriteImage(image, str(path))
    else:
        with TemporaryDirectory(prefix="nnunet-tam-") as temporary:
            staged = _staging_path(temporary, suffix)
            sitk.WriteImage(image, str(staged))
            shutil.copyfile(staged, path)
