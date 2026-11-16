"""Resolve the actual filename before NumPy adds a suffix implicitly."""
from pathlib import Path


def npz_output_path(path):
    path=Path(path)
    return path if str(path).endswith('.npz') else Path(str(path)+'.npz')
