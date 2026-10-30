"""Read, write, and display common medical-image formats.

The public API intentionally keeps image data as NumPy arrays while retaining
spatial metadata when the source format provides it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class TigreTool:
    """TIGRE helpers plus independent per-instance geometry and angle state."""

    DEFAULT_START_ANGLE = 0.0
    DEFAULT_END_ANGLE = 360.0
    DEFAULT_N_ANGLES = 360
    geometry: Any | None = None
    geo: Any | None = None
    angles: np.ndarray | None = None
    start_angle: float | None = None
    end_angle: float | None = None
    n_angles: int | None = None

    def __init__(self, geometry: Any | None = None, angles: np.ndarray | None = None):
        """Create an independent TIGRE configuration holder.

        Each instance owns its geometry and angle array, so multiple
        configurations can coexist in one notebook without overwriting one
        another.
        """
        self.geometry = geometry
        self.geo = geometry
        self.angles = None if angles is None else np.asarray(angles, dtype=np.float32)
        self.start_angle = None
        self.end_angle = None
        self.n_angles = None if angles is None else int(len(self.angles))

    def set_geometry(
        self,
        volume_shape: Iterable[int],
        voxel_spacing: Iterable[float],
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        mode: str = "cone",
    ):
        """Build and store this instance's TIGRE geometry."""
        self.geometry = self.build_tigre_geometry(
            volume_shape,
            voxel_spacing,
            n_detector=n_detector,
            d_detector=d_detector,
            dso=dso,
            dsd=dsd,
            mode=mode,
        )
        self.geo = self.geometry
        return self.geometry

    def set_angles(
        self,
        start_angle: float | None = None,
        end_angle: float | None = None,
        angles: int | None = None,
    ) -> np.ndarray:
        """Create and store angles from ``(start_angle, end_angle, angles)``.

        Angle bounds are specified in degrees, matching ``make_angles``.
        ``angles`` is the number of views, and the end angle is excluded.
        """
        start_angle = self.DEFAULT_START_ANGLE if start_angle is None else start_angle
        end_angle = self.DEFAULT_END_ANGLE if end_angle is None else end_angle
        angles = self.DEFAULT_N_ANGLES if angles is None else angles
        self.angles = self.make_angles(angles, start_angle, end_angle)
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)
        self.n_angles = int(angles)
        return self.angles

    @staticmethod
    def hu_to_mu(hu: np.ndarray, mu_water: float = 0.02) -> np.ndarray:
        """Convert Hounsfield units to non-negative linear attenuation values."""
        hu = np.asarray(hu, dtype=np.float32)
        return np.maximum(mu_water * (1.0 + hu / 1000.0), 0.0).astype(np.float32)

    @staticmethod
    def make_angles(
        n_angles: int,
        start_deg: float = 0.0,
        end_deg: float = 360.0,
    ) -> np.ndarray:
        """Return evenly spaced TIGRE angles in radians without the endpoint."""
        if n_angles < 1:
            raise ValueError("n_angles must be positive")
        return np.deg2rad(
            np.linspace(start_deg, end_deg, n_angles, endpoint=False)
        ).astype(np.float32)

    @staticmethod
    def build_tigre_geometry(
        volume_shape: Iterable[int],
        voxel_spacing: Iterable[float],
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        mode: str = "cone",
    ):
        """Create a TIGRE geometry using the ``(z, y, x)`` convention."""
        import tigre

        geo = tigre.geometry()
        geo.mode = mode
        geo.DSO = float(dso)
        geo.DSD = float(dsd)
        geo.nDetector = np.asarray(tuple(n_detector), dtype=np.int32)
        geo.dDetector = np.asarray(tuple(d_detector), dtype=np.float32)
        geo.sDetector = geo.nDetector * geo.dDetector
        geo.nVoxel = np.asarray(tuple(volume_shape), dtype=np.int32)
        geo.dVoxel = np.asarray(tuple(voxel_spacing), dtype=np.float32)
        geo.sVoxel = geo.nVoxel * geo.dVoxel
        geo.offOrigin = np.zeros(3, dtype=np.float32)
        geo.offDetector = np.zeros(2, dtype=np.float32)
        geo.rotDetector = np.zeros(3, dtype=np.float32)
        geo.COR = 0.0
        geo.accuracy = 0.5
        return geo

    set_geo = build_tigre_geometry

    @staticmethod
    def project_tigre(
        volume: np.ndarray,
        geometry: Any,
        angles: np.ndarray,
    ) -> np.ndarray:
        """Simulate cone-beam projections with TIGRE's ``Ax`` operator."""
        import tigre

        projections = tigre.Ax(
            np.asarray(volume, dtype=np.float32),
            geometry,
            np.asarray(angles, dtype=np.float32),
        )
        return np.asarray(projections, dtype=np.float32)

    def project(
        self,
        volume: np.ndarray,
        geometry: Any | None = None,
        angles: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project using stored state unless geometry or angles are provided."""
        geometry = self.geometry if geometry is None else geometry
        angles = self.angles if angles is None else angles
        if geometry is None or angles is None:
            raise ValueError("configure geometry and angles before calling project")
        return self.project_tigre(volume, geometry, angles)

    @staticmethod
    def reconstruct_tigre(
        projections: np.ndarray,
        geometry: Any,
        angles: np.ndarray,
        use_filter: bool = True,
    ) -> np.ndarray:
        """Reconstruct projections with TIGRE FDK or its unfiltered variant."""
        import tigre.algorithms as algorithms

        projections = np.asarray(projections, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        if use_filter:
            reconstruction = algorithms.fdk(projections, geometry, angles)
        else:
            reconstruction = algorithms.fdk_no_filter(projections, geometry, angles)
        return np.asarray(reconstruction, dtype=np.float32)

    def reconstruct(
        self,
        projections: np.ndarray,
        geometry: Any | None = None,
        angles: np.ndarray | None = None,
        use_filter: bool = True,
    ) -> np.ndarray:
        """Reconstruct using stored state unless geometry or angles are provided."""
        geometry = self.geometry if geometry is None else geometry
        angles = self.angles if angles is None else angles
        if geometry is None or angles is None:
            raise ValueError("configure geometry and angles before calling reconstruct")
        return self.reconstruct_tigre(projections, geometry, angles, use_filter=use_filter)

    @staticmethod
    def reconstruct_volume(
        volume: np.ndarray,
        spacing: Iterable[float],
        angles: np.ndarray,
        *,
        input_domain: str = "hu",
        n_detector: Iterable[int] = (512, 512),
        d_detector: Iterable[float] = (0.3, 0.3),
        dso: float = 780.0,
        dsd: float = 1200.0,
        use_filter: bool = True,
        return_projection: bool = False,
    ):
        """Project and reconstruct one volume using the shared TIGRE helpers."""
        if input_domain == "hu":
            volume_for_projection = TigreTool.hu_to_mu(volume)
        elif input_domain == "mu":
            volume_for_projection = np.asarray(volume, dtype=np.float32)
        else:
            raise ValueError("input_domain must be 'hu' or 'mu'")

        geometry = TigreTool.build_tigre_geometry(
            volume_for_projection.shape,
            spacing,
            n_detector=n_detector,
            d_detector=d_detector,
            dso=dso,
            dsd=dsd,
        )
        projections = TigreTool.project_tigre(volume_for_projection, geometry, angles)
        reconstruction = TigreTool.reconstruct_tigre(
            projections,
            geometry,
            angles,
            use_filter=use_filter,
        )
        if return_projection:
            return reconstruction, projections, geometry
        return reconstruction, geometry

# Backward-compatible function aliases for existing scripts and notebooks.
hu_to_mu = TigreTool.hu_to_mu
make_angles = TigreTool.make_angles
build_tigre_geometry = TigreTool.build_tigre_geometry
set_geo = TigreTool.set_geo
project_tigre = TigreTool.project_tigre
reconstruct_tigre = TigreTool.reconstruct_tigre
reconstruct_volume = TigreTool.reconstruct_volume


@dataclass
class MedicalImage:
    """An image array plus optional spatial metadata."""

    data: np.ndarray
    affine: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if self.affine is not None:
            self.affine = np.asarray(self.affine)


class MedicalImageTool:
    """Format-aware medical image I/O and 2D slice visualization."""

    _NUMPY_SUFFIXES = {".npy", ".npz"}
    _TIFF_SUFFIXES = {".tif", ".tiff"}
    _RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
    _NIFTI_SUFFIXES = {".nii", ".nii.gz"}

    @classmethod
    def read(cls, path: str | Path, *, npz_key: str | None = None) -> MedicalImage:
        path = Path(path)
        if path.is_dir():
            return cls._read_dicom_series(path)
        suffix = cls._suffix(path)
        if suffix == ".npy":
            return MedicalImage(np.load(path, allow_pickle=False))
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if not archive.files:
                    raise ValueError(f"NPZ archive is empty: {path}")
                key = npz_key if npz_key is not None else (
                    "data" if "data" in archive.files else archive.files[0]
                )
                if key not in archive.files:
                    raise KeyError(f"NPZ key '{key}' not found in {path}; available={archive.files}")
                return MedicalImage(archive[key], metadata={"npz_key": key})
        if suffix in cls._TIFF_SUFFIXES:
            import tifffile

            return MedicalImage(tifffile.imread(path))
        if suffix in cls._RASTER_SUFFIXES:
            from PIL import Image

            return MedicalImage(np.asarray(Image.open(path)))
        if suffix in cls._NIFTI_SUFFIXES:
            import nibabel as nib

            image = nib.load(str(path))
            return MedicalImage(
                np.asanyarray(image.dataobj),
                affine=image.affine,
                metadata={"header": image.header.copy(), "format": "nifti"},
            )
        if suffix == ".dcm":
            return cls._read_dicom_file(path)
        raise cls._unsupported(path)

    @classmethod
    def save(
        cls,
        image: MedicalImage | np.ndarray,
        path: str | Path,
        *,
        affine: np.ndarray | None = None,
        voxel_spacing: Sequence[float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        reference_dicom: str | Path | Any | None = None,
    ) -> Path:
        """Save an image, creating parent directories when necessary.

        ``voxel_spacing`` is interpreted in the same axis order as ``image``
        and is used to build a diagonal NIfTI affine when no explicit
        ``affine`` is provided. An explicit affine always takes precedence.

        For DICOM, ``reference_dicom`` may be a source path, a pydicom Dataset,
        or metadata returned by :meth:`read`. A reference is required so that
        the output preserves the source study's clinical metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        item = image if isinstance(image, MedicalImage) else MedicalImage(image, affine, dict(metadata or {}))
        if isinstance(image, MedicalImage):
            if affine is not None:
                item.affine = np.asarray(affine)
            if metadata:
                item.metadata.update(metadata)
        spacing_array = None
        if voxel_spacing is not None:
            spacing_array = np.asarray(voxel_spacing, dtype=np.float64)
            if (
                spacing_array.shape != (3,)
                or not np.all(np.isfinite(spacing_array))
                or np.any(spacing_array <= 0)
            ):
                raise ValueError(
                    "voxel_spacing must contain three finite positive values"
                )
        suffix = cls._suffix(path)
        if suffix == ".npy":
            np.save(path, item.data)
        elif suffix == ".npz":
            np.savez_compressed(path, data=item.data)
        elif suffix in cls._TIFF_SUFFIXES:
            import tifffile

            tifffile.imwrite(path, item.data, photometric="minisblack")
        elif suffix in cls._RASTER_SUFFIXES:
            from PIL import Image

            if item.data.ndim != 2:
                raise ValueError("PNG/JPEG/BMP output requires a 2D array")
            Image.fromarray(_to_image_array(item.data)).save(path)
        elif suffix in cls._NIFTI_SUFFIXES:
            import nibabel as nib

            if item.affine is not None:
                nifti_affine = item.affine
            elif spacing_array is not None:
                nifti_affine = np.diag(np.append(spacing_array, 1.0))
            else:
                nifti_affine = np.eye(4)
            header = item.metadata.get("header")
            nii = nib.Nifti1Image(item.data, nifti_affine, header=header)
            if spacing_array is not None:
                nii.header.set_xyzt_units(xyz="mm")
            nib.save(nii, str(path))
        elif suffix == ".dcm":
            cls._save_dicom(item, path, reference_dicom)
        else:
            raise cls._unsupported(path)
        return path

    @classmethod
    def show_slice(
        cls,
        image: MedicalImage | np.ndarray,
        *,
        axis: int = -1,
        index: int | None = None,
        cmap: str = "gray",
        window: tuple[float, float] | None = None,
        title: str | None = None,
        colorbar: bool = False,
        show: bool = False,
        save_path: str | Path | None = None,
        figsize: tuple[float, float] = (6.0, 6.0),
    ):
        """Display one slice and return ``(figure, axes)``."""
        import matplotlib.pyplot as plt

        data = image.data if isinstance(image, MedicalImage) else np.asarray(image)
        if data.ndim < 2:
            raise ValueError("show_slice requires an array with at least 2 dimensions")
        axis = axis if axis >= 0 else data.ndim + axis
        if axis < 0 or axis >= data.ndim:
            raise ValueError(f"axis must be in [0, {data.ndim - 1}]")
        axis_size = data.shape[axis]
        index = axis_size // 2 if index is None else index
        if index < 0:
            index += axis_size
        if index < 0 or index >= axis_size:
            raise IndexError(f"index must be in [0, {axis_size - 1}]")
        slice_data = np.take(data, index, axis=axis)
        vmin, vmax = window if window is not None else (None, None)
        figure, axes = plt.subplots(figsize=figsize)
        axes.imshow(np.squeeze(slice_data), cmap=cmap, vmin=vmin, vmax=vmax)
        axes.axis("off")
        if title:
            axes.set_title(title)
        if colorbar:
            figure.colorbar(axes.images[0], ax=axes)
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        return figure, axes

    @staticmethod
    def _suffix(path: Path) -> str:
        name = path.name.lower()
        return ".nii.gz" if name.endswith(".nii.gz") else path.suffix.lower()

    @staticmethod
    def _unsupported(path: Path) -> ValueError:
        supported = ".nii/.nii.gz, .dcm, DICOM directory, .tif/.tiff, .npy/.npz, .png/.jpg/.jpeg/.bmp"
        return ValueError(f"Unsupported image format for '{path}'. Supported formats: {supported}")

    @staticmethod
    def _read_dicom_file(path: Path) -> MedicalImage:
        import pydicom

        dataset = pydicom.dcmread(path)
        return MedicalImage(
            dataset.pixel_array,
            metadata={"dicom_dataset": dataset, "format": "dicom"},
        )

    @classmethod
    def _read_dicom_series(cls, directory: Path) -> MedicalImage:
        import pydicom
        from pydicom.errors import InvalidDicomError

        datasets = []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            try:
                dataset = pydicom.dcmread(path, stop_before_pixels=False)
                if hasattr(dataset, "PixelData"):
                    datasets.append(dataset)
            except (InvalidDicomError, OSError):
                continue
        if not datasets:
            raise ValueError(f"No readable DICOM images found in '{directory}'")
        datasets.sort(key=_dicom_sort_key)
        return MedicalImage(
            np.stack([dataset.pixel_array for dataset in datasets]),
            metadata={"dicom_datasets": datasets, "format": "dicom_series"},
        )

    @staticmethod
    def _save_dicom(item: MedicalImage, path: Path, reference_dicom: Any | None) -> None:
        import pydicom

        reference = reference_dicom or item.metadata.get("dicom_dataset")
        if reference is None:
            raise ValueError("DICOM output requires reference_dicom or source DICOM metadata")
        dataset = pydicom.dcmread(str(reference)) if isinstance(reference, (str, Path)) else reference.copy()
        data = np.asarray(item.data)
        if data.ndim != 2:
            raise ValueError("A .dcm target accepts one 2D slice; save a series separately")
        if data.dtype.kind == "f":
            data = np.rint(data).astype(np.int16)
        dataset.Rows, dataset.Columns = data.shape
        dataset.PixelData = np.ascontiguousarray(data).tobytes()
        dataset.save_as(path)


def _dicom_sort_key(dataset: Any) -> tuple[Any, ...]:
    position = getattr(dataset, "ImagePositionPatient", None)
    if position is not None and len(position) >= 3:
        return (0, float(position[2]))
    return (1, int(getattr(dataset, "InstanceNumber", 0)))


def _to_image_array(data: np.ndarray) -> np.ndarray:
    if data.dtype == np.uint8 or data.dtype == np.uint16:
        return data
    data = np.asarray(data, dtype=np.float32)
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if high <= low:
        return np.zeros(data.shape, dtype=np.uint8)
    return np.clip((data - low) / (high - low) * 255, 0, 255).astype(np.uint8)


class SimulationDataTool:
    """Data-oriented helpers used by simulation and reconstruction scripts."""

    @staticmethod
    def save_as_gif(
        sequences: np.ndarray,
        file_name: str | Path,
        minimum: float,
        maximum: float,
        duration: float = 0.5,
        loop: int = 0,
    ) -> Path:
        """Save a sequence of arrays as an 8-bit animated GIF."""
        if maximum <= minimum:
            raise ValueError("maximum must be greater than minimum")
        import imageio.v2 as imageio

        data = np.asarray(sequences, dtype=np.float32)
        data = np.clip(data, minimum, maximum)
        frames = np.rint((data - minimum) / (maximum - minimum) * 255).astype(np.uint8)
        output = Path(file_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(output, frames, duration=duration, loop=loop)
        return output

    @staticmethod
    def read_raw_data(
        file_name: str | Path,
        width: int,
        height: int,
        data_type: str | np.dtype = np.float32,
        mode: str = "hw",
    ) -> tuple[int, np.ndarray]:
        """Read a flat binary volume and return ``(slice_count, volume)``."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        values = np.fromfile(file_name, dtype=data_type)
        plane_size = width * height
        if values.size % plane_size:
            raise ValueError("raw file size is not divisible by width * height")
        slice_count = values.size // plane_size
        if mode == "hw":
            shape = (slice_count, height, width)
        elif mode == "wh":
            shape = (slice_count, width, height)
        else:
            raise ValueError("mode must be 'hw' or 'wh'")
        return slice_count, values.reshape(shape)

    @classmethod
    def read_raw_data_all(
        cls,
        directory: str | Path,
        width: int = 512,
        height: int = 512,
        start_index: int = 8,
        end_index: int = -4,
        data_type: str | np.dtype = np.float32,
    ) -> np.ndarray:
        """Read one 2-D raw frame from every numerically sortable file."""
        paths = sorted(
            (path for path in Path(directory).iterdir() if path.is_file()),
            key=lambda path: _numeric_sort_key(path.name, start_index, end_index),
        )
        if not paths:
            raise ValueError(f"No raw files found in '{directory}'")
        frames = []
        for path in paths:
            count, values = cls.read_raw_data(path, width, height, data_type)
            if count != 1:
                raise ValueError(f"Expected one frame per file, got {count} in '{path}'")
            frames.append(values[0])
        return np.stack(frames).astype(np.float32, copy=False)

    @staticmethod
    def read_dicom_file(file_name: str | Path):
        """Read one DICOM file and return ``(dataset, float32_pixels)``."""
        import pydicom

        dataset = pydicom.dcmread(file_name, force=True)
        return dataset, np.asarray(dataset.pixel_array, dtype=np.float32)

    @staticmethod
    def listsorter(directory: str | Path, start_index: int, end_index: int) -> list[str]:
        """Return filenames sorted by the numeric substring used by old scripts."""
        names = [path.name for path in Path(directory).iterdir() if path.is_file()]
        return sorted(names, key=lambda name: _numeric_sort_key(name, start_index, end_index))

    @classmethod
    def read_dicom_all(
        cls,
        directory: str | Path,
        sort_start: int,
        sort_end: int,
    ):
        """Read a sorted DICOM series and return ``(last_dataset, volume)``."""
        paths = [Path(directory) / name for name in cls.listsorter(directory, sort_start, sort_end)]
        if not paths:
            raise ValueError(f"No DICOM files found in '{directory}'")
        datasets = []
        frames = []
        for path in paths:
            dataset, frame = cls.read_dicom_file(path)
            datasets.append(dataset)
            frames.append(frame)
        return datasets[-1], np.stack(frames).astype(np.float32, copy=False)

    @staticmethod
    def read_json_data(file_name: str | Path) -> dict[str, Any]:
        """Read JSON5 when available, falling back to the standard JSON parser."""
        try:
            import json5
        except ImportError:
            import json

            with Path(file_name).open("r", encoding="utf-8") as handle:
                return json.load(handle)
        with Path(file_name).open("r", encoding="utf-8") as handle:
            return json5.load(handle)

    @staticmethod
    def make_dirs(directory: str | Path) -> Path:
        """Create a directory tree and return the resulting path."""
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        return output

    # Names used by the original Sim_data_demo scripts.
    dicomreader = read_dicom_file


class ProjectionTool:
    """Projection-domain readers and numerical preprocessing helpers."""

    @staticmethod
    def read_csv_angles(file_name: str | Path) -> np.ndarray:
        import csv

        values = []
        with Path(file_name).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if row and row[0].strip():
                    values.append(float(row[0]))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def read_xml_angles(file_name: str | Path) -> np.ndarray:
        from xml.etree import ElementTree

        root = ElementTree.parse(file_name).getroot()
        values = []
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "GantryAngle" and element.text:
                values.append(float(element.text))
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def read_projection_bin_series(
        directory: str | Path,
        width: int,
        height: int,
        pattern: str = "Proj_*.bin",
        data_type: str | np.dtype = np.float32,
    ) -> np.ndarray:
        """Read ``Proj_*.bin`` files into ``(projection, width, height)``."""
        paths = sorted(
            Path(directory).glob(pattern),
            key=lambda path: _numeric_sort_key(path.name, 0, None),
        )
        if not paths:
            raise ValueError(f"No projection files found in '{directory}'")
        frames = []
        expected = width * height
        for path in paths:
            values = np.fromfile(path, dtype=data_type)
            if values.size != expected:
                raise ValueError(f"Projection '{path}' has {values.size} values; expected {expected}")
            frames.append(values.reshape(width, height))
        return np.stack(frames).astype(np.float32, copy=False)

    @staticmethod
    def read_linatech_projections(
        rootdir: str | Path,
        num: int,
        i0,
        *,
        image_shape: tuple[int, int] = (1408, 1408),
        output_shape: tuple[int, int] = (352, 704),
        file_step: int = 2,
        gaussian_sigma: float = 1.0,
        polynomial_coefficients: tuple[float, float, float, float, float] = (
            3.99567731237236,
            -25.4493077640396,
            55.8284068254555,
            11.9551036611716,
            0.475674868518792,
        ),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read and preprocess the LinaTech DICOM projection series."""
        if num < 1:
            raise ValueError("num must be positive")
        if len(image_shape) != 2 or len(output_shape) != 2:
            raise ValueError("image_shape and output_shape must be 2-D")
        row_step = image_shape[0] // output_shape[0]
        column_step = image_shape[1] // output_shape[1]
        if (
            row_step < 1
            or column_step < 1
            or image_shape[0] % output_shape[0]
            or image_shape[1] % output_shape[1]
        ):
            raise ValueError("image_shape must be an integer multiple of output_shape")

        import pydicom

        gaussian_filter = None
        if gaussian_sigma > 0:
            from scipy.ndimage import gaussian_filter as scipy_gaussian_filter

            gaussian_filter = scipy_gaussian_filter

        background = np.asarray(i0, dtype=np.float32)
        output = np.empty((*output_shape, num), dtype=np.float32)
        angles = np.empty(num, dtype=np.float32)
        a, b, c, d, e = polynomial_coefficients
        for index in range(num):
            file_name = Path(rootdir) / f"{(index + 1) * file_step:03d}.dcm"
            dataset = pydicom.dcmread(str(file_name), force=True)
            angle = dataset.get("GantryAngle", getattr(dataset, "GantryAngle", None))
            if angle is None:
                raise ValueError(f"DICOM file '{file_name}' has no GantryAngle")
            angles[index] = float(angle)
            image = np.asarray(dataset.pixel_array, dtype=np.float32)
            if image.shape != tuple(image_shape):
                raise ValueError(
                    f"DICOM file '{file_name}' has shape {image.shape}; expected {image_shape}"
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                image = -np.log(image / (background + 1e-7))
            image[~np.isfinite(image)] = 0
            image = a * image**4 + b * image**3 + c * image**2 + d * image + e
            if gaussian_filter is not None:
                image = gaussian_filter(image, sigma=gaussian_sigma)
            sampled = np.zeros(output_shape, dtype=np.float32)
            for row_offset in range(row_step):
                for column_offset in range(column_step):
                    sampled += image[row_offset::row_step, column_offset::column_step]
            sampled /= row_step * column_step
            if sampled.shape != tuple(output_shape):
                raise ValueError(f"Downsampled projection has shape {sampled.shape}; expected {output_shape}")
            output[:, :, index] = sampled
        return output, angles

    @staticmethod
    def weight_gen(width: int, transition: int) -> np.ndarray:
        if width < 1 or transition < 1:
            raise ValueError("width and transition must be positive")
        mu = (np.arange(transition * 2, dtype=np.float32) + 1) / (2 * transition)
        weight = 10 * mu**3 - 15 * mu**4 + 6 * mu**5
        return np.tile(weight[None, :], (width, 1))

    @staticmethod
    def pad_linatech(projection: np.ndarray, padding: int) -> np.ndarray:
        projection = np.asarray(projection)
        if projection.ndim != 3:
            raise ValueError("projection must be a 3-D array")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if padding == 0:
            return projection.copy()
        if padding > projection.shape[1]:
            raise ValueError("padding cannot exceed the detector-column count")
        left = projection[:, padding - 1 :: -1, :]
        right = projection[:, -padding:, :]
        return np.concatenate((left, projection, right), axis=1)

    @staticmethod
    def circle_mask(radius: float, height: int, width: int, dtype=np.float32) -> np.ndarray:
        if radius < 0 or height < 1 or width < 1:
            raise ValueError("radius must be non-negative and dimensions must be positive")
        center_x, center_y = (width - 1) / 2, (height - 1) / 2
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2)
        return mask.astype(dtype)

    @classmethod
    def mask_generate_linatech(
        cls,
        height: int,
        width: int,
        depth: int,
        radius: float,
        dtype=np.float32,
    ) -> np.ndarray:
        if depth < 1:
            raise ValueError("depth must be positive")
        return np.repeat(cls.circle_mask(radius, height, width, dtype)[..., None], depth, axis=2)

    @staticmethod
    def add_poisson_noise(projection: np.ndarray, i0: float = 5e4) -> np.ndarray:
        if i0 <= 0:
            raise ValueError("i0 must be positive")
        projection = np.asarray(projection)
        output_dtype = projection.dtype if np.issubdtype(projection.dtype, np.floating) else np.float32
        expected_counts = i0 * np.exp(-projection) * 20
        counts = np.maximum(np.random.poisson(np.clip(expected_counts, 0, None)), 1)
        return (-np.log(counts / (i0 * 20))).astype(output_dtype)

    @staticmethod
    def add_poisson_gaussian_noise(
        projection: np.ndarray,
        poisson: float = 1e7,
        gaussian: tuple[float, float] | None = (0, 10),
    ) -> np.ndarray:
        if poisson <= 0:
            raise ValueError("poisson must be positive")
        projection = np.asarray(projection)
        output_dtype = projection.dtype if np.issubdtype(projection.dtype, np.floating) else np.float32
        scale = float(np.max(np.abs(projection)))
        scale = scale if scale > 0 else 1.0
        counts = poisson * np.exp(-projection / scale)
        if gaussian is not None:
            counts = counts + np.random.normal(gaussian[0], gaussian[1], size=projection.shape)
        counts = np.maximum(counts, 1)
        counts = np.random.poisson(counts) + 1
        return (-np.log(counts / poisson) * scale).astype(output_dtype)

    @staticmethod
    def add_poisson_gaussian_noise_v2(
        projection: np.ndarray,
        poisson: float = 1e7,
        gaussian: tuple[float, float] | None = (0, 10),
    ) -> np.ndarray:
        if poisson <= 0:
            raise ValueError("poisson must be positive")
        projection = np.asarray(projection)
        output_dtype = projection.dtype if np.issubdtype(projection.dtype, np.floating) else np.float32
        transmission = np.exp(-projection)
        maximum = float(np.max(transmission))
        maximum = maximum if maximum > 0 else 1.0
        counts = transmission / maximum * poisson
        if gaussian is not None:
            counts = counts + np.random.normal(gaussian[0], gaussian[1], size=projection.shape)
        counts = np.random.poisson(np.maximum(np.abs(counts), 1)) + 1
        return (-np.log(counts / poisson * maximum)).astype(output_dtype)

    @staticmethod
    def fill_zeros(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError("array must be 2-D")
        output = array.copy()
        zero_indices = np.argwhere(array == 0)
        for row, column in zero_indices:
            row_start, row_end = max(0, row - 1), min(array.shape[0], row + 2)
            col_start, col_end = max(0, column - 1), min(array.shape[1], column + 2)
            neighbors = array[row_start:row_end, col_start:col_end].ravel()
            neighbors = neighbors[neighbors != 0]
            if neighbors.size:
                output[row, column] = np.mean(neighbors)
        return output

    @classmethod
    def remove_zeros(cls, array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 2:
            return cls.fill_zeros(array)
        if array.ndim == 3:
            return np.stack([cls.fill_zeros(frame) for frame in array])
        raise ValueError("array must be 2-D or 3-D")

    @staticmethod
    def calculate_cdf(histogram: np.ndarray) -> np.ndarray:
        histogram = np.asarray(histogram)
        total = histogram.sum()
        if total <= 0:
            return np.zeros_like(histogram, dtype=np.float64)
        return histogram.cumsum() / total

    @classmethod
    def histogram_match(cls, source: np.ndarray, template: np.ndarray, bins: int = 256) -> np.ndarray:
        source = np.asarray(source)
        template = np.asarray(template)
        if bins < 2:
            raise ValueError("bins must be at least 2")
        source_values = source.ravel()
        template_values = template.ravel()
        source_hist, source_edges = np.histogram(source_values, bins=bins)
        template_hist, template_edges = np.histogram(template_values, bins=bins)
        source_cdf = cls.calculate_cdf(source_hist)
        template_cdf = cls.calculate_cdf(template_hist)
        mapped = np.interp(source_cdf, template_cdf, template_edges[:-1])
        return np.interp(source_values, source_edges[:-1], mapped).reshape(source.shape)

    @staticmethod
    def denoise_tv(image: np.ndarray, weight: float) -> np.ndarray:
        from skimage.restoration import denoise_tv_chambolle

        image = np.asarray(image, dtype=np.float32)
        output = image.copy()
        frames = output[None, ...] if output.ndim == 2 else output
        for index, frame in enumerate(frames):
            low, high = float(frame.min()), float(frame.max())
            if high <= low:
                continue
            normalized = (frame - low) / (high - low)
            frames[index] = denoise_tv_chambolle(normalized, weight=weight) * (high - low) + low
        return output

    # Names used by the original Sim_data_demo scripts.
    readCSVBin = read_csv_angles
    readXMLAngle = read_xml_angles
    readProj = read_projection_bin_series
    readProj_linatech = read_linatech_projections
    weightGen = weight_gen
    padding_linatech = pad_linatech
    maskGenerate_linatech = mask_generate_linatech
    circleCompute = circle_mask
    addPossionNoisy = add_poisson_noise
    addPossionGaussionNoisy = add_poisson_gaussian_noise
    addPossionGaussionNoisyV2 = add_poisson_gaussian_noise_v2
    fill_zeros = fill_zeros
    remove_zeros = remove_zeros
    denoiseTV = denoise_tv


class LinaTechTigreTool:
    """TIGRE helpers for LinaTech geometries with per-view calibration."""

    @staticmethod
    def set_geometry(
        DSD,
        DSO,
        nDetector=(512, 512),
        dDetector=(0.8, 0.8),
        nVoxel=(256, 256, 256),
        dVoxel=(1.0, 1.0, 1.0),
        offOrigin=(0.0, 0.0, 0.0),
        offDetector=(0.0, 0.0),
        accuracy: float = 0.5,
        COR=0.0,
        rotDetector=(0.0, 0.0, 0.0),
        mode: str = "cone",
        angles: np.ndarray | None = None,
        high_resolution: bool = True,
    ):
        """Build a geometry while preserving vector-valued per-view fields."""
        import tigre

        geometry = tigre.geometry(mode=mode, high_resolution=high_resolution)
        geometry.angles = np.asarray(
            np.linspace(0, 2 * np.pi, 2400, dtype=np.float32)
            if angles is None
            else angles,
            dtype=np.float32,
        )
        geometry.DSD = LinaTechTigreTool._scalar_or_array(DSD, np.float32)
        geometry.DSO = LinaTechTigreTool._scalar_or_array(DSO, np.float32)
        geometry.nDetector = np.asarray(nDetector, dtype=np.int32)
        geometry.dDetector = np.asarray(dDetector, dtype=np.float32)
        geometry.sDetector = geometry.nDetector * geometry.dDetector
        geometry.nVoxel = np.asarray(nVoxel, dtype=np.int32)
        geometry.dVoxel = np.asarray(dVoxel, dtype=np.float32)
        geometry.sVoxel = geometry.nVoxel * geometry.dVoxel
        geometry.offOrigin = np.asarray(offOrigin, dtype=np.float32)
        geometry.offDetector = np.asarray(offDetector, dtype=np.float32)
        geometry.accuracy = float(accuracy)
        geometry.COR = COR
        geometry.rotDetector = np.asarray(rotDetector, dtype=np.float32)
        geometry.mode = mode
        return geometry

    @staticmethod
    def _scalar_or_array(value, dtype):
        array = np.asarray(value, dtype=dtype)
        return float(array) if array.ndim == 0 else array

    @staticmethod
    def project(volume: np.ndarray, geometry, angles: np.ndarray) -> np.ndarray:
        import tigre

        result = tigre.Ax(
            np.asarray(volume, dtype=np.float32),
            geometry,
            np.asarray(angles, dtype=np.float32),
        )
        return np.asarray(result, dtype=np.float32)

    @staticmethod
    def reconstruct(
        projections: np.ndarray,
        geometry,
        angles: np.ndarray,
        filter_name: str = "hamming",
        parker: bool = True,
    ) -> np.ndarray:
        import tigre.algorithms as algorithms

        result = algorithms.fdk(
            np.asarray(projections, dtype=np.float32),
            geometry,
            np.asarray(angles, dtype=np.float32),
            filter=filter_name,
            parker=parker,
        )
        return np.asarray(result, dtype=np.float32)

    build_geometry = set_geometry
    set_geo = set_geometry
    project_tigre = project
    reconstruct_tigre = reconstruct


def _numeric_sort_key(name: str, start: int = 0, end: int | None = None):
    import re

    if end is not None:
        fragment = name[start:end]
        try:
            return (0, int(fragment))
        except ValueError:
            pass
    numbers = re.findall(r"\d+", name)
    return (1, int(numbers[-1]) if numbers else 0, name)


__all__ = [
    "MedicalImage",
    "MedicalImageTool",
    "TigreTool",
    "SimulationDataTool",
    "ProjectionTool",
    "LinaTechTigreTool",
    "build_tigre_geometry",
    "hu_to_mu",
    "make_angles",
    "project_tigre",
    "reconstruct_tigre",
    "reconstruct_volume",
    "set_geo",
]

