"""Small, synchronous image helpers for the take-home viewer.

Supported CT: one series of uncompressed little-endian, single-frame CT images,
with axis-aligned axial geometry. Oblique scans, gantry tilt and drifting stacks
are rejected, not silently flattened. PNGs use native pixels (no pixel-aspect
resampling), anterior at the top and the patient's right at the viewer's left.
NIfTI masks are single-file gzip NIfTI-1/2. Unknown spatial units assume mm.
"""
from dataclasses import dataclass
from io import BytesIO
from itertools import product
import gzip
import math
from pathlib import PurePosixPath
import zipfile
import zlib

import nibabel as nib
import numpy as np
from PIL import Image
import pydicom
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian

from .errors import AppError

MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ZIP_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_NIFTI_BYTES = 128 * 1024 * 1024  # expanded file must be strictly smaller
MAX_VOXELS = 64_000_000
GEOMETRY_TOLERANCE = 1e-4


@dataclass
class CTVolume:
    pixels: np.ndarray  # float32, [slice, row, column], in Hounsfield units
    affine: np.ndarray  # [column, row, slice, 1] -> RAS+ millimetres

    @property
    def slice_count(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        slices, rows, columns = self.pixels.shape
        return int(columns), int(rows), int(slices)


@dataclass
class Mask:
    data: np.ndarray  # uint8 binary values, [i, j, k]
    affine: np.ndarray  # [i, j, k, 1] -> RAS+ millimetres


def validate_affine(affine: np.ndarray) -> np.ndarray:
    """Validate a finite, invertible homogeneous voxel-to-world transform."""
    try:
        affine = np.asarray(affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError
        if not np.allclose(affine[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
            raise ValueError
        inverse = np.linalg.inv(affine)
        if not np.isfinite(inverse).all():
            raise ValueError
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise AppError("The image has missing or invalid positioning information.") from exc
    return affine


def _numbers(dataset, name: str, count: int) -> np.ndarray:
    try:
        values = np.asarray(getattr(dataset, name), dtype=np.float64).reshape(-1)
        if values.size != count or not np.isfinite(values).all():
            raise ValueError
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise AppError("A CT file has missing or invalid image details. Try the original ZIP.") from exc
    return values


def _integer(dataset, name: str) -> int:
    value = _numbers(dataset, name, 1)[0]
    if value != int(value):
        raise AppError("A CT file contains an invalid image setting. Try the original ZIP.")
    return int(value)


def _text(dataset, name: str) -> str:
    value = str(getattr(dataset, name, "")).strip()
    if not value:
        raise AppError("A CT file is missing required information. Try the original ZIP.")
    return value


def _axial_directions(column: np.ndarray, row: np.ndarray) -> np.ndarray:
    """Require cardinal in-plane axes, allowing 90-degree rotations/flips."""
    if not (np.isclose(np.linalg.norm(column), 1, atol=1e-4, rtol=0)
            and np.isclose(np.linalg.norm(row), 1, atol=1e-4, rtol=0)
            and np.isclose(column @ row, 0, atol=1e-4, rtol=0)):
        raise AppError("The scan's slice directions are invalid, so the images cannot be aligned.")
    column_axis, row_axis = int(np.argmax(abs(column))), int(np.argmax(abs(row)))
    if ({column_axis, row_axis} != {0, 1}
            or not np.isclose(abs(column[column_axis]), 1, atol=1e-6, rtol=0)
            or not np.isclose(abs(row[row_axis]), 1, atol=1e-6, rtol=0)
            or abs(column[2]) > 1e-4 or abs(row[2]) > 1e-4):
        raise AppError("This viewer only supports straight, horizontal CT slices. Use the supplied CT ZIP.")
    return np.cross(column, row)


def _zip_members(archive: zipfile.ZipFile):
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_ENTRIES:
        raise AppError("This ZIP contains too many files or folders. Choose a ZIP with one complete CT scan.")
    total = 0
    members = []
    for info in infos:
        name = info.filename.replace("\\", "/")
        parts = PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or any(":" in part for part in parts):
            raise AppError("This ZIP has an unsafe folder structure. Use the original CT ZIP.")
        if info.file_size < 0 or info.file_size > MAX_ZIP_MEMBER_BYTES:
            raise AppError("A file inside the ZIP is too large to open safely.")
        total += info.file_size
        if total > MAX_ZIP_EXPANDED_BYTES:
            raise AppError("This ZIP becomes too large when unpacked. Use the supplied CT ZIP.")
        if (info.is_dir() or not parts or "__MACOSX" in parts
                or any(part.startswith(".") for part in parts)
                or parts[-1].upper() == "DICOMDIR"):
            continue
        if info.flag_bits & 1:
            raise AppError("Password-protected ZIP files are not supported. Use an unprotected ZIP.")
        members.append(info)
    if not members:
        raise AppError("No CT images were found in this ZIP.")
    return members


def _read_dicom(raw: bytes):
    try:
        dataset = pydicom.dcmread(BytesIO(raw), force=False)
    except Exception as exc:
        raise AppError("One of the files in the ZIP could not be read as a CT image.") from exc
    transfer = str(getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", ""))
    if transfer not in {str(ExplicitVRLittleEndian), str(ImplicitVRLittleEndian)}:
        raise AppError("This CT file uses a storage format the viewer cannot read. Use the supplied CT ZIP.")
    if _text(dataset, "Modality") != "CT" or _text(dataset, "SOPClassUID") != str(CTImageStorage):
        raise AppError("The ZIP must contain CT images only.")
    frames = _integer(dataset, "NumberOfFrames") if "NumberOfFrames" in dataset else 1
    if frames != 1 or _integer(dataset, "SamplesPerPixel") != 1:
        raise AppError("Each CT file must contain one black-and-white slice. Use the supplied CT ZIP.")
    if _text(dataset, "PhotometricInterpretation") not in {"MONOCHROME1", "MONOCHROME2"}:
        raise AppError("This viewer only supports black-and-white CT images.")
    rows, columns = _integer(dataset, "Rows"), _integer(dataset, "Columns")
    if rows <= 0 or columns <= 0 or rows * columns > MAX_VOXELS:
        raise AppError("A CT image has an invalid size or is too large for this viewer.")
    allocated, stored = _integer(dataset, "BitsAllocated"), _integer(dataset, "BitsStored")
    if (allocated not in {8, 16} or not 1 <= stored <= allocated
            or _integer(dataset, "HighBit") != stored - 1
            or _integer(dataset, "PixelRepresentation") not in {0, 1}):
        raise AppError("This CT file stores its image in an unsupported format. Use the supplied CT ZIP.")
    expected = rows * columns * (allocated // 8)
    pixels = getattr(dataset, "PixelData", None)
    if not isinstance(pixels, bytes) or len(pixels) not in {expected, expected + expected % 2}:
        raise AppError("A CT image is incomplete or damaged. Try the original ZIP.")
    orientation = _numbers(dataset, "ImageOrientationPatient", 6)
    normal = _axial_directions(orientation[:3], orientation[3:])
    position = _numbers(dataset, "ImagePositionPatient", 3)
    spacing = _numbers(dataset, "PixelSpacing", 2)
    if np.any(spacing <= 0):
        raise AppError("The CT file contains invalid image measurements. Try the original ZIP.")
    slope, intercept = _numbers(dataset, "RescaleSlope", 1)[0], _numbers(dataset, "RescaleIntercept", 1)[0]
    if slope == 0:
        raise AppError("The CT file contains invalid brightness information. Try the original ZIP.")
    for name in ("SliceThickness", "SpacingBetweenSlices"):
        if name in dataset and _numbers(dataset, name, 1)[0] <= 0:
            raise AppError("The scan contains invalid slice thickness or spacing information. Try the original ZIP.")
    return dict(dataset=dataset, rows=rows, columns=columns, orientation=orientation,
                normal=normal, position=position, spacing=spacing, slope=slope,
                intercept=intercept, series=_text(dataset, "SeriesInstanceUID"),
                instance=(_integer(dataset, "InstanceNumber") if "InstanceNumber" in dataset
                          and dataset.InstanceNumber not in (None, "") else None),
                uid=_text(dataset, "SOPInstanceUID"))


def load_dicom_zip(raw: bytes) -> CTVolume:
    """Validate the entire ZIP before decompressing; sort slices geometrically."""
    images = []
    voxels = 0
    try:
        with zipfile.ZipFile(BytesIO(raw)) as archive:
            members = _zip_members(archive)
            for info in members:
                with archive.open(info) as stream:
                    content = stream.read(MAX_ZIP_MEMBER_BYTES + 1)
                if len(content) != info.file_size or len(content) > MAX_ZIP_MEMBER_BYTES:
                    raise AppError("A file in the ZIP is damaged or incomplete. Try the original ZIP.")
                image = _read_dicom(content)
                voxels += image["rows"] * image["columns"]
                if voxels > MAX_VOXELS:
                    raise AppError("The scan is too large for this viewer. Use the supplied CT ZIP.")
                images.append(image)
    except AppError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError, NotImplementedError, ValueError, zlib.error) as exc:
        raise AppError("This ZIP could not be opened. It may be damaged or in an unsupported format.") from exc

    first = images[0]
    instances, uids = set(), set()
    for image in images:
        if image["series"] != first["series"]:
            raise AppError("The ZIP contains more than one scan. Choose a ZIP containing one complete CT scan.")
        if ((image["instance"] is not None and image["instance"] in instances)
                or image["uid"] in uids):
            raise AppError("The ZIP contains duplicate CT slices. Use a copy with each slice included once.")
        if image["instance"] is not None:
            instances.add(image["instance"])
        uids.add(image["uid"])
        if ((image["rows"], image["columns"]) != (first["rows"], first["columns"])
                or not np.allclose(image["orientation"], first["orientation"], atol=1e-4, rtol=0)
                or not np.allclose(image["spacing"], first["spacing"], atol=1e-4, rtol=0)):
            raise AppError("The CT slices have different sizes or positioning information and cannot be combined into one scan.")
    normal = first["normal"]
    images.sort(key=lambda image: float(image["position"] @ normal))
    positions = np.array([image["position"] for image in images])
    if len(images) > 1:
        steps = np.diff(positions @ normal)
        if np.any(steps <= 1e-4):
            raise AppError("Two or more CT slices have the same position. Use the original complete scan.")
        slice_spacing = float(np.median(steps))
        if not np.allclose(steps, slice_spacing, atol=1e-3, rtol=1e-3):
            raise AppError("The gaps between CT slices are uneven. The scan may be missing slices.")
        expected_positions = positions[0] + np.arange(len(images))[:, None] * slice_spacing * normal
        if not np.allclose(positions, expected_positions, atol=1e-3, rtol=0):
            raise AppError("The CT slices do not line up correctly. Tilted or shifted slices are not supported.")
    else:
        dataset = first["dataset"]
        tag = "SpacingBetweenSlices" if "SpacingBetweenSlices" in dataset else "SliceThickness"
        slice_spacing = float(_numbers(dataset, tag, 1)[0])
        if slice_spacing <= 0:
            raise AppError("This single-slice scan is missing valid slice thickness or spacing information.")

    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = first["orientation"][:3] * first["spacing"][1]
    affine[:3, 1] = first["orientation"][3:] * first["spacing"][0]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = positions[0]
    affine[:2, :] *= -1  # DICOM LPS physical coordinates -> NIfTI RAS+
    validate_affine(affine)
    volume = np.empty((len(images), first["rows"], first["columns"]), dtype=np.float32)
    for index, image in enumerate(images):
        try:
            # pydicom handles BitsStored sign extension, including signed 12-bit data.
            decoded = image["dataset"].pixel_array
            with np.errstate(over="ignore", invalid="ignore"):
                hu = decoded.astype(np.float64) * image["slope"] + image["intercept"]
            if not np.isfinite(hu).all() or np.any(abs(hu) > np.finfo(np.float32).max):
                raise AppError("The CT file's brightness information produces invalid image values. Try the original ZIP.")
            volume[index] = hu
        except AppError:
            raise
        except Exception as exc:
            raise AppError("A CT image could not be read. It may be damaged; try the original ZIP.") from exc
    return CTVolume(volume, affine)


def load_mask(raw: bytes) -> Mask:
    """Read bounded gzip data, validate its header before allocating the mask.

The payload is viewed in-place, then checked/scaled in small blocks. This avoids
letting an untrusted NIfTI shape or scaling allocate an unbounded float volume.
"""
    if not raw.startswith(b"\x1f\x8b"):
        raise AppError("The overlay file is not in the required .nii.gz format.")
    try:
        with gzip.GzipFile(fileobj=BytesIO(raw)) as stream:
            expanded = stream.read(MAX_NIFTI_BYTES)
    except (OSError, EOFError, ValueError, zlib.error) as exc:
        raise AppError("The overlay file is damaged or incomplete and could not be opened.") from exc
    if len(expanded) >= MAX_NIFTI_BYTES:
        raise AppError("The overlay becomes too large when unpacked. It must be smaller than 128 MiB.")
    try:
        # Header sizes identify both endian variants without loading voxel data.
        sizes = {int.from_bytes(expanded[:4], byteorder) for byteorder in ("little", "big")}
        if 348 in sizes:
            header = nib.Nifti1Header.from_fileobj(BytesIO(expanded), check=False)
            minimum_offset, magic = 352, b"n+1"
        elif 540 in sizes:
            header = nib.Nifti2Header.from_fileobj(BytesIO(expanded), check=False)
            minimum_offset, magic = 544, b"n+2"
        else:
            raise AppError("The overlay file uses a format this viewer cannot read.")
        if not bytes(header["magic"]).startswith(magic):
            raise AppError("The overlay must be stored in one file, not separate image and information files.")
        dimensions = header["dim"]
        shape = tuple(int(value) for value in dimensions[1:4])
        if int(dimensions[0]) != 3 or any(size <= 0 for size in shape):
            raise AppError("The overlay must contain a complete 3D image; this file is empty or has the wrong dimensions.")
        count = math.prod(shape)
        if count > MAX_VOXELS:
            raise AppError("The overlay image is too large for this viewer.")
        dtype = header.get_data_dtype()
        if dtype.kind not in "iuf" or int(header["bitpix"]) != dtype.itemsize * 8:
            raise AppError("The overlay stores its image in an unsupported data format.")
        offset_value = float(header["vox_offset"])
        if not math.isfinite(offset_value) or offset_value != int(offset_value):
            raise AppError("The overlay file's image information is damaged.")
        offset = int(offset_value)
        if offset < minimum_offset or offset + count * dtype.itemsize > len(expanded):
            raise AppError("The overlay image is incomplete or its file information is damaged.")
        qform_code, sform_code = int(header["qform_code"]), int(header["sform_code"])
        if (qform_code not in range(6) or sform_code not in range(6)
                or (qform_code == 0 and sform_code == 0)):
            raise AppError("The overlay is missing valid positioning information, so it cannot be placed on the CT scan.")
        affine = np.asarray(header.get_best_affine(), dtype=np.float64)
        unit = header.get_xyzt_units()[0]
        if unit == "meter":
            affine[:3, :] *= 1000
        elif unit == "micron":
            affine[:3, :] *= 0.001
        elif unit not in ("mm", "unknown"):
            raise AppError("The viewer cannot understand the distance measurements in this overlay.")
        validate_affine(affine)
        slope, intercept = float(header["scl_slope"]), float(header["scl_inter"])
        if slope == 0 or math.isnan(slope):
            slope, intercept = 1.0, 0.0  # NIfTI's undefined/no-scaling convention
        if not math.isfinite(slope) or not math.isfinite(intercept):
            raise AppError("The overlay file contains invalid image-scaling information.")
        values = np.frombuffer(expanded, dtype=dtype, count=count, offset=offset)
        output = np.empty(count, dtype=np.uint8)
        for start in range(0, count, 65_536):
            with np.errstate(over="ignore", invalid="ignore"):
                block = values[start:start + 65_536].astype(np.float64) * slope + intercept
            if not np.isfinite(block).all() or not np.all((block == 0) | (block == 1)):
                raise AppError("The overlay contains invalid values. Only 0 (background) and 1 (marked area) are supported.")
            output[start:start + len(block)] = block.astype(np.uint8)
        return Mask(output.reshape(shape, order="F"), affine)
    except AppError:
        raise
    except (ValueError, TypeError, OverflowError, OSError, KeyError, nib.spatialimages.HeaderDataError) as exc:
        raise AppError("The overlay file contains damaged or invalid image information.") from exc


def validate_mask_geometry(ct: CTVolume, mask: Mask) -> None:
    """Reject grids whose physical bounding boxes do not overlap.

    This catches gross positioning errors, not registration errors. Empty masks
    and individual slices without lesions remain valid.
    """
    if (ct.pixels.ndim != 3 or mask.data.ndim != 3
            or not all(ct.pixels.shape) or not all(mask.data.shape)):
        raise AppError("The scan and overlay must both contain complete 3D images.")

    def bounds(shape, affine):
        affine = validate_affine(affine)
        # Voxel centres start at zero; volume boundaries extend half a voxel.
        corners = np.array(list(product(*[(-0.5, size - 0.5) for size in shape])))
        world = corners @ affine[:3, :3].T + affine[:3, 3]
        return world.min(axis=0), world.max(axis=0)

    ct_min, ct_max = bounds(ct.shape, ct.affine)
    mask_min, mask_max = bounds(mask.data.shape, mask.affine)
    if np.any(ct_max < mask_min - 1e-6) or np.any(mask_max < ct_min - 1e-6):
        raise AppError("The overlay is positioned entirely outside the CT scan. Check that it belongs to this scan.")


def _radiological_image(rgb: np.ndarray, affine: np.ndarray) -> np.ndarray:
    column = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
    row = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
    _axial_directions(column, row)
    if int(np.argmax(abs(column))) == 1:
        rgb = rgb.transpose(1, 0, 2)
        column, row = row, column
    if column[0] > 0:
        rgb = rgb[:, ::-1]
    if row[1] > 0:
        rgb = rgb[::-1]
    return rgb


def render_slice(ct: CTVolume, index: int,
                 overlays: list[tuple[Mask, tuple[int, int, int]]],
                 center: float = 40, width: float = 400) -> bytes:
    """Window HU, sample binary masks in world space, then orient the RGB image.

Overlays use nearest-neighbour sampling with 45% opacity, in list order. Pixels
outside a mask's field of view are transparent. No segmentation interpolation.
"""
    if ct.pixels.ndim != 3 or not all(ct.pixels.shape):
        raise AppError("The CT scan is empty or does not contain a complete 3D image.")
    if not isinstance(index, (int, np.integer)) or isinstance(index, bool) or not 0 <= index < ct.slice_count:
        raise AppError("That slice is not part of this scan. Choose a slice within the displayed range.")
    if not math.isfinite(center) or not math.isfinite(width) or width <= 0:
        raise AppError("The brightness or contrast settings are invalid.")
    affine = validate_affine(ct.affine)
    plane = ct.pixels[index]
    if not np.isfinite(plane).all():
        raise AppError("This CT slice contains invalid image values and cannot be displayed.")
    gray = np.clip((plane.astype(np.float64) - (center - width / 2)) / width, 0, 1)
    rgb = np.repeat(np.rint(gray * 255).astype(np.uint8)[..., None], 3, axis=2)
    rows, columns = plane.shape
    column_grid = np.arange(columns, dtype=np.float64)[None, :]
    for mask, color in overlays:
        validate_mask_geometry(ct, mask)
        if (len(color) != 3 or any(not isinstance(value, (int, np.integer))
                                  or not 0 <= value <= 255 for value in color)):
            raise AppError("An overlay colour is invalid. Reload the scan and try again.")
        mapping = np.linalg.inv(validate_affine(mask.affine)) @ affine
        # Row blocks bound temporary coordinate arrays even for large CT planes.
        for start in range(0, rows, 128):
            stop = min(start + 128, rows)
            row_grid = np.arange(start, stop, dtype=np.float64)[:, None]
            valid = np.ones((stop - start, columns), dtype=bool)
            indices = []
            for axis in range(3):
                coordinate = np.rint(mapping[axis, 0] * column_grid + mapping[axis, 1] * row_grid
                                     + mapping[axis, 2] * index + mapping[axis, 3])
                valid &= np.isfinite(coordinate) & (coordinate >= 0) & (coordinate < mask.data.shape[axis])
                coordinate = np.nan_to_num(coordinate, nan=0, posinf=0, neginf=0)
                indices.append(np.clip(coordinate, 0, mask.data.shape[axis] - 1).astype(np.int64))
            selected = valid & (mask.data[tuple(indices)] != 0)
            block = rgb[start:stop]
            block[selected] = np.rint(block[selected] * 0.55 + np.asarray(color) * 0.45).astype(np.uint8)
    rgb = _radiological_image(rgb, affine)
    stream = BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(stream, format="PNG")
    return stream.getvalue()
