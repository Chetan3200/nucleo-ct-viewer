"""Binary segmentation metrics; geometry is checked, never silently resampled."""
import numpy as np

from .errors import AppError
from .imaging import Mask, validate_affine


def dice_score(model: Mask, reference: Mask) -> float:
    """Return 2|A∩B|/(|A|+|B|); two empty masks deliberately score 1.0.

The masks must share their voxel grid: same 3D shape and RAS+ affine within
1e-4 mm absolute tolerance (zero relative tolerance). Different grids fail
rather than hiding an alignment problem by implicitly resampling.
"""
    if model.data.ndim != 3 or reference.data.ndim != 3 or not all(model.data.shape):
        raise AppError("Dice needs two complete 3D overlay images.")
    if model.data.shape != reference.data.shape:
        raise AppError("The model and reference overlays have different image sizes, so they cannot be compared directly.")
    model_affine = validate_affine(model.affine)
    reference_affine = validate_affine(reference.affine)
    if not np.allclose(model_affine, reference_affine, atol=1e-4, rtol=0):
        raise AppError("The model and reference overlays do not line up in space, so they cannot be compared directly.")
    for data in (model.data, reference.data):
        if not np.all((data == 0) | (data == 1)):
            raise AppError("Dice cannot be calculated because an overlay contains values other than 0 and 1.")
    total = int(np.count_nonzero(model.data)) + int(np.count_nonzero(reference.data))
    if total == 0:
        return 1.0
    intersection = int(np.count_nonzero(np.logical_and(model.data, reference.data)))
    return float(2 * intersection / total)
