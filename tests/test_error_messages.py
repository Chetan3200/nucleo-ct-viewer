"""Readable errors retain the existing status codes and validation behaviour."""
import numpy as np
import pytest
from pydicom.dataset import Dataset

from nucleo_viewer.api_client import error_message
from nucleo_viewer.errors import AppError
from nucleo_viewer.imaging import Mask, _numbers, _text
from nucleo_viewer.metrics import dice_score


@pytest.mark.parametrize("field,reader,count", [
    ("ImagePositionPatient", _numbers, 3),
    ("RescaleSlope", _numbers, 1),
    ("SOPInstanceUID", _text, None),
])
def test_missing_ct_details_do_not_expose_internal_tag_names(field, reader, count):
    with pytest.raises(AppError) as raised:
        if count is None:
            reader(Dataset(), field)
        else:
            reader(Dataset(), field, count)
    error = raised.value
    assert error.status_code == 400
    assert error.code == "invalid_input"
    assert field not in error.message
    assert "CT file" in error.message
    assert "Try the original ZIP" in error.message


def test_dice_position_mismatch_explains_why_comparison_is_unavailable():
    data = np.ones((2, 2, 2), dtype=np.uint8)
    shifted = np.eye(4)
    shifted[0, 3] = 2
    with pytest.raises(AppError) as raised:
        dice_score(Mask(data, np.eye(4)), Mask(data, shifted))
    assert raised.value.message == "The model and reference overlays do not line up in space, so they cannot be compared directly."


def test_upload_format_error_does_not_expose_http_implementation_details():
    message = error_message(415, {"error": {"code": "unsupported_media_type", "message": "private technical details"}})
    assert "original CT ZIP" in message
    assert "multipart" not in message
    assert "private technical details" not in message
