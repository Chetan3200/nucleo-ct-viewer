"""Local data smoke test; skipped when the assignment files are not distributed."""
from pathlib import Path
from io import BytesIO

import httpx
from PIL import Image
import numpy as np
import pytest

from nucleo_viewer.imaging import load_dicom_zip, load_mask, render_slice, validate_mask_geometry
from nucleo_viewer.app import create_app, Settings

DATA = Path(__file__).resolve().parent.parent / "Take-home case-001"


@pytest.mark.skipif(not (DATA / "case-001-dicom.zip").is_file(), reason="Supplied assignment data is not in this checkout")
def test_supplied_ct_and_reference_share_physical_geometry():
    ct = load_dicom_zip((DATA / "case-001-dicom.zip").read_bytes())
    reference = load_mask((DATA / "case-001-reference-lesions.nii.gz").read_bytes())
    assert ct.shape == (416, 350, 180)
    assert reference.data.shape == ct.shape
    np.testing.assert_allclose(ct.affine, reference.affine, atol=1e-4, rtol=0)
    validate_mask_geometry(ct, reference)
    plain = render_slice(ct, ct.slice_count // 2, [])
    overlay = render_slice(ct, ct.slice_count // 2, [(reference, (0, 211, 204))])
    assert plain.startswith(b"\x89PNG\r\n\x1a\n")
    assert overlay.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
@pytest.mark.skipif(not (DATA / "case-001-dicom.zip").is_file(), reason="Supplied assignment data is not in this checkout")
async def test_actual_supplied_reference_renders_before_any_analysis():
    def reject_external_request(request):
        raise AssertionError("Reference display must not need the segmentation service")

    app = create_app(Settings(data_dir=DATA, token=""), transport=httpx.MockTransport(reject_external_request))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            uploaded = await client.post("/api/cases", content=(DATA / "case-001-dicom.zip").read_bytes(), headers={"Content-Type": "application/zip"})
            assert uploaded.status_code == 200
            case = uploaded.json()
            assert case["reference_available"] is True
            assert case["status"] == "ready"
            assert case["structures"] == []
            assert case["dice"] is None
            # Pick a slice containing labelled lesions rather than an empty slice.
            index = int(np.count_nonzero(app.state.reference.data, axis=(0, 1)).argmax())
            route = f"/api/cases/{case['id']}/slice/{index}"
            plain = await client.get(route)
            overlay = await client.get(route + "?reference=true")
            assert plain.status_code == overlay.status_code == 200
            plain_pixels = np.asarray(Image.open(BytesIO(plain.content)))
            overlay_pixels = np.asarray(Image.open(BytesIO(overlay.content)))
            changed = np.any(plain_pixels != overlay_pixels, axis=2)
            assert np.count_nonzero(changed) > 0
            # Cyan overlay raises green/blue relative to red on a grayscale CT.
            assert np.all(overlay_pixels[changed, 1] > overlay_pixels[changed, 0])
