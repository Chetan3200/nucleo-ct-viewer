"""Local data smoke tests; skipped when the configured assignment files are absent."""
from io import BytesIO
import os
from pathlib import Path

from dotenv import get_key
import httpx
from PIL import Image
import numpy as np
import pytest

from nucleo_viewer.app import ROOT, Settings, create_app
from nucleo_viewer.imaging import load_dicom_zip, load_mask, render_slice, validate_mask_geometry

ZIP_NAME = "case-001-dicom.zip"
REFERENCE_NAME = "case-001-reference-lesions.nii.gz"
DEFAULT_DATA = ROOT / "Take-home case-001"
DOTENV_PATH = ROOT / ".env"


def configured_data_dir(environ=None, dotenv_path=DOTENV_PATH):
    """Match startup precedence while only reading the data-folder key from .env."""
    environ = os.environ if environ is None else environ
    configured = environ.get("NUCLEO_DATA_DIR")
    if configured is not None:
        return Path(configured)
    if dotenv_path.is_file():
        configured = get_key(str(dotenv_path), "NUCLEO_DATA_DIR")
        if configured is not None:
            return Path(configured)
    return DEFAULT_DATA


def supplied_files_available(data_dir):
    return (data_dir / ZIP_NAME).is_file() and (data_dir / REFERENCE_NAME).is_file()


DATA = configured_data_dir()
SUPPLIED_FILES_AVAILABLE = supplied_files_available(DATA)
SKIP_SUPPLIED_DATA = pytest.mark.skipif(
    not SUPPLIED_FILES_AVAILABLE,
    reason="Supplied ZIP and reference files are not both present in the configured data folder",
)


def test_settings_data_dir_defaults_to_repo_folder(monkeypatch, tmp_path):
    monkeypatch.delenv("NUCLEO_DATA_DIR", raising=False)
    assert Settings().data_dir == DEFAULT_DATA
    assert configured_data_dir({}, tmp_path / ".env") == DEFAULT_DATA


def test_configured_data_dir_reads_dotenv_without_loading_other_values(tmp_path, monkeypatch):
    monkeypatch.setenv("NUCLEO_API_TOKEN", "existing-test-token")
    configured = tmp_path / "configured-case"
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(f"NUCLEO_DATA_DIR={configured}\nNUCLEO_API_TOKEN=dummy-token\n")

    assert configured_data_dir({}, dotenv_path) == configured
    assert os.environ["NUCLEO_API_TOKEN"] == "existing-test-token"


def test_environment_data_dir_takes_precedence_over_dotenv(tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("NUCLEO_DATA_DIR=/from-dotenv\n")

    assert configured_data_dir({"NUCLEO_DATA_DIR": "/from-environment"}, dotenv_path) == Path("/from-environment")


@pytest.mark.parametrize("present", [ZIP_NAME, REFERENCE_NAME])
def test_supplied_files_require_both_zip_and_reference(tmp_path, present):
    (tmp_path / present).touch()
    assert not supplied_files_available(tmp_path)


@SKIP_SUPPLIED_DATA
def test_supplied_ct_and_reference_share_physical_geometry():
    ct = load_dicom_zip((DATA / ZIP_NAME).read_bytes())
    reference = load_mask((DATA / REFERENCE_NAME).read_bytes())
    assert ct.shape == (416, 350, 180)
    assert reference.data.shape == ct.shape
    np.testing.assert_allclose(ct.affine, reference.affine, atol=1e-4, rtol=0)
    validate_mask_geometry(ct, reference)
    plain = render_slice(ct, ct.slice_count // 2, [])
    overlay = render_slice(ct, ct.slice_count // 2, [(reference, (0, 211, 204))])
    assert plain.startswith(b"\x89PNG\r\n\x1a\n")
    assert overlay.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.asyncio
@SKIP_SUPPLIED_DATA
async def test_actual_supplied_reference_renders_before_any_analysis():
    def reject_external_request(request):
        raise AssertionError("Reference display must not need the segmentation service")

    app = create_app(Settings(data_dir=DATA, token=""), transport=httpx.MockTransport(reject_external_request))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            uploaded = await client.post("/api/cases", content=(DATA / ZIP_NAME).read_bytes(), headers={"Content-Type": "application/zip"})
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
