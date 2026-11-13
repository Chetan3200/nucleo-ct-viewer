"""The local reference must not depend on successful model inference."""
import asyncio

import httpx
import numpy as np
import pytest

import nucleo_viewer.app as app_module
from nucleo_viewer.errors import AppError
from nucleo_viewer.imaging import CTVolume, Mask

KNOWN_ZIP = b"known-supplied-zip"


@pytest.fixture
def local_case(monkeypatch, tmp_path):
    (tmp_path / "case-001-dicom.zip").write_bytes(KNOWN_ZIP)
    (tmp_path / "case-001-reference-lesions.nii.gz").write_bytes(b"reference")
    ct = CTVolume(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4))
    mask = Mask(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4))
    monkeypatch.setattr(app_module, "load_dicom_zip", lambda raw: ct)
    monkeypatch.setattr(app_module, "load_mask", lambda raw: mask)
    monkeypatch.setattr(app_module, "render_slice", lambda ct, index, overlays: b"with-reference" if overlays else b"plain")
    settings = app_module.Settings(base_url="https://segmentation.test", token="", data_dir=tmp_path)
    return settings, mask


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/api/sample", "/api/cases"])
async def test_reference_available_before_analysis_without_token(local_case, route):
    settings, mask = local_case
    calls = []

    def no_upstream_calls(request):
        calls.append(request)
        raise AssertionError("Local reference viewing must not call the segmentation service")

    app = app_module.create_app(settings, transport=httpx.MockTransport(no_upstream_calls))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            loaded = await client.post(route, content=KNOWN_ZIP, headers={"Content-Type": "application/zip"})
            assert loaded.status_code == 200
            data = loaded.json()
            assert data["reference_available"] is True
            assert data["status"] == "ready"
            assert data["structures"] == []
            assert data["dice"] is None
            assert app.state.cases[data["id"]].reference is mask
            overlay = await client.get(f"/api/cases/{data['id']}/slice/0?reference=true")
            plain = await client.get(f"/api/cases/{data['id']}/slice/0")
            assert overlay.status_code == plain.status_code == 200
            assert overlay.content == b"with-reference"
            assert plain.content == b"plain"
    assert calls == []


@pytest.mark.asyncio
async def test_same_geometry_different_scan_does_not_inherit_reference(local_case):
    settings, _ = local_case
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            loaded = await client.post("/api/cases", content=b"a-different-scan", headers={"Content-Type": "application/zip"})
            assert loaded.status_code == 200
            data = loaded.json()
            assert data["reference_available"] is False
            overlay = await client.get(f"/api/cases/{data['id']}/slice/0?reference=true")
            assert overlay.status_code == 400


@pytest.mark.asyncio
async def test_reference_survives_missing_token(local_case):
    settings, _ = local_case
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            loaded = (await client.post("/api/sample")).json()
            case_id = loaded["id"]
            failed = await client.post(f"/api/cases/{case_id}/segment")
            assert failed.status_code == 400
            assert failed.json()["error"]["code"] == "missing_token"
            status = (await client.get(f"/api/cases/{case_id}")).json()
            assert status["reference_available"] is True
            assert status["dice"] is None
            overlay = await client.get(f"/api/cases/{case_id}/slice/0?reference=true")
            assert overlay.status_code == 200
            assert overlay.content == b"with-reference"


@pytest.mark.asyncio
async def test_reference_stays_available_during_analysis_and_after_failure(local_case, monkeypatch):
    settings, _ = local_case
    started, finish = asyncio.Event(), asyncio.Event()

    class FailingClient:
        def __init__(self, *args):
            pass

        async def segment(self, raw, progress):
            progress("Waiting for segmentation")
            started.set()
            await finish.wait()
            raise AppError("The service is unavailable.", 502, "upstream_error")

    monkeypatch.setattr(app_module, "SegmentationClient", FailingClient)
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            case_id = (await client.post("/api/sample")).json()["id"]
            pending = asyncio.create_task(client.post(f"/api/cases/{case_id}/segment"))
            await asyncio.wait_for(started.wait(), timeout=3)
            during = (await client.get(f"/api/cases/{case_id}")).json()
            overlay = await client.get(f"/api/cases/{case_id}/slice/0?reference=true")
            assert during["status"] == "processing"
            assert during["reference_available"] is True
            assert during["dice"] is None
            assert overlay.status_code == 200
            finish.set()
            failed = await asyncio.wait_for(pending, timeout=3)
            assert failed.status_code == 502
            after = (await client.get(f"/api/cases/{case_id}")).json()
            assert after["status"] == "error"
            assert after["reference_available"] is True
            assert after["structures"] == []
            assert after["dice"] is None
            assert (await client.get(f"/api/cases/{case_id}/slice/0?reference=true")).content == b"with-reference"
