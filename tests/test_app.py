import asyncio
from pathlib import Path

import httpx
import numpy as np
import pytest

import nucleo_viewer.app as app_module
from nucleo_viewer.errors import AppError
from nucleo_viewer.imaging import CTVolume, Mask


def tiny_ct(marker: float = 0.0) -> CTVolume:
    pixels = np.full((2, 2, 2), marker, dtype=np.float32)
    return CTVolume(pixels=pixels, affine=np.eye(4, dtype=np.float64))


def tiny_mask(which: bytes) -> Mask:
    data = np.zeros((2, 2, 2), dtype=np.uint8)
    if which == b"A":
        data[0, 0, 0] = 1
    else:
        data[1, 1, 1] = 1
    return Mask(data=data, affine=np.eye(4, dtype=np.float64))


def make_settings(tmp_path: Path, token: str = "") -> app_module.Settings:
    return app_module.Settings(base_url="https://segmentation.test", token=token, data_dir=tmp_path)


@pytest.mark.asyncio
async def test_failed_local_zip_upload_has_clear_error_and_valid_upload_recovers(monkeypatch, tmp_path):
    def fake_load(raw: bytes) -> CTVolume:
        if raw == b"bad-zip":
            raise AppError("The local ZIP could not be read.", 400, "invalid_zip")
        return tiny_ct(7)

    monkeypatch.setattr(app_module, "load_dicom_zip", fake_load)
    application = app_module.create_app(make_settings(tmp_path))
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
            bad = await client.post("/api/cases", content=b"bad-zip", headers={"Content-Type": "application/zip"})
            good = await client.post("/api/cases", content=b"valid-zip", headers={"Content-Type": "application/zip"})
            status = await client.get(f"/api/cases/{good.json()['id']}")

    assert bad.status_code == 400
    assert bad.json()["error"] == {"code": "invalid_zip", "message": "The local ZIP could not be read."}
    assert good.status_code == 200
    assert good.json()["status"] == "ready"
    assert status.status_code == 200
    assert status.json()["id"] == good.json()["id"]


@pytest.mark.asyncio
async def test_case_routes_validate_upload_type_and_slice_index(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_dicom_zip", lambda raw: tiny_ct())
    monkeypatch.setattr(app_module, "render_slice", lambda ct, index, overlays: b"png")
    application = app_module.create_app(make_settings(tmp_path))
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
            wrong_type = await client.post("/api/cases", content=b"zip", headers={"Content-Type": "multipart/form-data"})
            loaded = await client.post("/api/cases", content=b"zip", headers={"Content-Type": "application/zip"})
            case_id = loaded.json()["id"]
            invalid_slice = await client.get(f"/api/cases/{case_id}/slice/2")
            valid_slice = await client.get(f"/api/cases/{case_id}/slice/1")

    assert wrong_type.status_code == 415
    assert wrong_type.json()["error"]["code"] == "unsupported_media_type"
    assert invalid_slice.status_code == 400
    assert invalid_slice.json()["error"]["code"] == "invalid_slice"
    assert valid_slice.status_code == 200
    assert valid_slice.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_two_cases_keep_masks_dice_and_progress_isolated_when_segments_finish_out_of_order(monkeypatch, tmp_path):
    started_a = asyncio.Event()
    release_a = asyncio.Event()
    finished_b = asyncio.Event()

    def fake_load(raw: bytes) -> CTVolume:
        return tiny_ct(1 if raw == b"A" else 2)

    def fake_load_mask(raw: bytes) -> Mask:
        return tiny_mask(raw)

    def fake_render(ct: CTVolume, index: int, overlays) -> bytes:
        assert len(overlays) == 1
        mask = overlays[0][0]
        identity = b"A" if mask.data[0, 0, 0] else b"B"
        return f"ct-{int(ct.pixels[0, 0, 0])}:".encode() + identity

    class DelayedSegmentationClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def segment(self, raw_zip: bytes, progress):
            if raw_zip == b"A":
                progress("A: waiting for backend")
                started_a.set()
                await release_a.wait()
                return "case-001", [("lesions", b"A")]
            progress("B: complete first")
            finished_b.set()
            return "case-001", [("lesions", b"B")]

    monkeypatch.setattr(app_module, "load_dicom_zip", fake_load)
    monkeypatch.setattr(app_module, "load_mask", fake_load_mask)
    monkeypatch.setattr(app_module, "render_slice", fake_render)
    monkeypatch.setattr(app_module, "SegmentationClient", DelayedSegmentationClient)
    application = app_module.create_app(make_settings(tmp_path, token="configured-for-test"))
    async with application.router.lifespan_context(application):
        # Both backend replies use case-001, so only per-local-case state can prevent mixing.
        application.state.reference = tiny_mask(b"A")
        application.state.reference_error = ""
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
            first = await client.post("/api/cases", content=b"A", headers={"Content-Type": "application/zip"})
            second = await client.post("/api/cases", content=b"B", headers={"Content-Type": "application/zip"})
            first_id, second_id = first.json()["id"], second.json()["id"]
            assert first_id != second_id

            pending_a = asyncio.create_task(client.post(f"/api/cases/{first_id}/segment"))
            await asyncio.wait_for(started_a.wait(), timeout=1)
            pending_b = asyncio.create_task(client.post(f"/api/cases/{second_id}/segment"))
            await asyncio.wait_for(finished_b.wait(), timeout=1)
            result_b = await asyncio.wait_for(pending_b, timeout=1)

            status_a = await client.get(f"/api/cases/{first_id}")
            assert status_a.json()["status"] == "processing"
            assert status_a.json()["progress"] == "A: waiting for backend"

            release_a.set()
            result_a = await asyncio.wait_for(pending_a, timeout=1)
            image_a = await client.get(f"/api/cases/{first_id}/slice/0?structures=model-0")
            image_b = await client.get(f"/api/cases/{second_id}/slice/0?structures=model-0")

    assert result_a.status_code == result_b.status_code == 200
    assert result_a.json()["dice"] == 1.0
    assert result_b.json()["dice"] == 0.0
    assert result_a.json()["structures"] == [{"id": "model-0", "name": "lesions", "color": app_module.MODEL_COLOURS[0]}]
    assert result_b.json()["structures"] == [{"id": "model-0", "name": "lesions", "color": app_module.MODEL_COLOURS[0]}]
    assert image_a.content == b"ct-1:A"
    assert image_b.content == b"ct-2:B"


@pytest.mark.asyncio
async def test_missing_token_still_allows_loading_and_slice_review(monkeypatch, tmp_path):
    monkeypatch.setattr(app_module, "load_dicom_zip", lambda raw: tiny_ct())
    monkeypatch.setattr(app_module, "render_slice", lambda ct, index, overlays: b"plain-png")
    application = app_module.create_app(make_settings(tmp_path, token=""))
    async with application.router.lifespan_context(application):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
            config = await client.get("/api/config")
            loaded = await client.post("/api/cases", content=b"zip", headers={"Content-Type": "application/zip"})
            case_id = loaded.json()["id"]
            image = await client.get(f"/api/cases/{case_id}/slice/0")
            segment = await client.post(f"/api/cases/{case_id}/segment")
            after_failure = await client.get(f"/api/cases/{case_id}")

    assert config.json()["token_configured"] is False
    assert loaded.status_code == image.status_code == 200
    assert image.content == b"plain-png"
    assert segment.status_code == 400
    assert segment.json()["error"]["code"] == "missing_token"
    assert after_failure.json()["status"] == "error"
    assert "NUCLEO_API_TOKEN" in segment.json()["error"]["message"]
