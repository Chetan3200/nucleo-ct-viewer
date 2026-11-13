import json

import httpx
import pytest

from nucleo_viewer.api_client import MAX_MASK_BYTES, SegmentationClient
from nucleo_viewer.errors import AppError

DUMMY_TOKEN = "test-token-for-mocks-only"
BASE_URL = "https://segmentation.test"


async def make_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=BASE_URL)


@pytest.mark.asyncio
async def test_segment_sends_raw_zip_and_authenticates_post_and_mask_get():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"case_id": "backend-case", "structures": [{"name": "lesions", "url": "/v1/masks/lesions"}]},
            )
        return httpx.Response(200, content=b"mask-bytes")

    client = await make_client(handler)
    try:
        remote = SegmentationClient(client, BASE_URL, DUMMY_TOKEN)
        case_id, masks = await remote.segment(b"PK\x03\x04tiny-zip", lambda _: None)
    finally:
        await client.aclose()

    assert case_id == "backend-case"
    assert masks == [("lesions", b"mask-bytes")]
    assert [request.method for request in seen] == ["POST", "GET"]
    assert seen[0].content == b"PK\x03\x04tiny-zip"
    assert seen[0].headers["content-type"] == "application/zip"
    assert all(request.headers["authorization"] == f"Bearer {DUMMY_TOKEN}" for request in seen)
    assert seen[1].url.path == "/v1/masks/lesions"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401])
async def test_non_retryable_upload_errors_are_not_retried(status: int):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(status, json={"error": {"code": "bad_request", "message": "untrusted remote text"}})

    client = await make_client(handler)
    try:
        remote = SegmentationClient(client, BASE_URL, DUMMY_TOKEN, max_retries=3)
        with pytest.raises(AppError) as raised:
            await remote.segment(b"zip", lambda _: None)
    finally:
        await client.aclose()

    assert calls == 1
    assert raised.value.code == "upstream_error"
    assert "untrusted remote text" not in raised.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "headers", "payload", "expected_delay"),
    [
        (429, {"Retry-After": "5"}, {"error": {}}, 5.0),
        (503, {}, {"error": {"details": {"retry_after_s": 6}}}, 6.0),
    ],
)
async def test_busy_retries_honor_retry_after_with_fake_sleep(status, headers, payload, expected_delay):
    calls = 0
    sleeps = []
    progress = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status, headers=headers, content=json.dumps(payload).encode())
        return httpx.Response(200, content=b"ok")

    client = await make_client(handler)
    try:
        remote = SegmentationClient(client, BASE_URL, DUMMY_TOKEN, sleep=fake_sleep, random_value=lambda: 0.0)
        body = await remote._request("GET", "/v1/masks/a", limit=MAX_MASK_BYTES, progress=progress.append)
    finally:
        await client.aclose()

    assert body == b"ok"
    assert calls == 2
    assert sleeps == [expected_delay]
    assert len(progress) == 1


@pytest.mark.asyncio
async def test_busy_retries_are_bounded():
    calls = 0
    sleeps = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, headers={"Retry-After": "1"}, json={"error": {}})

    client = await make_client(handler)
    try:
        remote = SegmentationClient(client, BASE_URL, DUMMY_TOKEN, sleep=fake_sleep, random_value=lambda: 0.0, max_retries=2)
        with pytest.raises(AppError) as raised:
            await remote._request("GET", "/v1/masks/a", limit=MAX_MASK_BYTES, progress=lambda _: None)
    finally:
        await client.aclose()

    assert raised.value.code == "upstream_error"
    assert calls == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_corrupt_json_and_mask_http_failure_are_safe_errors():
    def corrupt_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not valid json")

    client = await make_client(corrupt_handler)
    try:
        with pytest.raises(AppError, match="unreadable results"):
            await SegmentationClient(client, BASE_URL, DUMMY_TOKEN).segment(b"zip", lambda _: None)
    finally:
        await client.aclose()

    calls = []

    def download_error_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json={"case_id": "backend-case", "structures": [{"name": "lesions", "url": "/v1/masks/a"}]})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "remote message"}})

    client = await make_client(download_error_handler)
    try:
        with pytest.raises(AppError) as raised:
            await SegmentationClient(client, BASE_URL, DUMMY_TOKEN).segment(b"zip", lambda _: None)
    finally:
        await client.aclose()

    assert calls == ["POST", "GET"]
    assert raised.value.code == "upstream_error"
    assert "remote message" not in raised.value.message


@pytest.mark.asyncio
async def test_off_origin_mask_url_is_rejected_before_any_download_get():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"case_id": "backend-case", "structures": [{"name": "lesions", "url": "https://evil.example/mask"}]})

    client = await make_client(handler)
    try:
        with pytest.raises(AppError) as raised:
            await SegmentationClient(client, BASE_URL, DUMMY_TOKEN).segment(b"zip", lambda _: None)
    finally:
        await client.aclose()

    assert raised.value.code == "invalid_response"
    assert [request.method for request in seen] == ["POST"]
