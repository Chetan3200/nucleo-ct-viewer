"""Async client for the supplied service. No model outputs are cached or invented."""

import asyncio
import json
import math
import random
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from .errors import AppError

Progress = Callable[[str], None]
MAX_MASK_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 64 * 1024


def error_message(status: int, payload: object) -> str:
    """Branch on stable codes/reasons, never the server's free-text message."""
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        error = {}
    details = error.get("details", {})
    reason = details.get("reason") if isinstance(details, dict) else None
    reason = reason if isinstance(reason, str) else None
    code = error.get("code")
    if status == 401 or code == "unauthorized":
        if reason in ("expired", "revoked"):
            return "Your API token has expired or been revoked. Update .env with a valid token and restart the app."
        return "The service did not accept the API token. Check NUCLEO_API_TOKEN in .env and restart the app."
    if status == 404 or code == "not_found":
        return "The service does not recognise this scan." if reason == "unknown_series" else "The requested scan or overlay is not available on the service."
    if status == 413 or code == "too_large":
        return "This ZIP is too large. The service accepts files up to 150 MiB."
    if status == 415 or code == "unsupported_media_type":
        return "The service could not accept the upload format. Select the original CT ZIP and try again."
    if status == 422 or code == "unprocessable":
        messages = {
            "multiple_series": "The ZIP contains multiple scans. Select a ZIP containing one complete CT scan.",
            "missing_tags": "Some CT files are missing required image information.",
            "incomplete_series": "This scan has missing slices. Upload the complete series.",
            "inconsistent_spacing": "The spacing between CT slices is inconsistent.",
            "duplicate_position": "The scan contains duplicate slice positions.",
            "duplicate_instance": "The scan contains duplicate slices.",
            "unsupported_transfer_syntax": "This scan uses a storage format the service cannot read. Use the supplied CT ZIP.",
        }
        return messages.get(reason, "The CT files are readable, but they cannot be combined into one supported scan.")
    if status == 400 or code == "bad_request":
        return "The service could not read this DICOM ZIP. It may be empty, corrupt, or incomplete."
    if status == 408 or code == "timeout":
        return "The service took too long to receive or process the scan. Please try again."
    if status in (429, 503):
        return "The service is busy. Automatic retries have stopped; please try again later."
    return "The segmentation service could not complete the request. Please try again later."


class SegmentationClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        token: str,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_value: Callable[[], float] = random.random,
        max_retries: int = 3,
    ):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            raise AppError("The service address is invalid. Check NUCLEO_API_BASE_URL in .env and use the address provided for the assignment.", 400, "configuration")
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.token = token.strip()
        self.sleep = sleep
        self.random_value = random_value
        self.max_retries = max_retries

    def _url(self, path: str) -> str:
        # Never send the token to a URL supplied by an untrusted remote host.
        parts = urlsplit(path)
        if parts.scheme or parts.netloc or parts.fragment or not path.startswith("/v1/") or "\\" in path or ".." in path:
            raise AppError("The service provided an unsafe or invalid overlay download link.", 502, "invalid_response")
        return self.base_url + path

    async def _request(self, method: str, path: str, *, body: bytes | None = None, limit: int, progress: Progress) -> bytes:
        if not self.token:
            raise AppError("Set NUCLEO_API_TOKEN in .env and restart the app before analysing a scan.", 400, "missing_token")
        headers = {"Authorization": f"Bearer {self.token}"}
        if body is not None:
            headers["Content-Type"] = "application/zip"
        for attempt in range(self.max_retries + 1):
            try:
                async with self.client.stream(method, self._url(path), headers=headers, content=body, follow_redirects=False) as response:
                    allowed = limit if response.status_code == 200 else MAX_JSON_BYTES
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(data) + len(chunk) > allowed:
                            raise AppError("The service returned an unexpectedly large response.", 502, "invalid_response")
                        data.extend(chunk)
                    status = response.status_code
                    retry_header = response.headers.get("Retry-After")
                if status == 200:
                    return bytes(data)
                try:
                    payload = json.loads(data)
                except (ValueError, UnicodeError):
                    payload = {}
                if status not in (429, 503) or attempt == self.max_retries:
                    raise AppError(error_message(status, payload), 502, "upstream_error")
                details = payload.get("error", {}) if isinstance(payload, dict) else {}
                details = details.get("details", {}) if isinstance(details, dict) else {}
                fallback = details.get("retry_after_s") if isinstance(details, dict) else None
                retry_after = 0.0
                for candidate in (retry_header, fallback):
                    try:
                        value = float(candidate)
                    except (ValueError, TypeError, OverflowError):
                        continue
                    if math.isfinite(value) and value >= 0:
                        retry_after = max(retry_after, value)
                # Never shorten Retry-After. Stop rather than retry too early.
                delay = max(retry_after, 2**attempt + self.random_value())
                if delay > 120:
                    raise AppError("The service requested a long pause. Please try again later.", 503, "retry_later")
                progress(f"Service busy. Retrying in {math.ceil(delay)} seconds (retry {attempt + 1}/{self.max_retries})…")
                await self.sleep(delay)
            except httpx.TimeoutException as exc:
                raise AppError("The service did not respond in time. Please try again.", 504, "timeout") from exc
            except httpx.RequestError as exc:
                raise AppError("Could not reach the segmentation service. Check your connection and API base URL.", 502, "network_error") from exc
        raise AssertionError("Retry loop exhausted without a response")

    async def segment(self, raw_zip: bytes, progress: Progress) -> tuple[str, list[tuple[str, bytes]]]:
        progress("Uploading the CT and waiting for segmentation…")
        raw = await self._request("POST", "/v1/segmentations", body=raw_zip, limit=MAX_JSON_BYTES, progress=progress)
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise AppError("The service returned unreadable results. Please try again.", 502, "invalid_response") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("case_id"), str) or not payload["case_id"] or len(payload["case_id"]) > 128:
            raise AppError("The service did not identify the uploaded scan. Please try again.", 502, "invalid_response")
        structures = payload.get("structures")
        if not isinstance(structures, list) or len(structures) > 32:
            raise AppError("The service did not provide a usable list of overlays. Please try again.", 502, "invalid_response")
        seen = set()
        for item in structures:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip() or len(item["name"]) > 128 or not isinstance(item.get("url"), str):
                raise AppError("The service returned an overlay without a usable name or download link.", 502, "invalid_response")
            if item["name"] in seen:
                raise AppError("The service returned duplicate overlay names, so the results cannot be shown safely.", 502, "invalid_response")
            seen.add(item["name"])
            self._url(item["url"])
        masks = []
        total_bytes = 0
        for index, item in enumerate(structures):
            progress(f"Downloading mask {index + 1} of {len(structures)}…")
            mask = await self._request("GET", item["url"], limit=MAX_MASK_BYTES, progress=progress)
            total_bytes += len(mask)
            if total_bytes > 64 * 1024 * 1024:
                raise AppError("The service returned more overlay data than this viewer can handle.", 502, "invalid_response")
            masks.append((item["name"], mask))
        return payload["case_id"], masks
