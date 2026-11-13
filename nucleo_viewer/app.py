"""Thin local server: async network I/O, image work in worker threads."""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .api_client import SegmentationClient
from .errors import AppError
from .imaging import CTVolume, Mask, load_dicom_zip, load_mask, render_slice, validate_mask_geometry
from .metrics import dice_score

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAX_UPLOAD = 150 * 1024 * 1024
MAX_CASES = 3
CASE_TTL = 20 * 60
MODEL_COLOURS = ["#ff6b55", "#ffce56", "#bf94ff", "#7fdd78", "#ff8ed4", "#69a9ff"]
REFERENCE_COLOUR = "#00d3cc"


@dataclass
class Case:
    id: str
    ct: CTVolume
    raw_zip: bytes
    structures: list = field(default_factory=list)
    reference: Mask | None = None
    dice: float | None = None
    dice_note: str = "Analyse the scan to compare the model with the reference."
    status: str = "ready"
    progress: str = "CT loaded. Ready to analyse."
    busy: bool = False
    touched: float = field(default_factory=time.monotonic)

    def metadata(self):
        return {
            "id": self.id,
            "slice_count": self.ct.slice_count,
            "rows": self.ct.pixels.shape[1],
            "columns": self.ct.pixels.shape[2],
            "structures": [{"id": item["id"], "name": item["name"], "color": item["color"]} for item in self.structures],
            "reference_available": self.reference is not None,
            "reference_color": REFERENCE_COLOUR,
            "dice": self.dice,
            "dice_note": self.dice_note,
            "status": self.status,
            "progress": self.progress,
        }


@dataclass
class Settings:
    base_url: str = field(default_factory=lambda: os.getenv("NUCLEO_API_BASE_URL", "https://bnj2vb4zpt.us-east-1.awsapprunner.com"))
    token: str = field(default_factory=lambda: os.getenv("NUCLEO_API_TOKEN", ""), repr=False)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("NUCLEO_DATA_DIR", str(ROOT / "Take-home case-001"))))


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.cases = {}
        app.state.load_slots = asyncio.Semaphore(2)
        app.state.segment_slots = asyncio.Semaphore(2)
        app.state.render_slots = asyncio.Semaphore(2)
        app.state.reference = None
        app.state.sample_digest = None
        app.state.reference_error = "The reference overlay file was not found. Place it in the configured data folder and restart the app."
        reference_path = settings.data_dir / "case-001-reference-lesions.nii.gz"
        if reference_path.is_file():
            try:
                def read_reference():
                    if reference_path.stat().st_size > 32 * 1024 * 1024:
                        raise AppError("The reference file is too large.")
                    return load_mask(reference_path.read_bytes())
                app.state.reference = await asyncio.to_thread(read_reference)
                app.state.reference_error = ""
            except (AppError, OSError) as exc:
                app.state.reference_error = exc.message if isinstance(exc, AppError) else "The reference file could not be read."
        # Recognise the supplied ZIP locally, without an API call. Matching only
        # dimensions/geometry could attach this reference to an unrelated scan.
        sample_path = settings.data_dir / "case-001-dicom.zip"
        if app.state.reference is not None and sample_path.is_file():
            def hash_sample():
                if sample_path.stat().st_size > MAX_UPLOAD:
                    return None
                with sample_path.open("rb") as stream:
                    return hashlib.file_digest(stream, "sha256").digest()
            try:
                app.state.sample_digest = await asyncio.to_thread(hash_sample)
            except OSError:
                pass  # Local viewing still works; the API can identify the case later.
        timeout = httpx.Timeout(45, connect=10, write=70, pool=10)
        async with httpx.AsyncClient(timeout=timeout, transport=transport, follow_redirects=False) as client:
            app.state.http = client
            yield
        app.state.cases.clear()

    app = FastAPI(title="Nucleo CT viewer", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def same_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != f"{request.url.scheme}://{request.url.netloc}":
            return JSONResponse({"error": {"code": "origin", "message": "Open the viewer directly on this computer to upload a scan."}}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": exc.message}}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc):
        return JSONResponse({"error": {"code": "invalid_request", "message": "The requested slice or options are invalid."}}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request, exc):
        logging.error("Unexpected viewer error: %s", type(exc).__name__)
        return JSONResponse({"error": {"code": "internal", "message": "The viewer could not complete this action. Please try loading the scan again."}}, status_code=500)

    def get_case(case_id):
        case = app.state.cases.get(case_id)
        if case is None:
            raise AppError("This scan is no longer loaded. Please load it again.", 404, "case_expired")
        case.touched = time.monotonic()
        return case

    async def take_slot(semaphore):
        if semaphore.locked():
            raise AppError("Two requests are already running. Wait for one to finish, then try again.", 429, "local_busy")
        await semaphore.acquire()

    async def read_upload(request):
        header = request.headers.get("content-length")
        if header:
            try:
                length = int(header)
            except ValueError as exc:
                raise AppError("The upload has an invalid size.") from exc
            if length < 0 or length > MAX_UPLOAD:
                raise AppError("Choose a ZIP smaller than 150 MiB.", 413, "too_large")
        data = bytearray()
        try:
            async with asyncio.timeout(70):
                async for chunk in request.stream():
                    if len(data) + len(chunk) > MAX_UPLOAD:
                        raise AppError("Choose a ZIP smaller than 150 MiB.", 413, "too_large")
                    data.extend(chunk)
        except TimeoutError as exc:
            raise AppError("The upload took too long. Please try again.", 408, "timeout") from exc
        return bytes(data)

    async def add_case(raw):
        def prepare_case():
            ct = load_dicom_zip(raw)
            reference = None
            if app.state.sample_digest is not None and hashlib.sha256(raw).digest() == app.state.sample_digest:
                validate_mask_geometry(ct, app.state.reference)
                reference = app.state.reference
            return ct, reference
        ct, reference = await asyncio.to_thread(prepare_case)
        # No await between lookup/eviction/insertion: these mutations are atomic
        # on the single event loop. Busy cases are never evicted.
        cases = app.state.cases
        now = time.monotonic()
        for key in list(cases):
            if not cases[key].busy and now - cases[key].touched > CASE_TTL:
                del cases[key]
        if len(cases) >= MAX_CASES:
            idle = [case for case in cases.values() if not case.busy]
            if not idle:
                raise AppError("The viewer is busy. Wait for a scan to finish before loading another.", 429, "local_busy")
            del cases[min(idle, key=lambda item: item.touched).id]
        case = Case(id=uuid.uuid4().hex, ct=ct, raw_zip=raw, reference=reference)
        if reference is not None:
            case.progress = "CT and reference loaded. Ready to analyse."
        cases[case.id] = case
        return case.metadata()

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    async def config():
        return {"sample_available": (settings.data_dir / "case-001-dicom.zip").is_file(), "reference_available": app.state.reference is not None, "token_configured": bool(settings.token.strip())}

    @app.post("/api/cases")
    async def load_case(request: Request):
        if request.headers.get("content-type", "").split(";")[0] not in ("application/zip", "application/x-zip-compressed"):
            raise AppError("Select a DICOM ZIP file.", 415, "unsupported_media_type")
        await take_slot(app.state.load_slots)
        try:
            return await add_case(await read_upload(request))
        finally:
            app.state.load_slots.release()

    @app.post("/api/sample")
    async def load_sample():
        await take_slot(app.state.load_slots)
        try:
            def read_sample():
                path = settings.data_dir / "case-001-dicom.zip"
                if not path.is_file():
                    raise AppError("The supplied ZIP was not found. Select it manually or set NUCLEO_DATA_DIR.", 404, "missing_sample")
                if path.stat().st_size > MAX_UPLOAD:
                    raise AppError("The supplied ZIP is larger than 150 MiB.", 413, "too_large")
                return path.read_bytes()
            return await add_case(await asyncio.to_thread(read_sample))
        finally:
            app.state.load_slots.release()

    @app.get("/api/cases/{case_id}")
    async def case_status(case_id: str):
        return get_case(case_id).metadata()

    @app.post("/api/cases/{case_id}/segment")
    async def segment(case_id: str):
        case = get_case(case_id)
        if case.busy:
            raise AppError("This scan is already being analysed.", 409, "already_running")
        await take_slot(app.state.segment_slots)
        case.busy = True
        case.status = "processing"
        # Reference is local ground truth, independent of the model request.
        case.structures, case.dice = [], None
        case.dice_note = "Waiting for model output."
        try:
            client = SegmentationClient(app.state.http, settings.base_url, settings.token)
            def progress(message):
                case.progress = message
            async with asyncio.timeout(180):
                backend_id, downloads = await client.segment(case.raw_zip, progress)
                progress("Checking mask geometry and calculating Dice…")
                def prepare_results():
                    items = []
                    memory_bytes = 0
                    for i, (name, raw) in enumerate(downloads):
                        mask = load_mask(raw)
                        validate_mask_geometry(case.ct, mask)
                        memory_bytes += mask.data.nbytes
                        if memory_bytes > 128 * 1024 * 1024:
                            raise AppError("The returned overlays are too large for this viewer to display.", 502, "too_large")
                        items.append({"id": f"model-{i}", "name": name, "color": MODEL_COLOURS[i % len(MODEL_COLOURS)], "mask": mask})
                    reference = app.state.reference if backend_id == "case-001" else None
                    if reference is not None:
                        validate_mask_geometry(case.ct, reference)
                    score = None
                    note = "Dice is available only for the returned lesions mask and the supplied case-001 reference."
                    lesion = next((item["mask"] for item in items if item["name"].strip().lower() == "lesions"), None)
                    if reference is None:
                        note = app.state.reference_error if backend_id == "case-001" else "No reference is provided for this case."
                    elif lesion is not None:
                        try:
                            score = dice_score(lesion, reference)
                            note = "3D Dice: model lesions vs manual reference, on the original matching grids."
                        except AppError as exc:
                            note = f"Dice unavailable: {exc.message}"
                    return items, reference, score, note
                # Commit all output together only after every download/parse succeeds.
                case.structures, case.reference, case.dice, case.dice_note = await asyncio.to_thread(prepare_results)
            case.status, case.progress = "complete", "Segmentation complete. Toggle overlays to compare."
            return case.metadata()
        except asyncio.CancelledError:
            case.status, case.progress = "error", "Analysis was cancelled. Please try again."
            case.dice_note = "No model result available."
            raise
        except TimeoutError as exc:
            case.status, case.progress = "error", "Analysis timed out. Please try again later."
            case.dice_note = "No model result available."
            raise AppError(case.progress, 504, "timeout") from exc
        except AppError as exc:
            case.status, case.progress = "error", exc.message
            case.dice_note = "No model result available."
            raise
        except Exception as exc:
            case.status, case.progress = "error", "Could not read or process the model results. Please try again."
            case.dice_note = "No model result available."
            logging.error("Segmentation processing error: %s", type(exc).__name__)
            raise AppError(case.progress, 502, "invalid_response") from exc
        finally:
            case.busy = False
            case.touched = time.monotonic()
            app.state.segment_slots.release()

    @app.get("/api/cases/{case_id}/slice/{index}")
    async def slice_image(case_id: str, index: int, structures: str = "", reference: bool = False):
        case = get_case(case_id)
        if not 0 <= index < case.ct.slice_count:
            raise AppError("That slice is outside this scan.", 400, "invalid_slice")
        selected = set(filter(None, structures.split(",")))
        available = {item["id"]: item for item in case.structures}
        if selected - available.keys():
            raise AppError("One of the selected overlays is no longer available. Refresh the scan.", 400, "invalid_structure")
        overlays = []
        for key in available:
            if key in selected:
                item = available[key]
                colour = tuple(int(item["color"][i:i+2], 16) for i in (1, 3, 5))
                overlays.append((item["mask"], colour))
        if reference:
            if case.reference is None:
                raise AppError("The reference overlay is not available for this scan.", 400, "missing_reference")
            overlays.append((case.reference, (0, 211, 204)))
        await take_slot(app.state.render_slots)
        try:
            png = await asyncio.to_thread(render_slice, case.ct, index, overlays)
            return Response(content=png, media_type="image/png")
        finally:
            app.state.render_slots.release()

    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return app


app = create_app()
