# Nucleo CT viewer

A local axial CT viewer built with Python, FastAPI and plain JavaScript. It displays API-generated segmentation masks alongside a supplied reference mask and calculates their 3D Dice score.

This is research/demo software, not a clinical tool.

## Setup and run

Requires **uv**. Python 3.12 is pinned; dependencies are locked in `uv.lock`. Run these setup commands from the repository directory:

```bash
uv sync --locked
cp .env.example .env
```

Set `NUCLEO_API_TOKEN` in `.env` to enable segmentation. The token is used only by the Python backend and is not exposed to browser code.

Place the supplied imaging files in the following directory, or set `NUCLEO_DATA_DIR` in `.env` to their folder:

```text
Take-home case-001/
├── case-001-dicom.zip
└── case-001-reference-lesions.nii.gz
```

The imaging files and `.env` are excluded from Git. A manually selected DICOM ZIP can also be viewed without the supplied files.

Start the app:

```bash
uv run --locked python -m nucleo_viewer
```

Open <http://127.0.0.1:8000>. Stop with Ctrl+C. Configuration:

- `NUCLEO_API_BASE_URL`: defaults to `https://bnj2vb4zpt.us-east-1.awsapprunner.com`.
- `NUCLEO_API_TOKEN`: required for segmentation, not local CT viewing.
- `NUCLEO_DATA_DIR`: optional path to the supplied data folder.
- `PORT`: local port, default 8000.

Restart after configuration or Python backend changes, then refresh the browser and reload the CT. The app binds to loopback; use one worker because cases are stored in memory.

## Usage

1. Load the supplied case or select a supported DICOM ZIP. When both supplied files are configured and the selected ZIP matches the supplied archive, the reference overlay is enabled immediately, without analysis or an API token.
2. Scroll over the image, use the slider, or enter a zero-based slice index. The fixed window is level 40 HU / width 400 HU.
3. Click **Analyze**. The original ZIP bytes are sent to the API, not multipart data. Model masks come only from the API response.
4. Toggle each returned structure and the separately coloured reference. All available overlays start enabled after success.
5. Another case can be loaded while analysis is pending. The latest selection owns the screen; older results and errors cannot overwrite it.

While a new slice or overlay image loads, the previous image is hidden. A failed image request shows an error instead of stale pixels. An expired case clears the viewer and must be reloaded.

Before analysis, the app identifies the supplied archive by the SHA-256 hash of its ZIP bytes, independently of its filename. Matching dimensions alone never attach a reference to an unrelated scan. Repacked or changed ZIPs can receive the reference after the API identifies them as `case-001`, subject to geometry validation. An attached local reference remains available during analysis and after a failed analysis; model overlays and Dice require a successful analysis.

## Code map and async design

- `app.py`: local routes, per-case state and resource limits.
- `api_client.py`: HTTPX async uploads/downloads, retries and error mapping.
- `imaging.py`: DICOM/NIfTI reading, geometry and PNG slices.
- `metrics.py`: grid checking and Dice.
- `static/viewer.js`: interface events and display; `state.mjs` supplies stale-result guards.
- `tests/`: synthetic imaging, mocked HTTP/app, local-data and JavaScript tests.

Network calls and retry sleeps are async. Image processing runs through `asyncio.to_thread`, not on the event loop. There is no database, frontend build, inference backend or background-job service.

## Geometry and Dice

DICOM slices are sorted by physical position, not filename. RescaleSlope/Intercept convert pixels to HU. CT storage is `[slice, row, column]`; its affine maps `[column, row, slice, 1]` into RAS+ millimetres, converting DICOM's LPS coordinates.

Masks retain NIfTI `[i, j, k]` order. Overlay coordinates are computed as:

```text
mask_voxel = inverse(mask_affine) @ ct_affine @ ct_voxel
```

Nearest-neighbour sampling preserves binary labels. Completely disjoint physical bounding boxes are rejected; out-of-mask pixels are transparent. CT and overlays are oriented together: anterior at the top, patient's right at the viewer's left. Overlays use 45% opacity; reference is drawn last.

Dice compares the supplied reference with the first returned mask whose name is `lesions` (case-insensitive, ignoring surrounding whitespace). The structure list is dynamic; other returned masks are displayed as overlays but are not scored against this reference. Shapes and affines must match (absolute tolerance `1e-4`, zero relative tolerance). No resampling is used for the score.

```text
Dice = 2 * intersection_voxels / (model_foreground + reference_foreground)
```

Both empty masks score 1; one empty mask scores 0. Grid mismatch leaves Dice unavailable with an explanation. No target or measured score is hard-coded.

## Robustness

- Bounded raw uploads: 150 MiB; ZIPs: 5,000 entries, 64 MiB/member, 256 MiB expanded locally. Ignore directory/hidden/macOS metadata entries. Reject unsafe paths and corrupt data.
- Validate one supported CT series, dimensions, geometry, duplicate slices, spacing and pixel lengths. Bound volumes at 64 million voxels; expanded NIfTI below 128 MiB; combined model masks at 128 MiB. Require binary masks and explicit valid qform/sform geometry.
- HTTP timeouts and a 180-second analysis deadline. Retry only 429/503, at most three retries per request, respecting Retry-After and exponential backoff with jitter. A requested pause over 120 seconds stops safely rather than retrying early.
- Map API codes/reasons to plain messages; never display raw upstream messages or stack traces. Validate JSON and download URLs; no redirects or token forwarding to external mask URLs.
- Each upload has a unique local ID, even when backend case IDs match. Commit results together after all downloads/validation. Reanalysis clears old model masks/Dice, not the local reference. Separate browser guards cover selection and slice races.
- At most two loads, analyses and renders per category concurrently. Store at most three cases; evict the oldest idle case. On new loads, remove idle cases older than 20 minutes. Busy cases are not evicted; nothing persists after exit.

## Tests and observed results

After setup:

```bash
uv run --locked pytest -q
node --test tests/*.test.mjs
```

Node is only required for the JavaScript tests. The Python suite covers corrupt uploads and recovery, geometry and overlay alignment, Dice edge cases, authentication and raw ZIP requests, retries, malformed responses, reference-overlay lifecycle and concurrent analyses. HTTP tests use mocks and do not call the shared segmentation service.

- **Python:** 69 tests passed, including both supplied-data tests. These use `NUCLEO_DATA_DIR` from the environment, then `.env`, or the default `Take-home case-001/` folder. They skip if either required imaging file is absent.
- **JavaScript:** 14 tests passed, covering race guards and the actual viewer's image-request failure handling, retry recovery, expired-case reset and stale-response isolation. Viewer tests use lightweight DOM, network and image-decoder doubles without additional dependencies.
- **Supplied-case live result:** analysis completed with both model and reference overlays available, and a displayed 3D Dice score of **0.928**. This is a single-case result, not a general model-accuracy estimate.

## Limitations

Supported: axis-aligned axial, single-frame, monochrome, uncompressed little-endian 8/16-bit CT, including in-plane rotations/flips. Overlay files must be single-file, gzip-compressed NIfTI-1/2. Oblique scans and gantry tilt/drift are rejected. Geometry cannot prove a uniformly cropped series is complete; the service decides whether it recognises the case.

Unknown NIfTI units assume mm. Native-pixel PNGs do not correct nonsquare pixel aspect ratios; the supplied case has square pixels. Measurements, extra windows and deeper header-versus-API-metadata checks are omitted.

## Attribution and AI assistance

Supplied CT/reference data: Wasserthal, J. (2026). *Liver lesions CT training dataset*. Zenodo. <https://doi.org/10.5281/zenodo.20272572>. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data is not redistributed in this repository. Synthetic arrays are used only in tests, not as model predictions.

AI tools assisted with planning, implementation, code review and test development.
