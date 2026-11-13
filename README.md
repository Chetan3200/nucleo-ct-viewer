# Nucleo CT viewer

A local CT and segmentation-comparison viewer built with Python, FastAPI and plain JavaScript. This is research/demo software, not a clinical tool or diagnosis.

## Setup and run

Requires **uv**. Python 3.12 is pinned; dependencies are locked in `uv.lock`. Run these setup commands from the repository directory:

```bash
uv sync --locked
cp .env.example .env
```

Edit `.env` and set `NUCLEO_API_TOKEN` to your personal token. Never commit it or put it in browser code. Then launch with one command:

```bash
uv run --locked python -m nucleo_viewer
```

Open <http://127.0.0.1:8000>. Stop with Ctrl+C. Configuration:

- `NUCLEO_API_BASE_URL`: defaults to `https://bnj2vb4zpt.us-east-1.awsapprunner.com`.
- `NUCLEO_API_TOKEN`: required for segmentation, not local CT viewing.
- `NUCLEO_DATA_DIR`: optional path to the supplied data folder.
- `PORT`: local port, default 8000.

Restart after configuration or Python backend changes, then refresh the browser and reload the CT. The app binds to loopback; use one worker because cases are stored in memory.

By default, the supplied files should be in `Take-home case-001/`: `case-001-dicom.zip` and `case-001-reference-lesions.nii.gz`. The original files are unchanged and excluded from Git, along with `.env`. A reviewer must supply these files or configure their location. ZIP selection also works without the bundled sample.

## Use

1. Load the supplied case or choose its DICOM ZIP. The local reference overlay is enabled immediately, without analysis or an API token.
2. Scroll over the image, use the slider, or enter a zero-based slice index. The fixed window is level 40 HU / width 400 HU.
3. Click **Analyze**. The original ZIP bytes are sent to the API, not multipart data. Model masks come only from the API response.
4. Toggle each returned structure and the separately coloured reference. All available overlays start enabled after success.
5. Another case can be loaded while analysis is pending. The latest selection owns the screen; older results and errors cannot overwrite it.

Before analysis, the app recognises the supplied ZIP by comparing its contents (SHA-256) with the bundled ZIP, independently of its filename. Matching dimensions alone never attach a reference to an unrelated scan. Repacked or changed ZIPs can still receive the reference after the API identifies them as `case-001`. The local reference stays available while analysis runs and if it fails; model overlays and Dice still require a successful analysis.

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

Dice compares the full 3D model `lesions` mask with the supplied reference. The structure list is dynamic; the name `lesions` only selects the mask comparable to this reference. Shapes and affines must match (absolute tolerance `1e-4`, zero relative tolerance). No resampling is used for the score.

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

## Tests and verification

After setup:

```bash
uv run --locked pytest -q
node --test tests/ui_state.test.mjs
```

Node is only needed for the JavaScript tests, not the app. Python tests cover corrupt uploads and recovery, geometry/overlay alignment, Dice edge cases, authentication/raw ZIP bodies, retries, malformed responses and out-of-order concurrent analyses. HTTP tests are mocked. The supplied-data smoke test skips when the data is absent.

With the supplied data present, verification passed all 64 Python tests (including readable-error regressions, reference rendering before analysis, preservation during/after failed analysis, rejection of unrelated same-geometry scans, and layout regressions). The final source-only check passed 62 Python tests with the two supplied-data tests skipped because the downloaded data files were not in the default folder. Five JavaScript race tests and the lockfile consistency check also passed. An AST comparison confirmed that the error-message cleanup did not change non-message application code. Browser testing against the running app loaded the actual supplied ZIP, displayed CT slices and navigated between them. The empty image state was verified, and study-ID wrapping was checked at 390, 600, 1024 and 1440px widths. Earlier isolated browser checks also covered mocked concurrent selection, dynamic overlays and failed reanalysis.

**Live check:** Chetan tested analysis and both overlay controls. A subsequent live run was independently observed to complete, with the model lesions and reference overlays visible and a displayed 3D Dice score of **0.928** on the supplied case. This is a result for this single case, not a general model-accuracy claim.

## Limitations and next steps

Supported: axis-aligned axial, single-frame, uncompressed little-endian 8/16-bit CT, including in-plane rotations/flips. Oblique scans and gantry tilt/drift are rejected. Geometry cannot prove a uniformly cropped series is complete; the service decides whether it recognises the case.

Unknown NIfTI units assume mm. Native-pixel PNGs do not correct nonsquare pixel aspect ratios; the supplied case has square pixels. Measurements, extra windows and deeper header-versus-API-metadata checks are omitted.

Before submitting, rerun the tests and review overlay alignment and the displayed Dice score. Do not load-test the shared service. Separately prepare for the interview using the OmniLesion repository requested in the brief.

## Attribution and AI assistance

Wasserthal, J. (2026). *Liver lesions CT training dataset*. Zenodo. <https://doi.org/10.5281/zenodo.20272572>. CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>. Use only the supplied public case for this assignment; synthetic test arrays are not model predictions.

AI tools assisted planning, implementation, review and tests. Review the code and run verification before submitting so you can explain and modify it in the interview.
