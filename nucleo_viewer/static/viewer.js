import { createRaceGuards } from './state.mjs';

const guards = createRaceGuards();
const ui = {
  file: document.querySelector('#case-file'), fileName: document.querySelector('#file-name'), sample: document.querySelector('#load-sample'),
  analyze: document.querySelector('#analyze'), analysisHelp: document.querySelector('#analysis-help'), connection: document.querySelector('#connection-status'),
  caseId: document.querySelector('#case-id'), dimensions: document.querySelector('#case-dimensions'), sliceReadout: document.querySelector('#slice-readout'),
  models: document.querySelector('#model-overlays'), reference: document.querySelector('#reference-overlay'), dice: document.querySelector('#dice-value'), diceNote: document.querySelector('#dice-note'),
  image: document.querySelector('#ct-image'), empty: document.querySelector('#empty-viewport'), viewport: document.querySelector('#viewport'),
  slider: document.querySelector('#slice-slider'), number: document.querySelector('#slice-number'), total: document.querySelector('#slice-total'),
  progress: document.querySelector('#progress'), dot: document.querySelector('#status-dot'),
};

const view = { caseId: null, sliceCount: 0, sliceIndex: 0, structures: [], referenceAvailable: false, selection: 0, analyzing: false, status: 'ready' };
let imageController = null;
let displayedUrl = null;
let imageErrorMessage = null;
let wheelTimer = null;
let pendingWheelIndex = null;
let pollTimer = null;
let segmentRun = 0;

function setStatus(message, isError = false) {
  ui.progress.textContent = message || 'Working…';
  ui.dot.classList.toggle('error', isError);
}
function clearImage(message = 'Load a CT case to begin review') {
  if (imageController) imageController.abort();
  imageController = null;
  if (displayedUrl) URL.revokeObjectURL(displayedUrl);
  displayedUrl = null;
  ui.image.removeAttribute('src'); ui.image.hidden = true; ui.empty.textContent = message; ui.empty.hidden = false; ui.viewport.setAttribute('aria-busy', 'false');
}
function resetSelection() {
  view.caseId = null; view.sliceCount = 0; view.sliceIndex = 0; view.structures = []; view.referenceAvailable = false; view.analyzing = false; view.status = 'ready';
  imageErrorMessage = null;
  if (pollTimer) clearTimeout(pollTimer); pollTimer = null;
  if (wheelTimer) clearTimeout(wheelTimer); wheelTimer = null; pendingWheelIndex = null;
  clearImage();
  ui.analyze.disabled = true; ui.analysisHelp.textContent = 'Load a case before requesting model overlays.';
  ui.caseId.textContent = '—'; ui.dimensions.textContent = '—'; ui.sliceReadout.textContent = '—';
  ui.slider.min = '0'; ui.slider.max = '0'; ui.slider.value = '0'; ui.slider.disabled = true;
  ui.number.min = '0'; ui.number.max = '0'; ui.number.value = '0'; ui.number.disabled = true; ui.number.setCustomValidity(''); ui.total.textContent = '/ —';
  ui.reference.checked = false; ui.reference.disabled = true;
  ui.models.replaceChildren(); const empty = document.createElement('p'); empty.className = 'empty-state'; empty.textContent = 'Run analysis to view model overlays.'; ui.models.append(empty);
  ui.dice.textContent = '—'; ui.diceNote.textContent = 'Available when model and reference annotations are present.';
}
function errorFromResponse(payload, fallback) {
  return payload && payload.error && typeof payload.error.message === 'string' ? payload.error.message : fallback;
}
async function jsonResponse(response) {
  let payload = null;
  try { payload = await response.json(); } catch { /* handled below */ }
  if (!response.ok) throw new Error(errorFromResponse(payload, `Request failed (${response.status}).`));
  return payload;
}
function selectedStructures() {
  return [...ui.models.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
}
function renderModels() {
  ui.models.replaceChildren();
  if (!view.structures.length) {
    const empty = document.createElement('p'); empty.className = 'empty-state';
    empty.textContent = view.analyzing ? 'Waiting for model overlays…' : (view.status === 'complete' ? 'No model overlays were returned for this case.' : 'Run analysis to view model overlays.');
    ui.models.append(empty);
  }
  for (const structure of view.structures) {
    const label = document.createElement('label'); label.className = 'overlay-row';
    const input = document.createElement('input'); input.type = 'checkbox'; input.value = structure.id; input.checked = true; input.setAttribute('aria-label', `Show ${structure.name} model overlay`);
    const swatch = document.createElement('span'); swatch.className = 'swatch'; swatch.style.backgroundColor = structure.color || '#75a9ff'; swatch.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span'); name.textContent = structure.name || structure.id;
    input.addEventListener('change', () => requestSlice(view.sliceIndex));
    label.append(input, swatch, name); ui.models.append(label);
  }
  ui.reference.disabled = !view.referenceAvailable;
  if (!view.referenceAvailable) ui.reference.checked = false;
}
function applyMetadata(metadata, selection) {
  if (!guards.isCurrentSelection(selection)) return false;
  if (!metadata || typeof metadata.id !== 'string' || !Number.isInteger(metadata.slice_count) || metadata.slice_count < 1) throw new Error('The server returned invalid case metadata.');
  const referenceChecked = view.caseId === metadata.id && view.referenceAvailable ? ui.reference.checked : true;
  view.caseId = metadata.id; view.sliceCount = metadata.slice_count; view.structures = Array.isArray(metadata.structures) ? metadata.structures : [];
  view.referenceAvailable = Boolean(metadata.reference_available);
  view.status = metadata.status || 'ready';
  const desired = Number.isInteger(view.sliceIndex) ? view.sliceIndex : Math.floor(view.sliceCount / 2);
  view.sliceIndex = Math.max(0, Math.min(view.sliceCount - 1, desired));
  ui.caseId.textContent = metadata.id; ui.dimensions.textContent = `${metadata.rows ?? '—'} × ${metadata.columns ?? '—'} × ${metadata.slice_count}`;
  ui.slider.min = '0'; ui.slider.max = String(view.sliceCount - 1); ui.slider.value = String(view.sliceIndex); ui.slider.disabled = false;
  ui.number.min = '0'; ui.number.max = String(view.sliceCount - 1); ui.number.value = String(view.sliceIndex); ui.number.disabled = false;
  ui.total.textContent = `/ ${view.sliceCount - 1}`; ui.sliceReadout.textContent = `${view.sliceIndex} / ${view.sliceCount - 1}`;
  renderModels();
  ui.reference.checked = view.referenceAvailable && referenceChecked;
  ui.dice.textContent = typeof metadata.dice === 'number' ? metadata.dice.toFixed(3) : '—';
  ui.diceNote.textContent = metadata.dice_note || 'No Dice score is available.';
  ui.analyze.disabled = view.analyzing;
  ui.analysisHelp.textContent = view.analyzing ? 'Analysis is in progress for this case.' : (view.structures.length ? 'Choose any returned overlays to add them to the image.' : 'Request lesion segmentation for this case.');
  setStatus(metadata.progress || (metadata.status === 'complete' ? 'Analysis complete.' : 'Case ready.'));
  return true;
}
async function requestSlice(index) {
  if (!view.caseId || !view.sliceCount) return;
  const selection = view.selection;
  const clamped = Math.max(0, Math.min(view.sliceCount - 1, Math.trunc(index)));
  view.sliceIndex = clamped; ui.slider.value = String(clamped); ui.number.value = String(clamped); ui.sliceReadout.textContent = `${clamped} / ${view.sliceCount - 1}`;
  const ticket = guards.beginImage(selection);
  // Keep the last successful image visible until its replacement is decoded.
  if (imageController) imageController.abort();
  const controller = new AbortController(); imageController = controller;
  const query = new URLSearchParams();
  const structures = selectedStructures(); if (structures.length) query.set('structures', structures.join(',')); if (ui.reference.checked) query.set('reference', 'true');
  ui.viewport.setAttribute('aria-busy', 'true');
  let objectUrl = null;
  try {
    const suffix = query.toString() ? `?${query}` : '';
    const response = await fetch(`/api/cases/${encodeURIComponent(view.caseId)}/slice/${clamped}${suffix}`, { signal: controller.signal });
    if (!response.ok) {
      let payload = null; try { payload = await response.json(); } catch {}
      const error = new Error(errorFromResponse(payload, `Could not load slice (${response.status}).`));
      error.code = payload?.error?.code;
      throw error;
    }
    const blob = await response.blob(); objectUrl = URL.createObjectURL(blob);
    const decoder = new Image(); decoder.src = objectUrl; await decoder.decode();
    if (!guards.isCurrentImage(ticket) || controller.signal.aborted) return;
    if (displayedUrl) URL.revokeObjectURL(displayedUrl);
    displayedUrl = objectUrl; objectUrl = null;
    ui.image.src = displayedUrl; ui.image.hidden = false; ui.empty.hidden = true;
    // A successful retry clears only its own error, not an unrelated analysis status.
    if (imageErrorMessage && ui.progress.textContent === imageErrorMessage) {
      setStatus(view.analyzing ? 'Analysis is in progress.' : 'Image slice loaded.');
    }
    imageErrorMessage = null;
  } catch (error) {
    if (!controller.signal.aborted && guards.isCurrentImage(ticket)) {
      if (error.code === 'case_expired') {
        view.selection = guards.beginSelection();
        resetSelection();
        ui.empty.textContent = 'This scan is no longer loaded. Please load it again.';
      } else {
        clearImage('Could not load this image slice. Try again.');
      }
      imageErrorMessage = error.message || 'Could not load image slice.';
      setStatus(imageErrorMessage, true);
    }
  } finally {
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    if (guards.isCurrentImage(ticket)) ui.viewport.setAttribute('aria-busy', 'false');
  }
}
function setSliceFromNumber() {
  const value = Number(ui.number.value);
  const valid = Number.isInteger(value) && value >= 0 && value < view.sliceCount;
  ui.number.setCustomValidity(valid ? '' : `Enter a whole slice number from 0 to ${Math.max(0, view.sliceCount - 1)}.`);
  if (!valid) { ui.number.reportValidity(); return; }
  requestSlice(value);
}
async function selectCase(source, file = null) {
  const selection = guards.beginSelection(); view.selection = selection; resetSelection();
  setStatus(source === 'sample' ? 'Loading supplied case…' : 'Uploading CT ZIP…');
  try {
    const response = source === 'sample'
      ? await fetch('/api/sample', { method: 'POST' })
      : await fetch('/api/cases', { method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file });
    const metadata = await jsonResponse(response);
    if (!guards.isCurrentSelection(selection)) return;
    view.sliceIndex = Math.floor(metadata.slice_count / 2);
    applyMetadata(metadata, selection);
    await requestSlice(view.sliceIndex);
  } catch (error) {
    if (guards.isCurrentSelection(selection)) { setStatus(error.message || 'Could not load this case.', true); ui.analysisHelp.textContent = 'Load a valid CT ZIP or the supplied case to continue.'; }
  }
}
function schedulePoll(caseId, selection, run) {
  pollTimer = setTimeout(async () => {
    if (!guards.isCurrentSelection(selection) || run !== segmentRun || !view.analyzing || view.caseId !== caseId) return;
    try {
      const response = await fetch(`/api/cases/${encodeURIComponent(caseId)}`);
      const metadata = await jsonResponse(response);
      if (!guards.isCurrentSelection(selection) || run !== segmentRun || !view.analyzing || view.caseId !== caseId) return;
      setStatus(metadata.progress || 'Analysis is still running.');
    } catch (error) {
      if (guards.isCurrentSelection(selection) && run === segmentRun && view.analyzing) setStatus(`Waiting for analysis: ${error.message || 'temporary status error.'}`, true);
    }
    if (guards.isCurrentSelection(selection) && run === segmentRun && view.analyzing) schedulePoll(caseId, selection, run);
  }, 1000);
}
async function analyzeCase() {
  if (!view.caseId || view.analyzing) return;
  const selection = view.selection; const caseId = view.caseId; const run = ++segmentRun;
  view.analyzing = true; ui.analyze.disabled = true; ui.analysisHelp.textContent = 'Analysis is in progress for this case.';
  // Clear previous model results, but preserve the local reference and its toggle.
  view.structures = []; view.status = 'processing'; renderModels();
  ui.dice.textContent = '—'; ui.diceNote.textContent = 'Waiting for model output.';
  clearImage(); requestSlice(view.sliceIndex);
  setStatus('Submitting analysis…');
  schedulePoll(caseId, selection, run);
  try {
    const response = await fetch(`/api/cases/${encodeURIComponent(caseId)}/segment`, { method: 'POST' });
    const metadata = await jsonResponse(response);
    if (!guards.isCurrentSelection(selection) || run !== segmentRun || view.caseId !== caseId) return;
    view.analyzing = false; if (pollTimer) clearTimeout(pollTimer); pollTimer = null;
    applyMetadata(metadata, selection);
    await requestSlice(view.sliceIndex);
  } catch (error) {
    if (guards.isCurrentSelection(selection) && run === segmentRun && view.caseId === caseId) {
      view.analyzing = false; view.status = 'error'; renderModels();
      if (pollTimer) clearTimeout(pollTimer); pollTimer = null; ui.analyze.disabled = false;
      ui.diceNote.textContent = 'No model result available.';
      ui.analysisHelp.textContent = 'Analysis did not complete. You can retry this case.'; setStatus(error.message || 'Analysis failed.', true);
    }
  }
}

ui.file.addEventListener('change', () => {
  const file = ui.file.files && ui.file.files[0];
  ui.file.setCustomValidity('');
  if (!file) { ui.fileName.textContent = 'No file selected'; return; }
  ui.fileName.textContent = file.name;
  ui.file.value = ''; // Selecting the same file again must trigger change.
  if (!/\.zip$/i.test(file.name) || file.size > 150 * 1024 * 1024) {
    view.selection = guards.beginSelection(); resetSelection();
    setStatus(!/\.zip$/i.test(file.name) ? 'Choose a .zip file containing one CT series.' : 'Choose a ZIP smaller than 150 MiB.', true);
    return;
  }
  selectCase('file', file);
});
ui.sample.addEventListener('click', () => { ui.fileName.textContent = 'Supplied case-001'; selectCase('sample'); });
ui.analyze.addEventListener('click', analyzeCase);
ui.slider.addEventListener('input', () => requestSlice(Number(ui.slider.value)));
ui.number.addEventListener('change', setSliceFromNumber);
ui.number.addEventListener('input', () => ui.number.setCustomValidity(''));
ui.reference.addEventListener('change', () => requestSlice(view.sliceIndex));
ui.viewport.addEventListener('wheel', (event) => {
  if (!view.sliceCount) return; event.preventDefault();
  const base = pendingWheelIndex ?? view.sliceIndex; pendingWheelIndex = Math.max(0, Math.min(view.sliceCount - 1, base + (event.deltaY > 0 ? 1 : -1)));
  if (wheelTimer) clearTimeout(wheelTimer);
  wheelTimer = setTimeout(() => { const target = pendingWheelIndex; pendingWheelIndex = null; requestSlice(target); }, 75);
}, { passive: false });

(async function loadConfig() {
  const selection = guards.currentSelection();
  try {
    const response = await fetch('/api/config'); const config = await jsonResponse(response);
    ui.sample.disabled = !config.sample_available;
    if (!config.sample_available) ui.sample.title = 'No supplied case is available on this server.';
    ui.connection.textContent = config.token_configured ? 'Segmentation service configured' : 'Set NUCLEO_API_TOKEN in .env and restart the app to enable analysis.';
  } catch (error) { ui.connection.textContent = 'Could not read server configuration.'; if (guards.isCurrentSelection(selection)) setStatus(error.message || 'Configuration request failed.', true); }
})();
