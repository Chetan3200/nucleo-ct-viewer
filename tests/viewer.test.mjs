import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createRaceGuards } from '../nucleo_viewer/static/state.mjs';

const source = readFileSync(new URL('../nucleo_viewer/static/viewer.js', import.meta.url), 'utf8');

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function element(tag = 'div') {
  const classes = new Set(), attributes = new Map();
  return {
    tag, children: [], style: {}, textContent: '', value: '', hidden: false, disabled: false, checked: false,
    classList: {
      toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute(name, value) { attributes.set(name, value); },
    getAttribute(name) { return attributes.get(name); },
    removeAttribute(name) { attributes.delete(name); delete this[name]; },
    setCustomValidity() {}, addEventListener() {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    querySelectorAll() {
      return this.children.flatMap(child => [
        ...(child.tag === 'input' && child.type === 'checkbox' && child.checked ? [child] : []),
        ...child.querySelectorAll(),
      ]);
    },
  };
}

function json(payload, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => payload };
}
function imageResponse() { return { ok: true, blob: async () => ({}) }; }
function metadata(id = 'case-a') {
  return {
    id, slice_count: 180, rows: 350, columns: 416, reference_available: true,
    structures: [{ id: 'model-0', name: 'lesions', color: '#75a9ff' }],
    dice: 0.5, status: 'complete', progress: 'Analysis complete.',
  };
}

function harness() {
  const nodes = new Map([...source.matchAll(/document\.querySelector\('(#[^']+)'\)/g)]
    .map(([, selector]) => [selector, element()]));
  const pending = [], created = [], revoked = [], timers = new Map();
  let nextTimer = 0;
  const decoder = { fail: false, wait: null };
  const fetch = (url, options = {}) => {
    if (url === '/api/config') return Promise.resolve(json({ sample_available: true, token_configured: true }));
    const request = deferred(); pending.push({ url, options, ...request }); return request.promise;
  };
  // Execute the actual viewer script, replacing only its import with the real factory.
  // The small DOM/fetch doubles keep this test dependency-free and offline.
  const api = vm.runInNewContext(source.replace("import { createRaceGuards } from './state.mjs';", '')
    + '\n({ requestSlice, selectCase, analyzeCase });', {
    document: { querySelector: selector => nodes.get(selector), createElement: element },
    createRaceGuards, fetch, AbortController, URLSearchParams,
    URL: {
      createObjectURL() { const url = `blob:test-${created.length}`; created.push(url); return url; },
      revokeObjectURL(url) { revoked.push(url); },
    },
    Image: class { async decode() { if (decoder.wait) await decoder.wait; if (decoder.fail) throw new Error('Image decoding failed.'); } },
    setTimeout(fn) { const id = ++nextTimer; timers.set(id, fn); return id; },
    clearTimeout(id) { timers.delete(id); },
  });
  return {
    api, pending, created, revoked, decoder, timers,
    node: id => nodes.get(`#${id}`),
    async load(id = 'case-a') {
      const loading = api.selectCase('sample');
      pending.shift().resolve(json(metadata(id)));
      await new Promise(resolve => setImmediate(resolve));
      assert.match(pending[0].url, /\/slice\/90\?/);
      pending.shift().resolve(imageResponse());
      await loading;
    },
  };
}

function assertNoImage(h) {
  assert.equal(h.node('ct-image').hidden, true);
  assert.equal(h.node('ct-image').src, undefined);
  assert.equal(h.node('empty-viewport').hidden, false);
  assert.equal(h.node('viewport').getAttribute('aria-busy'), 'false');
}

test('requestSlice keeps the previous image while loading but clears it on a current HTTP failure', async () => {
  const h = harness(); await h.load();
  const oldUrl = h.node('ct-image').src;
  const loading = h.api.requestSlice(91);
  assert.equal(h.node('slice-number').value, '91');
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('ct-image').src, oldUrl);
  assert.equal(h.node('empty-viewport').hidden, true);
  assert.ok(!h.revoked.includes(oldUrl));
  assert.equal(h.node('viewport').getAttribute('aria-busy'), 'true');
  h.pending.shift().resolve(json({ error: { code: 'internal', message: 'Rendering failed.' } }, 500));
  await loading;
  assertNoImage(h);
  assert.equal(h.node('progress').textContent, 'Rendering failed.');
  assert.equal(h.node('case-id').textContent, 'case-a');
  assert.equal(h.node('slice-number').disabled, false);
  assert.equal(h.node('reference-overlay').checked, true);
  assert.ok(h.revoked.includes(oldUrl));
});

test('slice navigation swaps images only after the replacement has decoded', async () => {
  const h = harness(); await h.load();
  const oldUrl = h.node('ct-image').src, decoded = deferred();
  h.decoder.wait = decoded.promise;
  const loading = h.api.requestSlice(91);
  h.pending.shift().resolve(imageResponse());
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.node('ct-image').src, oldUrl);
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('empty-viewport').hidden, true);
  assert.ok(!h.revoked.includes(oldUrl));
  decoded.resolve(); await loading;
  assert.notEqual(h.node('ct-image').src, oldUrl);
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('empty-viewport').hidden, true);
  assert.ok(h.revoked.includes(oldUrl));
  assert.equal(h.node('viewport').getAttribute('aria-busy'), 'false');
});

test('failed reference-toggle request does not retain reference-containing pixels; retry recovers', async () => {
  const h = harness(); await h.load();
  h.node('reference-overlay').checked = false;
  const failed = h.api.requestSlice(90);
  assert.ok(!h.pending[0].url.includes('reference=true'));
  h.pending.shift().reject(new Error('Network unavailable.')); await failed;
  assertNoImage(h);
  assert.equal(h.node('reference-overlay').checked, false);
  const retry = h.api.requestSlice(90);
  h.pending.shift().resolve(imageResponse()); await retry;
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('empty-viewport').hidden, true);
  assert.equal(h.node('status-dot').classList.contains('error'), false);
});

test('an image decoding failure clears the image and releases the failed object URL', async () => {
  const h = harness(); await h.load(); h.decoder.fail = true;
  const loading = h.api.requestSlice(91);
  h.pending.shift().resolve(imageResponse()); await loading;
  assertNoImage(h);
  assert.equal(h.node('progress').textContent, 'Image decoding failed.');
  assert.ok(h.created.every(url => h.revoked.includes(url)));
});

test('a non-JSON image error uses the HTTP fallback without keeping old pixels', async () => {
  const h = harness(); await h.load();
  const loading = h.api.requestSlice(91);
  h.pending.shift().resolve({ ok: false, status: 502, json: async () => { throw new Error('Not JSON'); } });
  await loading;
  assertNoImage(h);
  assert.equal(h.node('progress').textContent, 'Could not load slice (502).');
});

test('case_expired clears controls, overlays and Dice and stops further slice requests', async () => {
  const h = harness(); await h.load();
  const loading = h.api.requestSlice(91);
  h.pending.shift().resolve(json({ error: { code: 'case_expired', message: 'This scan is no longer loaded. Please load it again.' } }, 404));
  await loading;
  assertNoImage(h);
  assert.equal(h.node('case-id').textContent, '—');
  assert.equal(h.node('dice-value').textContent, '—');
  for (const id of ['analyze', 'slice-number', 'slice-slider', 'reference-overlay']) assert.equal(h.node(id).disabled, true);
  assert.equal(h.node('reference-overlay').checked, false);
  assert.equal(h.node('model-overlays').querySelectorAll().length, 0);
  await h.api.requestSlice(92);
  assert.equal(h.pending.length, 0);
  await h.load('case-b');
  assert.equal(h.node('case-id').textContent, 'case-b');
  assert.equal(h.node('ct-image').hidden, false);
});

test('an old slice failure cannot clear a newer successful image', async () => {
  const h = harness(); await h.load();
  const first = h.api.requestSlice(91), old = h.pending.shift();
  const second = h.api.requestSlice(92); h.pending.shift().resolve(imageResponse()); await second;
  const currentUrl = h.node('ct-image').src;
  old.reject(new Error('Old request failed.')); await first;
  assert.equal(h.node('ct-image').src, currentUrl);
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('slice-number').value, '92');
  assert.ok(!h.revoked.includes(currentUrl));
});

test('an expired response from the previous selection cannot reset the new case', async () => {
  const h = harness(); await h.load();
  const first = h.api.requestSlice(91), old = h.pending.shift();
  await h.load('case-b');
  old.resolve(json({ error: { code: 'case_expired', message: 'Old case expired.' } }, 404)); await first;
  assert.equal(h.node('case-id').textContent, 'case-b');
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('slice-number').disabled, false);
});

test('expiring a case invalidates an outstanding analysis and cancels its polling', async () => {
  const h = harness(); await h.load();
  const analyzing = h.api.analyzeCase();
  const rendering = h.pending.shift(), analysis = h.pending.shift();
  rendering.resolve(json({ error: { code: 'case_expired', message: 'This scan is no longer loaded. Please load it again.' } }, 404));
  await new Promise(resolve => setImmediate(resolve));
  assertNoImage(h);
  assert.equal(h.timers.size, 0);
  analysis.resolve(json(metadata('case-a'))); await analyzing;
  assert.equal(h.node('case-id').textContent, '—');
  assert.equal(h.node('dice-value').textContent, '—');
  assert.equal(h.pending.length, 0);
});

test('a successful image retry preserves a newer analysis failure message', async () => {
  const h = harness(); await h.load();
  const failed = h.api.requestSlice(91);
  h.pending.shift().reject(new Error('Image request failed.')); await failed;
  const analyzing = h.api.analyzeCase();
  const rendering = h.pending.shift(), analysis = h.pending.shift();
  analysis.resolve(json({ error: { message: 'Analysis service unavailable.' } }, 503));
  await analyzing;
  rendering.resolve(imageResponse());
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.node('ct-image').hidden, false);
  assert.equal(h.node('progress').textContent, 'Analysis service unavailable.');
  assert.equal(h.node('status-dot').classList.contains('error'), true);
});
