import test from 'node:test';
import assert from 'node:assert/strict';
import { createRaceGuards } from '../nucleo_viewer/static/state.mjs';

function deferred() { let resolve, reject; const promise = new Promise((res, rej) => { resolve = res; reject = rej; }); return { promise, resolve, reject }; }

async function settleSelection(guards, ticket, request, commits, errors) {
  try { const value = await request; if (guards.isCurrentSelection(ticket)) commits.push(value); }
  catch (error) { if (guards.isCurrentSelection(ticket)) errors.push(error.message); }
}

test('late A success cannot replace newer B selection', async () => {
  const guards = createRaceGuards(), a = deferred(), b = deferred(), commits = [], errors = [];
  const aTicket = guards.beginSelection(); const pa = settleSelection(guards, aTicket, a.promise, commits, errors);
  const bTicket = guards.beginSelection(); const pb = settleSelection(guards, bTicket, b.promise, commits, errors);
  b.resolve('B'); await pb; a.resolve('A'); await pa;
  assert.deepEqual(commits, ['B']); assert.deepEqual(errors, []);
});

test('late A failure cannot show an error after B succeeds', async () => {
  const guards = createRaceGuards(), a = deferred(), b = deferred(), commits = [], errors = [];
  const aTicket = guards.beginSelection(); const pa = settleSelection(guards, aTicket, a.promise, commits, errors);
  const bTicket = guards.beginSelection(); const pb = settleSelection(guards, bTicket, b.promise, commits, errors);
  b.resolve('B'); await pb; a.reject(new Error('A failed')); await pa;
  assert.deepEqual(commits, ['B']); assert.deepEqual(errors, []);
});

test('latest selection failure is shown when it is still current', async () => {
  const guards = createRaceGuards(), request = deferred(), commits = [], errors = [];
  const ticket = guards.beginSelection(); const pending = settleSelection(guards, ticket, request.promise, commits, errors);
  request.reject(new Error('bad ZIP')); await pending;
  assert.deepEqual(commits, []); assert.deepEqual(errors, ['bad ZIP']);
});

test('slow earlier slice cannot replace newer slice image', async () => {
  const guards = createRaceGuards(), first = deferred(), second = deferred(), committedImages = [];
  const selection = guards.beginSelection(); const firstTicket = guards.beginImage(selection); const secondTicket = guards.beginImage(selection);
  const commit = async (ticket, pending, value) => { await pending; if (guards.isCurrentImage(ticket)) committedImages.push(value); };
  const p1 = commit(firstTicket, first.promise, 'slice 11'); const p2 = commit(secondTicket, second.promise, 'slice 12');
  second.resolve(); await p2; first.resolve(); await p1;
  assert.deepEqual(committedImages, ['slice 12']);
});

test('starting a new selection invalidates outstanding images', async () => {
  const guards = createRaceGuards(); const oldSelection = guards.beginSelection(); const imageTicket = guards.beginImage(oldSelection);
  guards.beginSelection();
  assert.equal(guards.isCurrentImage(imageTicket), false);
});
