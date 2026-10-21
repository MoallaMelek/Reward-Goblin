import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

globalThis.window = {};
const source = await readFile(new URL('../../frontend/components/api.js', import.meta.url), 'utf8');
const { api } = await import(`data:text/javascript,${encodeURIComponent(source)}`);

test('HTTP failures with plain text preserve status and server detail', async () => {
  globalThis.fetch = async () => ({ok: false, status: 503, statusText: 'Service Unavailable', text: async () => 'Backend restarting'});
  await assert.rejects(api.meta(), /503.*Backend restarting/);
});

test('structured FastAPI error details remain readable', async () => {
  globalThis.fetch = async () => ({ok: false, status: 422, statusText: 'Unprocessable Entity', text: async () => JSON.stringify({detail: 'invalid seed'})});
  await assert.rejects(api.meta(), /invalid seed/);
});

test('successful JSON responses are returned unchanged', async () => {
  globalThis.fetch = async () => ({ok: true, text: async () => '{"limits":{"max_seeds":8}}'});
  assert.deepEqual(await api.meta(), {limits: {max_seeds: 8}});
});

test('JSON and empty HTTP errors retain their status code', async () => {
  for (const body of ['{"detail":"invalid seed"}', '']) {
    globalThis.fetch = async () => ({ok: false, status: 422, statusText: 'Unprocessable Entity', text: async () => body});
    await assert.rejects(api.meta(), /HTTP 422:/);
  }
});


test('static demo never claims uncomputed rewards passed validation', async () => {
  globalThis.window = {REWARD_GOBLIN_STATIC: true};
  globalThis.fetch = async () => ({ok: true, text: async () => '{"known":{"ok":true}}'});
  const {api: snapshot} = await import(`data:text/javascript,${encodeURIComponent(source + '\n// static')}`);
  assert.equal((await snapshot.validate({}, 'known')).ok, true);
  for (const key of [null, 'changed', '__proto__', 'constructor']) {
    const result = await snapshot.validate({}, key);
    assert.equal(result.ok, false);
    assert.equal(result.offline, true);
  }
});
