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
