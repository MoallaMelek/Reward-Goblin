import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = (await readFile(new URL('../../frontend/replay/player.js', import.meta.url), 'utf8'))
  .replace(/^import .*;$/gm, '');
const { Player } = await import(`data:text/javascript,${encodeURIComponent(source)}`);

test('play is idempotent while a playback loop is active', () => {
  let scheduled = 0;
  globalThis.requestAnimationFrame = () => ++scheduled;
  const player = Object.assign(Object.create(Player.prototype), {
    rec: {}, t: 0, n: 10, playing: false,
  });
  player.play();
  player.play();
  assert.equal(scheduled, 1);
});

test('loading another replay stops playback and clears fractional progress', () => {
  let cancelled = null;
  globalThis.cancelAnimationFrame = (id) => { cancelled = id; };
  const player = Object.assign(Object.create(Player.prototype), {
    playing: true, _raf: 7, _acc: .9,
    left: {setLayout() {}}, right: {setLayout() {}},
    render() {}, _drawTimeline() {},
  });
  player.load({frames: {agent: [[0, 0]]}, layout: {}});
  assert.equal(player.playing, false);
  assert.equal(cancelled, 7);
  assert.equal(player._acc, 0);
});
