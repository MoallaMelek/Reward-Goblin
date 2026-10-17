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
