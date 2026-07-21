// The energy based model core: Langevin sampler, replay buffer, and the contrastive
// divergence training step. Shared by the browser app and the Node build script, so
// the tf module is always passed in.
//
// Sign convention note (important, and matches the book):
// the network outputs a REALNESS score s(x). Real images should score HIGH, generated
// images should score LOW. The energy is E(x) = -s(x), so low energy means realistic.
// The Langevin sampler climbs the score (descends the energy) to turn noise into digits.
//
// No long dashes are used anywhere in this project.

import { IMG_SIZE } from './model-defs.js';

// ---------------------------------------------------------------------------
// Langevin dynamics sampler (Example 7-4).
// Starting from inpImgs, take `steps` noisy gradient steps that increase the model
// score, gradually turning noise into a plausible observation. Memory is managed
// carefully: one tensor is kept alive across the loop, everything else is tidied,
// so long runs (hundreds of steps) do not leak.
//
// opts.snapshotAt: optional array of step indices to capture (returns clones).
// Returns { final } or { final, snapshots: Map(step -> tensor) }.
// The caller owns every returned tensor and must dispose them.
// ---------------------------------------------------------------------------
export function generateSamples(tf, model, inpImgs, steps, stepSize, noise, opts = {}) {
  const snapAt = opts.snapshotAt ? new Set(opts.snapshotAt) : null;
  const snapshots = snapAt ? new Map() : null;

  let imgs = tf.keep(inpImgs.clone());
  if (snapAt && snapAt.has(0)) snapshots.set(0, tf.keep(imgs.clone()));

  for (let i = 0; i < steps; i++) {
    const next = tf.tidy(() => {
      // add a little noise, then keep pixels in range
      let x = imgs.add(tf.randomNormal(imgs.shape, 0, noise));
      x = x.clipByValue(-1, 1);
      // gradient of the (negated) score with respect to the input image
      const gradFn = tf.grad((xx) => model.predict(xx).mul(-1).sum());
      let g = gradFn(x);
      g = g.clipByValue(-0.03, 0.03); // stops any single step from exploding
      x = x.add(g.mul(-stepSize));    // move to raise the score (lower the energy)
      x = x.clipByValue(-1, 1);
      return x;
    });
    imgs.dispose();
    imgs = tf.keep(next);
    if (snapAt && snapAt.has(i + 1)) snapshots.set(i + 1, tf.keep(imgs.clone()));
  }

  return snapshots ? { final: imgs, snapshots } : { final: imgs };
}

// ---------------------------------------------------------------------------
// Replay buffer (Example 7-5).
// Keeps a pool of previously generated samples. Each draw is mostly pulled from the
// pool with a small fraction (5 percent) freshly seeded from noise, then all of them
// are pushed through the Langevin sampler. This makes fake samples steadily better
// across training without restarting from pure noise every time.
// ---------------------------------------------------------------------------
export function createBuffer(tf, model, { size = 128, maxLen = 8192 } = {}) {
  const S = IMG_SIZE;
  // start the pool with `size` random noise images in [-1, 1]
  let examples = [];
  tf.tidy(() => {
    const init = tf.randomUniform([size, S, S, 1], -1, 1);
    const parts = tf.split(init, size, 0);
    parts.forEach((p) => examples.push(tf.keep(p)));
  });

  function sampleNewExamples(steps, stepSize, noise) {
    const nNew = Math.max(1, Math.round(size * 0.05)); // roughly 5 percent fresh
    const nOld = size - nNew;

    const start = tf.tidy(() => {
      const rand = tf.randomUniform([nNew, S, S, 1], -1, 1);
      // pull nOld random items from the pool
      const idx = [];
      for (let i = 0; i < nOld; i++) idx.push(Math.floor(Math.random() * examples.length));
      const old = tf.concat(idx.map((i) => examples[i]), 0);
      return tf.concat([rand, old], 0);
    });

    const { final } = generateSamples(tf, model, start, steps, stepSize, noise);
    start.dispose();

    // split the fresh batch back into single images and prepend to the pool
    const newParts = tf.tidy(() => tf.split(final, size, 0).map((p) => tf.keep(p)));
    examples = newParts.concat(examples);
    // trim the pool and dispose anything that falls off the end
    if (examples.length > maxLen) {
      examples.slice(maxLen).forEach((t) => t.dispose());
      examples = examples.slice(0, maxLen);
    }
    return final; // caller owns this batch tensor
  }

  function currentSamples(n) {
    return tf.tidy(() => tf.concat(examples.slice(0, n), 0).clone());
  }

  function dispose() {
    examples.forEach((t) => t.dispose());
    examples = [];
  }

  return { sampleNewExamples, currentSamples, dispose, get length() { return examples.length; } };
}

// ---------------------------------------------------------------------------
// One contrastive divergence training step (Example 7-6).
// Real images are nudged with a touch of noise, fake images are drawn from the buffer,
// both are scored, and the loss pushes real scores up and fake scores down. A small
// regularizer keeps the raw scores from drifting too far from zero.
// Returns plain-number metrics for display. Manages its own tensors.
// ---------------------------------------------------------------------------
export function cdTrainStep(tf, model, buffer, optimizer, realBatch, cfg) {
  const { alpha = 0.1, cdSteps = 60, stepSize = 10, noise = 0.005 } = cfg;

  const realImgs = tf.tidy(() =>
    realBatch.add(tf.randomNormal(realBatch.shape, 0, noise)).clipByValue(-1, 1)
  );
  const fakeImgs = buffer.sampleNewExamples(cdSteps, stepSize, noise);
  const inp = tf.tidy(() => tf.concat([realImgs, fakeImgs], 0));

  let metrics;
  const lossFn = () =>
    tf.tidy(() => {
      const out = model.predict(inp);
      const [realOut, fakeOut] = tf.split(out, 2, 0);
      const cdiv = fakeOut.mean().sub(realOut.mean());
      const reg = realOut.square().add(fakeOut.square()).mean().mul(alpha);
      return reg.add(cdiv);
    });

  const { value, grads } = tf.variableGrads(lossFn);
  optimizer.applyGradients(grads);

  metrics = tf.tidy(() => {
    const out = model.predict(inp);
    const [realOut, fakeOut] = tf.split(out, 2, 0);
    const real = realOut.mean();
    const fake = fakeOut.mean();
    const cdiv = fake.sub(real);
    const reg = realOut.square().add(fakeOut.square()).mean().mul(alpha);
    return {
      loss: value.dataSync()[0],
      cdiv: cdiv.dataSync()[0],
      reg: reg.dataSync()[0],
      real: real.dataSync()[0],
      fake: fake.dataSync()[0],
    };
  });

  value.dispose();
  Object.values(grads).forEach((g) => g.dispose());
  realImgs.dispose();
  fakeImgs.dispose();
  inp.dispose();

  return metrics;
}
