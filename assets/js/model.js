// Browser side model helpers: build the energy network and the classifier from the
// shared definitions, and load the weights the Keras build script exported. The weight
// order matches tfjs getWeights, so setWeights loads them directly.
// No long dashes are used anywhere in this project.

import { buildEnergyModel, buildClassifier } from './model-defs.js';

const tf = window.tf;

// Load a weights.bin + weights.json checkpoint into a freshly built model.
async function loadWeightsInto(model, dir) {
  const meta = await fetch(`${dir}/weights.json`).then((r) => r.json());
  const buf = await fetch(`${dir}/weights.bin`).then((r) => r.arrayBuffer());
  const floats = new Float32Array(buf);

  const tensors = [];
  let offset = 0;
  for (const shape of meta.shapes) {
    const n = shape.reduce((a, b) => a * b, 1);
    const slice = floats.subarray(offset, offset + n);
    tensors.push(tf.tensor(slice, shape));
    offset += n;
  }
  model.setWeights(tensors);
  tensors.forEach((t) => t.dispose());
  return model;
}

export async function loadEnergyCheckpoint() {
  const model = buildEnergyModel(tf);
  await loadWeightsInto(model, 'data/checkpoints/ebm');
  return model;
}

export async function loadClassifierCheckpoint() {
  const model = buildClassifier(tf);
  await loadWeightsInto(model, 'data/checkpoints/clf');
  return model;
}

// Build a fresh, untrained energy model for live training from scratch.
export function freshEnergyModel() {
  return buildEnergyModel(tf);
}

// Score a batch and return the mean energy E(x) = -score as a plain number.
// A LOW energy means the model finds the input realistic.
export function meanEnergy(model, batch) {
  return tf.tidy(() => model.predict(batch).mul(-1).mean().dataSync()[0]);
}
