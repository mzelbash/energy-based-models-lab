// Browser side MNIST loader. Reads the packed sprite + metadata, draws it to an
// offscreen canvas, and exposes the raw digits plus a preprocessing routine that
// matches the book: scale pixels to [-1, 1] and pad from 28x28 to 32x32.
// No long dashes are used anywhere in this project.

const tf = window.tf;
let cache = null;

export const IMG_SIZE = 32;
export const RAW_SIZE = 28;

// Load once and cache. Returns { meta, gray: Uint8ClampedArray of count*28*28 }.
export async function loadData() {
  if (cache) return cache;
  const meta = await fetch('data/mnist_subset.json').then((r) => r.json());
  const img = await loadImage('data/mnist_subset.png');

  const c = document.createElement('canvas');
  c.width = img.width;
  c.height = img.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);

  const S = meta.imageSize;
  const gray = new Uint8ClampedArray(meta.count * S * S);
  for (let i = 0; i < meta.count; i++) {
    const cr = Math.floor(i / meta.cols);
    const cc = i % meta.cols;
    const cell = ctx.getImageData(cc * S, cr * S, S, S).data;
    for (let p = 0; p < S * S; p++) gray[i * S * S + p] = cell[p * 4]; // red channel
  }
  cache = { meta, gray, S };
  return cache;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = src;
  });
}

// Get the raw 28x28 uint8 pixels for one digit (for the pixel inspector).
export function rawDigit(i) {
  const { gray, S } = cache;
  return { pixels: gray.subarray(i * S * S, (i + 1) * S * S), size: S };
}

// Build a preprocessed [n,32,32,1] tensor in [-1,1] for the given indices.
export function batchTensor(indices) {
  const { gray, S } = cache;
  const n = indices.length;
  const flat = new Float32Array(n * S * S);
  for (let k = 0; k < n; k++) {
    const base = indices[k] * S * S;
    for (let p = 0; p < S * S; p++) flat[k * S * S + p] = gray[base + p] / 255;
  }
  return tf.tidy(() => {
    let t = tf.tensor4d(flat, [n, S, S, 1]);
    t = t.mul(2).sub(1); // [0,1] -> [-1,1]
    t = t.pad([[0, 0], [2, 2], [2, 2], [0, 0]], -1); // 28 -> 32
    return t;
  });
}

// Indices of the training and test splits.
export function splitIndices() {
  const { meta } = cache;
  const [trS, trE] = meta.splits.train;
  const [teS, teE] = meta.splits.test;
  const train = [];
  const test = [];
  for (let i = trS; i < trE; i++) train.push(i);
  for (let i = teS; i < teE; i++) test.push(i);
  return { train, test, labels: meta.labels };
}
