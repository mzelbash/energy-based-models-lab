// Small rendering helpers: draw a [-1,1] image tensor to a canvas, build grids of
// samples, and build labelled snapshot strips. Kept dependency free.
// No long dashes are used anywhere in this project.

const tf = window.tf;

// Draw a single HxW image (values in [-1,1]) to a canvas at a given display scale.
export function drawImage(canvas, data, w, h, scale = 4) {
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = w * scale + 'px';
  canvas.style.height = h * scale + 'px';
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    // map [-1,1] to [0,255], invert so ink is dark on light (MNIST convention here)
    let v = (data[i] + 1) * 0.5; // [0,1], 1 = white stroke on black background
    v = Math.max(0, Math.min(1, v));
    const g = Math.round(v * 255);
    img.data[i * 4] = g;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = g;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// Draw one image tensor of shape [1,H,W,1] or [H,W,1].
export function drawTensor(canvas, tensor, scale = 4) {
  const t = tensor.rank === 4 ? tensor.squeeze([0]) : tensor;
  const [h, w] = t.shape;
  const data = t.dataSync();
  drawImage(canvas, data, w, h, scale);
  if (t !== tensor) t.dispose();
}

// Render a batch tensor [n,H,W,1] into a wrapping grid of small canvases.
export function drawGrid(container, batch, { cols = 8, scale = 3, max = 64 } = {}) {
  container.innerHTML = '';
  const n = Math.min(batch.shape[0], max);
  const data = batch.dataSync();
  const [, h, w] = batch.shape;
  const stride = h * w;
  container.style.display = 'grid';
  container.style.gridTemplateColumns = `repeat(${cols}, max-content)`;
  container.style.gap = '4px';
  for (let i = 0; i < n; i++) {
    const cv = document.createElement('canvas');
    drawImage(cv, data.subarray(i * stride, (i + 1) * stride), w, h, scale);
    cv.style.borderRadius = '4px';
    container.appendChild(cv);
  }
}

// Build a snapshot strip from a Map(step -> tensor[1,H,W,1]).
export function drawSnapStrip(container, snapMap, scale = 3) {
  container.innerHTML = '';
  const steps = [...snapMap.keys()].sort((a, b) => a - b);
  for (const s of steps) {
    const wrap = document.createElement('div');
    wrap.className = 'snap';
    const cv = document.createElement('canvas');
    drawTensor(cv, snapMap.get(s), scale);
    cv.style.borderRadius = '4px';
    cv.style.border = '1px solid var(--line)';
    const lab = document.createElement('div');
    lab.className = 'step';
    lab.textContent = 'step ' + s;
    wrap.appendChild(cv);
    wrap.appendChild(lab);
    container.appendChild(wrap);
  }
}

export function makeNoise(n = 1) {
  return tf.randomUniform([n, 32, 32, 1], -1, 1);
}
