// App orchestration: tabs, lazy per tab initialization, and every interactive demo.
// Loads the shared TensorFlow.js core and wires it to the DOM defined in index.html.
// No long dashes are used anywhere in this project.

import { initThemes } from './themes.js';
import * as data from './data.js';
import { LISTINGS, renderListing, renderSource } from './code.js';
import * as viz from './viz.js';
import { LineChart, drawBars } from './charts.js';
import {
  loadEnergyCheckpoint, loadClassifierCheckpoint, freshEnergyModel, meanEnergy,
} from './model.js';
import { generateSamples, createBuffer, cdTrainStep } from './ebm-core.js';
import { RBM, visibleToImage } from './rbm.js';

const tf = window.tf;

const TABS = [
  { id: 'start', label: 'Start here' },
  { id: 'intuition', label: 'Intuition' },
  { id: 'data', label: 'The data' },
  { id: 'energy', label: 'Energy function' },
  { id: 'langevin', label: 'Langevin sampling' },
  { id: 'training', label: 'Training' },
  { id: 'analysis', label: 'Generate & analyze' },
  { id: 'other', label: 'Other EBMs' },
  { id: 'jem', label: 'Classifier as EBM' },
  { id: 'appendix', label: 'Code appendix' },
];

const state = {
  ebm: null,
  ebmReady: false,
  clf: null,
  clfReady: false,
  dataReady: false,
  inited: {},
  split: null,
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  initThemes();
  buildTabs();
  fillListings();
  await tf.ready();

  // load data and checkpoints in parallel, but do not block the UI shell
  loadEverything();

  // deep link support and initial tab
  const hash = location.hash.replace('#', '');
  showTab(TABS.some((t) => t.id === hash) ? hash : 'start');

  document.querySelectorAll('[data-goto]').forEach((b) =>
    b.addEventListener('click', () => showTab(b.dataset.goto))
  );
}

async function loadEverything() {
  const status = document.getElementById('model-status');
  try {
    await data.loadData();
    state.split = data.splitIndices();
    state.dataReady = true;
  } catch (e) {
    console.error(e);
    setStatus('data failed, run a local server', false);
    return;
  }

  // Checkpoints may still be training. Try to load, retry a few times.
  loadCheckpointsWithRetry();
  // re init the currently visible tab now that data exists
  reinit(currentTab);
}

async function loadCheckpointsWithRetry(attempt = 0) {
  try {
    if (!state.ebmReady) {
      state.ebm = await loadEnergyCheckpoint();
      state.ebmReady = true;
    }
    if (!state.clfReady) {
      state.clf = await loadClassifierCheckpoint();
      state.clfReady = true;
    }
    setStatus('checkpoint loaded', true);
    reinit(currentTab);
  } catch (e) {
    if (attempt < 20) {
      setStatus('waiting for checkpoint', 'busy');
      setTimeout(() => loadCheckpointsWithRetry(attempt + 1), 3000);
    } else {
      setStatus('checkpoint not found', false);
      console.warn('checkpoint load failed', e);
    }
  }
}

function setStatus(text, on) {
  const el = document.getElementById('model-status');
  el.querySelector('.txt').textContent = text;
  el.className = 'pill' + (on === true ? ' on' : on === 'busy' ? ' busy' : '');
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
let currentTab = 'start';

function buildTabs() {
  const bar = document.getElementById('tabbar');
  TABS.forEach((t, i) => {
    const b = document.createElement('button');
    b.className = 'tab-btn';
    b.setAttribute('role', 'tab');
    b.dataset.tab = t.id;
    b.innerHTML = `<span class="tab-btn__num">${i + 1}</span>${t.label}`;
    b.addEventListener('click', () => showTab(t.id));
    bar.appendChild(b);
  });
}

function showTab(id) {
  currentTab = id;
  location.hash = id;
  TABS.forEach((t) => {
    const panel = document.getElementById('tab-' + t.id);
    if (panel) panel.hidden = t.id !== id;
  });
  document.querySelectorAll('.tab-btn').forEach((b) =>
    b.setAttribute('aria-selected', String(b.dataset.tab === id))
  );
  window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  reinit(id);
}

// lazy init, safe to call repeatedly
function reinit(id) {
  const fn = INIT[id];
  if (fn) fn();
}

// ---------------------------------------------------------------------------
// Code listings
// ---------------------------------------------------------------------------
function fillListings() {
  document.querySelectorAll('[data-listing]').forEach((el) => {
    const key = el.dataset.listing;
    if (LISTINGS[key]) el.appendChild(renderListing(LISTINGS[key]));
  });
  const appx = document.getElementById('appendix-all');
  if (appx) {
    ['load', 'preprocess', 'energy', 'langevin', 'buffer', 'train', 'generate', 'jem', 'boltzmann'].forEach((k) => {
      appx.appendChild(renderListing(LISTINGS[k]));
      appx.appendChild(document.createElement('hr')).className = 'soft';
    });
  }
  fillAppendixSources();
}

// Load the real source files into the appendix so they never drift from the code.
async function fillAppendixSources() {
  const load = async (path, id, lineComment) => {
    const el = document.getElementById(id);
    if (!el) return;
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(String(res.status));
      renderSource(el, await res.text(), lineComment);
    } catch (e) {
      el.innerHTML = `<p class="small muted">Could not load <span class="kbd">${path}</span> here. You can read it in the repository.</p>`;
    }
  };
  load('build/train_checkpoints.py', 'appendix-pretrain', '#');
  load('assets/js/ebm-core.js', 'appendix-scratch', '//');
}

// ===========================================================================
// TAB: intuition, Boltzmann distribution
// ===========================================================================
function initIntuition() {
  if (state.inited.intuition) return;
  state.inited.intuition = true;
  const controls = document.getElementById('boltz-controls');
  const canvas = document.getElementById('boltz-canvas');
  const note = document.getElementById('boltz-note');
  const energies = [0.5, 1.5, -0.5, 2.0];
  const names = ['A', 'B', 'C', 'D'];

  function render() {
    const probs = softmax(energies.map((e) => -e));
    drawBars(canvas, probs, names, { title: 'Probability p(x) from energy', highlight: probs.indexOf(Math.max(...probs)) });
    const best = names[probs.indexOf(Math.max(...probs))];
    note.innerHTML = `Lowest energy is candidate <b>${best}</b>, so it gets the highest probability. Shift any energy and the whole distribution rebalances, because probabilities must sum to 1.`;
  }
  controls.innerHTML = '';
  energies.forEach((e, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'control';
    wrap.innerHTML = `<label>Energy of candidate ${names[i]} <b><span id="be-${i}">${e.toFixed(1)}</span></b></label>`;
    const input = document.createElement('input');
    input.type = 'range'; input.min = '-3'; input.max = '3'; input.step = '0.1'; input.value = String(e);
    input.addEventListener('input', () => {
      energies[i] = parseFloat(input.value);
      document.getElementById('be-' + i).textContent = energies[i].toFixed(1);
      render();
    });
    wrap.appendChild(input);
    controls.appendChild(wrap);
  });
  render();
  window.addEventListener('themechange', render);
}

function softmax(arr) {
  const m = Math.max(...arr);
  const ex = arr.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

// ===========================================================================
// TAB: data
// ===========================================================================
function initData() {
  if (!state.dataReady) return;
  if (state.inited.data) return;
  state.inited.data = true;

  const grid = document.getElementById('data-grid');
  function showSample() {
    const idx = pickRandom(state.split.train, 40);
    const batch = data.batchTensor(idx);
    viz.drawGrid(grid, batch, { cols: 10, scale: 2, max: 40 });
    batch.dispose();
  }
  document.getElementById('data-reshuffle').onclick = showSample;
  showSample();

  // inspector
  let curIdx = state.split.train[0];
  let curData = null;
  const inspect = document.getElementById('data-inspect');
  function showInspect() {
    const batch = data.batchTensor([curIdx]);
    curData = batch.dataSync().slice();
    viz.drawTensor(inspect, batch, 8);
    batch.dispose();
  }
  document.getElementById('data-next').onclick = () => {
    curIdx = pickRandom(state.split.train, 1)[0];
    showInspect();
  };
  inspect.addEventListener('mousemove', (ev) => {
    if (!curData) return;
    const r = inspect.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - r.left) / r.width) * 32);
    const y = Math.floor(((ev.clientY - r.top) / r.height) * 32);
    if (x < 0 || y < 0 || x > 31 || y > 31) return;
    const v = curData[y * 32 + x];
    document.getElementById('px-val').textContent = v.toFixed(2);
    document.getElementById('px-hint').textContent = `pixel (row ${y}, col ${x})`;
  });
  showInspect();
}

// ===========================================================================
// TAB: energy function
// ===========================================================================
function initEnergy() {
  if (state.inited.energy) return;
  state.inited.energy = true;

  // architecture diagram
  const layersInfo = [
    ['Input', '32 x 32 x 1'],
    ['Conv2D 16, stride 2', '16 x 16 x 16'],
    ['Conv2D 32, stride 2', '8 x 8 x 32'],
    ['Conv2D 64, stride 2', '4 x 4 x 64'],
    ['Conv2D 64, stride 2', '2 x 2 x 64'],
    ['Flatten', '256'],
    ['Dense 64 (swish)', '64'],
    ['Dense 1 (linear)', '1  = the score'],
  ];
  const arch = document.getElementById('arch-diagram');
  arch.innerHTML = '';
  layersInfo.forEach(([name, shape], i) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;justify-content:space-between;gap:10px;padding:8px 12px;border:1px solid var(--line);border-radius:8px;margin:6px 0;background:var(--bg-sunken)';
    const w = 100 - i * 9;
    row.innerHTML = `<span style="font-weight:600;font-size:0.86rem">${name}</span><span class="small muted" style="font-variant-numeric:tabular-nums">${shape}</span>`;
    row.style.marginLeft = (i * 6) + 'px';
    row.style.borderLeft = '3px solid var(--brand)';
    arch.appendChild(row);
  });

  drawSwish();
  window.addEventListener('themechange', drawSwish);

  document.getElementById('energy-eval').onclick = scoreExamples;
  scoreExamples();
}

function drawSwish() {
  const cv = document.getElementById('swish-canvas');
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 320, h = 180;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.height = h + 'px';
  const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const line = css('--line'), faint = css('--ink-faint'), brand = css('--brand'), accent = css('--accent');
  const x0 = 30, y0 = h - 24, plotW = w - 45, plotH = h - 40;
  const xmin = -6, xmax = 6, ymin = -1, ymax = 6;
  const X = (x) => x0 + ((x - xmin) / (xmax - xmin)) * plotW;
  const Y = (y) => (h - 16) - ((y - ymin) / (ymax - ymin)) * plotH;
  ctx.strokeStyle = line; ctx.beginPath(); ctx.moveTo(X(xmin), Y(0)); ctx.lineTo(X(xmax), Y(0)); ctx.moveTo(X(0), Y(ymin)); ctx.lineTo(X(0), Y(ymax)); ctx.stroke();
  const plot = (fn, color) => {
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const x = xmin + (px / plotW) * (xmax - xmin); const y = fn(x);
      px === 0 ? ctx.moveTo(X(x), Y(y)) : ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
  };
  plot((x) => Math.max(0, x), faint);         // ReLU for reference
  plot((x) => x / (1 + Math.exp(-x)), brand);  // swish
  ctx.fillStyle = brand; ctx.font = '600 11px system-ui'; ctx.fillText('swish', X(3.2), Y(4.2));
  ctx.fillStyle = faint; ctx.fillText('ReLU', X(3.2), Y(2.4));
}

function scoreExamples() {
  if (!state.ebmReady) { document.getElementById('energy-verdict').textContent = 'Waiting for the trained checkpoint to load.'; return; }
  // Average over a batch so a single unlucky sample cannot mislead.
  const n = 16;
  const idx = pickRandom(state.split.train, n);
  const real = data.batchTensor(idx);
  const noise = viz.makeNoise(n);
  const real1 = tf.tidy(() => real.slice([0, 0, 0, 0], [1, -1, -1, -1]));
  const noise1 = tf.tidy(() => noise.slice([0, 0, 0, 0], [1, -1, -1, -1]));
  viz.drawTensor(document.getElementById('energy-real'), real1, 5);
  viz.drawTensor(document.getElementById('energy-noise'), noise1, 5);
  const eReal = meanEnergy(state.ebm, real);
  const eNoise = meanEnergy(state.ebm, noise);
  document.getElementById('e-real').textContent = eReal.toFixed(2);
  document.getElementById('e-noise').textContent = eNoise.toFixed(2);
  document.getElementById('energy-verdict').textContent =
    eReal < eNoise ? `Correct: real digits average lower energy (${eReal.toFixed(2)} vs ${eNoise.toFixed(2)}), averaged over ${n} samples of each.`
                   : 'This model is still early, energies are close.';
  real.dispose(); noise.dispose(); real1.dispose(); noise1.dispose();
}

// ===========================================================================
// TAB: langevin sampler (animated)
// ===========================================================================
let langLoop = { running: false };
function initLangevin() {
  if (state.inited.langevin) return;
  state.inited.langevin = true;
  bindSlider('lang-ss', 'lang-ss-v');
  bindSlider('lang-noise', 'lang-noise-v', (v) => (v / 1000).toFixed(3));
  bindSlider('lang-steps', 'lang-steps-v');
  document.getElementById('lang-run').onclick = runLangevin;
  document.getElementById('lang-stop').onclick = () => (langLoop.running = false);
}

async function runLangevin() {
  if (!state.ebmReady) { document.getElementById('lang-step').textContent = 'checkpoint still loading'; return; }
  if (langLoop.running) return;
  langLoop.running = true;
  document.getElementById('lang-run').disabled = true;
  document.getElementById('lang-stop').disabled = false;

  const stepSize = +document.getElementById('lang-ss').value;
  const noise = +document.getElementById('lang-noise').value / 1000;
  const steps = +document.getElementById('lang-steps').value;
  const snapAt = new Set([0, 1, 3, 5, 10, 30, 50, 100, 200, 300, 500, steps].filter((s) => s <= steps));
  const snaps = new Map();

  let img = tf.keep(viz.makeNoise(1));
  const cv = document.getElementById('lang-canvas');
  const canvasScale = 7;

  for (let i = 0; i <= steps; i++) {
    if (!langLoop.running) break;
    if (i > 0) {
      const next = tf.tidy(() => {
        let x = img.add(tf.randomNormal(img.shape, 0, noise)).clipByValue(-1, 1);
        const g = tf.grad((xx) => state.ebm.predict(xx).mul(-1).sum())(x).clipByValue(-0.03, 0.03);
        return x.add(g.mul(-stepSize)).clipByValue(-1, 1);
      });
      img.dispose(); img = tf.keep(next);
    }
    if (snapAt.has(i)) snaps.set(i, tf.keep(img.clone()));
    if (i % 2 === 0 || i === steps) {
      viz.drawTensor(cv, img, canvasScale);
      document.getElementById('lang-step').textContent = 'step ' + i;
      document.getElementById('lang-energy').textContent = meanEnergy(state.ebm, img).toFixed(2);
      document.getElementById('lang-mem').textContent = tf.memory().numTensors;
      await tf.nextFrame();
    }
  }

  viz.drawSnapStrip(document.getElementById('lang-snaps'), snaps, 3);
  snaps.forEach((t) => t.dispose());
  img.dispose();
  langLoop.running = false;
  document.getElementById('lang-run').disabled = false;
  document.getElementById('lang-stop').disabled = true;
}

// ===========================================================================
// TAB: training (live contrastive divergence)
// ===========================================================================
let trLoop = { running: false };
let trCharts = null;
function initTraining() {
  if (state.inited.training) return;
  state.inited.training = true;
  bindSlider('tr-steps', 'tr-steps-v');
  bindSlider('tr-cd', 'tr-cd-v');
  trCharts = {
    cdiv: new LineChart(document.getElementById('tr-chart-cdiv'), { title: 'Contrast (fake minus real)' }),
    real: new LineChart(document.getElementById('tr-chart-real'), { title: 'Real energy' }),
    fake: new LineChart(document.getElementById('tr-chart-fake'), { title: 'Fake energy' }),
  };
  document.getElementById('tr-run').onclick = runTraining;
  document.getElementById('tr-stop').onclick = () => (trLoop.running = false);
}

async function runTraining() {
  if (!state.dataReady) return;
  if (trLoop.running) return;
  trLoop.running = true;
  setTrStatus('training', 'busy');
  const runBtn = document.getElementById('tr-run');
  const prog = document.getElementById('tr-progress');
  const progFill = document.getElementById('tr-progress-fill');
  const progLabel = document.getElementById('tr-progress-label');
  runBtn.disabled = true;
  runBtn.textContent = 'Training ...';
  document.getElementById('tr-stop').disabled = false;
  Object.values(trCharts).forEach((c) => c.reset());

  const totalSteps = +document.getElementById('tr-steps').value;
  const cdSteps = +document.getElementById('tr-cd').value;

  // reveal the progress bar right away so the click has immediate feedback
  prog.hidden = false;
  progFill.style.width = '0%';
  progLabel.textContent = `starting, 0 of ${totalSteps} steps`;
  await tf.nextFrame();

  const model = freshEnergyModel();
  const optimizer = tf.train.adam(1e-4);
  const buffer = createBuffer(tf, model, { size: 64, maxLen: 2048 });
  const cfg = { alpha: 0.1, cdSteps, stepSize: 10, noise: 0.005 };

  const trainIdx = state.split.train;
  let done = 0;
  for (let s = 0; s < totalSteps; s++) {
    if (!trLoop.running) break;
    const idx = pickRandom(trainIdx, 64);
    const realBatch = data.batchTensor(idx);
    const m = cdTrainStep(tf, model, buffer, optimizer, realBatch, cfg);
    realBatch.dispose();
    done = s + 1;

    const eReal = -m.real, eFake = -m.fake; // energy = negative score
    document.getElementById('tr-real').textContent = eReal.toFixed(2);
    document.getElementById('tr-fake').textContent = eFake.toFixed(2);
    document.getElementById('tr-cdiv').textContent = (eFake - eReal).toFixed(2);
    trCharts.cdiv.push(eFake - eReal);
    trCharts.real.push(eReal);
    trCharts.fake.push(eFake);

    progFill.style.width = ((done / totalSteps) * 100).toFixed(1) + '%';
    progLabel.textContent = `training, step ${done} of ${totalSteps}`;

    if (s % 5 === 0 || s === totalSteps - 1) {
      const fakes = buffer.currentSamples(24);
      viz.drawGrid(document.getElementById('tr-buffer'), fakes, { cols: 12, scale: 2, max: 24 });
      fakes.dispose();
      await tf.nextFrame();
    }
  }

  const completed = done >= totalSteps;
  progFill.style.width = ((done / totalSteps) * 100).toFixed(1) + '%';
  progLabel.textContent = completed
    ? `training complete, ${done} steps`
    : `stopped at step ${done} of ${totalSteps}`;

  // offer this freshly trained model to the rest of the app
  if (state.ebm) state.ebm.dispose();
  state.ebm = model;
  state.ebmReady = true;
  setStatus('using your trained model', true);
  buffer.dispose();
  optimizer.dispose();
  trLoop.running = false;
  setTrStatus(completed ? 'done' : 'stopped', true);
  runBtn.disabled = false;
  runBtn.textContent = 'Train from scratch';
  document.getElementById('tr-stop').disabled = true;
}

function setTrStatus(text, on) {
  const el = document.getElementById('tr-status');
  el.querySelector('.txt').textContent = text;
  el.className = 'pill' + (on === true ? ' on' : on === 'busy' ? ' busy' : '');
}

// ===========================================================================
// TAB: analysis
// ===========================================================================
function initAnalysis() {
  if (state.inited.analysis) return;
  state.inited.analysis = true;
  document.getElementById('an-gen').onclick = generateGallery;
  document.getElementById('an-replay').onclick = replayOne;
}

async function generateGallery() {
  if (!state.ebmReady) { document.getElementById('an-status').textContent = 'checkpoint still loading'; return; }
  document.getElementById('an-status').textContent = 'sampling ...';
  await tf.nextFrame();
  const start = viz.makeNoise(16);
  const { final } = generateSamples(tf, state.ebm, start, 400, 10, 0.005);
  start.dispose();
  viz.drawGrid(document.getElementById('an-gallery'), final, { cols: 8, scale: 3, max: 16 });
  final.dispose();
  document.getElementById('an-status').textContent = 'done';
}

async function replayOne() {
  if (!state.ebmReady) { document.getElementById('an-status').textContent = 'checkpoint still loading'; return; }
  const start = viz.makeNoise(1);
  const steps = 500;
  const snapAt = [0, 1, 3, 5, 10, 30, 50, 100, 300, 500];
  const { final, snapshots } = generateSamples(tf, state.ebm, start, steps, 10, 0.005, { snapshotAt: snapAt });
  start.dispose(); final.dispose();
  viz.drawSnapStrip(document.getElementById('an-snaps'), snapshots, 3);
  snapshots.forEach((t) => t.dispose());
}

// ===========================================================================
// TAB: other EBMs, RBM
// ===========================================================================
let rbm = null;
function initOther() {
  if (state.inited.other) return;
  state.inited.other = true;
  document.getElementById('rbm-train').onclick = trainRBM;
  document.getElementById('rbm-gibbs').onclick = runGibbs;
}

async function trainRBM() {
  if (!state.dataReady) return;
  const note = document.getElementById('rbm-note');
  const trainBtn = document.getElementById('rbm-train');
  const gibbsBtn = document.getElementById('rbm-gibbs');
  note.style.color = '';
  note.textContent = 'Training the RBM ...';
  trainBtn.disabled = true;
  trainBtn.textContent = 'Training ...';
  gibbsBtn.disabled = true;
  await tf.nextFrame();
  if (rbm) rbm.dispose();
  rbm = new RBM();
  const idx = state.split.train;
  const batch = data.batchTensor(idx);
  await rbm.train(batch, {
    epochs: 12, batchSize: 64, lr: 0.05,
    onEpoch: (e, t) => (note.textContent = `Training the RBM ... epoch ${e} of ${t}`),
  });
  batch.dispose();
  // clear success feedback, and point the student to the next step
  note.innerHTML = '✅ <b>Training complete.</b> Now click <b>Run Gibbs sampling</b> below to watch it reconstruct a digit.';
  note.style.color = 'var(--good)';
  trainBtn.disabled = false;
  trainBtn.textContent = 'Retrain the RBM';
  trainBtn.classList.remove('primary');
  gibbsBtn.disabled = false;
  gibbsBtn.classList.add('primary'); // highlight the next step
}

async function runGibbs() {
  if (!rbm) return;
  const note = document.getElementById('rbm-note');
  const gibbsBtn = document.getElementById('rbm-gibbs');
  gibbsBtn.disabled = true;
  note.style.color = '';
  note.textContent = 'Running Gibbs sampling, watch the hidden half fill in ...';
  const idx = pickRandom(state.split.train, 1);
  const batch = data.batchTensor(idx);
  let v = RBM.binarize(batch);
  batch.dispose();
  // hide the bottom half to make reconstruction visible
  v = tf.tidy(() => {
    const mask = tf.concat([tf.ones([1, 512]), tf.zeros([1, 512])], 1);
    return tf.keep(v.mul(mask));
  });
  const cv = document.getElementById('rbm-canvas');
  for (let s = 0; s <= 40; s++) {
    const { vSample, vProb } = rbm.gibbsOnce(v);
    vProb.dispose();
    v.dispose(); v = tf.keep(vSample);
    if (s % 2 === 0) {
      const img = visibleToImage(v);
      viz.drawTensor(cv, img, 7); img.dispose();
      document.getElementById('rbm-step').textContent = 'Gibbs step ' + s;
      await tf.nextFrame();
    }
  }
  v.dispose();
  note.innerHTML = '<b>Done.</b> That was 40 Gibbs steps, one reconstruction. Click <b>Run Gibbs sampling</b> again for a new digit.';
  gibbsBtn.disabled = false;
}

// ===========================================================================
// TAB: classifier as EBM
// ===========================================================================
function initJem() {
  if (state.inited.jem) return;
  state.inited.jem = true;
  document.getElementById('jem-eval').onclick = jemEval;
  jemEval();
}

function jemEval() {
  if (!state.clfReady) { document.getElementById('jem-verdict').textContent = 'Waiting for the classifier checkpoint.'; return; }
  // Average over a batch. A single noise image can fool an overconfident classifier,
  // which is exactly why we compare averages here (and why the JEM paper adds energy
  // training). Averaged over many samples, real digits sit at clearly lower energy.
  const n = 48;
  const idx = pickRandom(state.split.train, n);
  const real = data.batchTensor(idx);
  const noise = viz.makeNoise(n);
  const real1 = tf.tidy(() => real.slice([0, 0, 0, 0], [1, -1, -1, -1]));
  const noise1 = tf.tidy(() => noise.slice([0, 0, 0, 0], [1, -1, -1, -1]));
  viz.drawTensor(document.getElementById('jem-real'), real1, 5);
  viz.drawTensor(document.getElementById('jem-noise'), noise1, 5);

  const eReal = classifierEnergy(real);
  const eNoise = classifierEnergy(noise);
  const pred = tf.tidy(() => state.clf.predict(real1).argMax(1).dataSync()[0]);
  document.getElementById('jem-e-real').textContent = eReal.toFixed(2);
  document.getElementById('jem-e-noise').textContent = eNoise.toFixed(2);
  document.getElementById('jem-pred').textContent = String(pred);
  document.getElementById('jem-real-cap').textContent = 'real digit, label ' + state.split.labels[idx[0]];
  document.getElementById('jem-verdict').textContent =
    eReal < eNoise ? `Real digits average lower energy (${eReal.toFixed(1)} vs ${eNoise.toFixed(1)}), over ${n} samples of each.`
                   : 'Averaged energies are close here, try again.';
  real.dispose(); noise.dispose(); real1.dispose(); noise1.dispose();
}

function classifierEnergy(batch) {
  // E(x) = -logsumexp(logits)
  return tf.tidy(() => state.clf.predict(batch).logSumExp(1).mul(-1).mean().dataSync()[0]);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function pickRandom(arr, k) {
  const out = [];
  for (let i = 0; i < k; i++) out.push(arr[Math.floor(Math.random() * arr.length)]);
  return out;
}

function bindSlider(id, labelId, fmt = (v) => v) {
  const s = document.getElementById(id);
  const l = document.getElementById(labelId);
  const upd = () => (l.textContent = fmt(+s.value));
  s.addEventListener('input', upd);
  upd();
}

const INIT = {
  intuition: initIntuition,
  data: initData,
  energy: initEnergy,
  langevin: initLangevin,
  training: initTraining,
  analysis: initAnalysis,
  other: initOther,
  jem: initJem,
};

boot();
