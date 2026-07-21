// Headless smoke test of the running app. Drives the page with puppeteer-core using the
// installed Chrome, walks the tabs, exercises the non trivial demos, and reports console
// errors. This is verification, not a build artifact.
// No long dashes are used anywhere in this project.

import fs from 'fs';
import puppeteer from 'puppeteer-core';

const CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
];
const CHROME = CANDIDATES.find((p) => fs.existsSync(p)) || CANDIDATES[0];
const URL = 'http://localhost:8000/';

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function main() {
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: 'new',
    args: ['--no-sandbox', '--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'],
  });
  const page = await browser.newPage();
  const errors = [];
  const logs = [];
  page.on('console', (m) => {
    logs.push(`[${m.type()}] ${m.text()}`);
    if (m.type() === 'error') errors.push(m.text());
  });
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));

  await page.goto(URL, { waitUntil: 'networkidle2', timeout: 60000 });
  await wait(2000);

  const results = {};
  const step = async (name, fn) => {
    try { results[name] = await fn(); }
    catch (e) { results[name] = 'ERR: ' + e.message; }
  };

  // If the app failed to boot, tabs will be missing. Report immediately.
  const bootTabs = await page.$$eval('.tab-btn', (els) => els.length).catch(() => 0);
  if (!bootTabs) {
    console.log('BOOT FAILED: no tabs rendered. Console follows:');
    logs.forEach((l) => console.log('  ', l));
    await browser.close();
    process.exit(3);
  }

  const tab = async (i) => {
    await page.$$eval('.tab-btn', (els, idx) => els[idx] && els[idx].click(), i);
    await wait(700);
  };

  await step('backend', async () => { await page.evaluate(() => window.tf.ready()); return page.evaluate(() => window.tf.getBackend()); });
  await step('tabCount', () => page.$$eval('.tab-btn', (els) => els.length));
  await step('listings', () => page.$$eval('pre.code', (els) => els.length));

  await tab(1);
  await step('boltzCanvas', () => page.$eval('#boltz-canvas', (c) => c.width > 0 && c.height > 0));

  await tab(2);
  await wait(800);
  await step('dataGridCells', () => page.$$eval('#data-grid canvas', (e) => e.length));

  await tab(3);
  await page.waitForFunction(() => {
    const t = document.querySelector('#model-status .txt');
    return t && (t.textContent.includes('loaded') || t.textContent.includes('your trained'));
  }, { timeout: 45000 }).catch(() => {});
  await step('modelStatus', () => page.$eval('#model-status .txt', (e) => e.textContent));
  await page.click('#energy-eval').catch(() => {});
  await wait(800);
  await step('eReal', () => page.$eval('#e-real', (e) => e.textContent));
  await step('eNoise', () => page.$eval('#e-noise', (e) => e.textContent));

  await tab(4);
  await page.$eval('#lang-steps', (s) => (s.value = 60)).catch(() => {});
  await page.click('#lang-run').catch(() => {});
  await wait(8000);
  await step('langStep', () => page.$eval('#lang-step', (e) => e.textContent));
  await step('langEnergy', () => page.$eval('#lang-energy', (e) => e.textContent));
  await step('langSnaps', () => page.$$eval('#lang-snaps .snap', (e) => e.length));

  await tab(8);
  await wait(500);
  await page.click('#jem-eval').catch(() => {});
  await wait(800);
  await step('jemReal', () => page.$eval('#jem-e-real', (e) => e.textContent));
  await step('jemNoise', () => page.$eval('#jem-e-noise', (e) => e.textContent));
  await step('jemPred', () => page.$eval('#jem-pred', (e) => e.textContent));

  await step('tensors', () => page.evaluate(() => window.tf.memory().numTensors));

  console.log('RESULTS', JSON.stringify(results, null, 2));
  console.log('CONSOLE ERRORS:', errors.length);
  errors.slice(0, 20).forEach((e) => console.log('  ', e));

  await browser.close();
  if (errors.length) process.exit(2);
}

main().catch((e) => { console.error('SMOKE FAILED', e); process.exit(1); });
