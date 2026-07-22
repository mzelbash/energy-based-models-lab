// Tiny dependency free line chart on a canvas. Used for the live training metrics and
// the Boltzmann probability bars. Recolors on theme change by reading CSS variables.
// No long dashes are used anywhere in this project.

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export class LineChart {
  constructor(canvas, { title = '', color = null, height = 150 } = {}) {
    this.canvas = canvas;
    this.title = title;
    this.color = color;
    this.height = height;
    this.series = []; // array of numbers
    window.addEventListener('themechange', () => this.draw());
  }

  push(v) {
    this.series.push(v);
    this.draw();
  }

  reset() {
    this.series = [];
    this.draw();
  }

  draw() {
    const cv = this.canvas;
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth || 260;
    const h = this.height;
    cv.width = w * dpr;
    cv.height = h * dpr;
    cv.style.height = h + 'px';
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const ink = cssVar('--ink');
    const faint = cssVar('--ink-3');
    const line = cssVar('--line');
    const brand = this.color || cssVar('--brand');

    const padL = 42, padR = 10, padT = 24, padB = 20;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // title
    ctx.fillStyle = ink;
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.fillText(this.title, padL, 15);

    const s = this.series;
    if (s.length === 0) {
      ctx.fillStyle = faint;
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('no data yet', padL, padT + plotH / 2);
      return;
    }

    let min = Math.min(...s);
    let max = Math.max(...s);
    if (min === max) { min -= 1; max += 1; }
    const pad = (max - min) * 0.12;
    min -= pad; max += pad;

    const x = (i) => padL + (s.length === 1 ? plotW / 2 : (i / (s.length - 1)) * plotW);
    const y = (v) => padT + plotH - ((v - min) / (max - min)) * plotH;

    // axes and zero line
    ctx.strokeStyle = line;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();
    if (min < 0 && max > 0) {
      ctx.strokeStyle = faint;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, y(0)); ctx.lineTo(padL + plotW, y(0));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // y labels
    ctx.fillStyle = faint;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(max.toFixed(2), padL - 6, padT + 4);
    ctx.fillText(min.toFixed(2), padL - 6, padT + plotH);
    ctx.textAlign = 'left';

    // line
    ctx.strokeStyle = brand;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v))));
    ctx.stroke();

    // last point
    const li = s.length - 1;
    ctx.fillStyle = brand;
    ctx.beginPath();
    ctx.arc(x(li), y(s[li]), 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = ink;
    ctx.font = '600 11px system-ui, sans-serif';
    ctx.fillText(s[li].toFixed(3), Math.min(x(li) + 6, w - 40), y(s[li]) - 6);
  }
}

// Draw a set of labelled probability or energy bars.
export function drawBars(canvas, values, labels, { title = '', height = 170, highlight = null } = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320;
  const h = height;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const ink = cssVar('--ink'), faint = cssVar('--ink-3'), brand = cssVar('--brand'), accent = cssVar('--accent');
  const padT = 22, padB = 22, padL = 8, padR = 8;
  const plotH = h - padT - padB;
  const n = values.length;
  const gap = 8;
  const bw = (w - padL - padR - gap * (n - 1)) / n;
  const max = Math.max(...values, 1e-6);

  ctx.fillStyle = ink; ctx.font = '600 12px system-ui, sans-serif';
  ctx.fillText(title, padL, 14);

  values.forEach((v, i) => {
    const bx = padL + i * (bw + gap);
    const bh = (v / max) * plotH;
    ctx.fillStyle = highlight === i ? accent : brand;
    ctx.beginPath();
    const r = 4, by = padT + plotH - bh;
    ctx.roundRect(bx, by, bw, bh, [r, r, 0, 0]);
    ctx.fill();
    ctx.fillStyle = faint; ctx.font = '10px system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(labels[i], bx + bw / 2, h - 8);
    ctx.fillStyle = ink; ctx.font = '600 10px system-ui, sans-serif';
    ctx.fillText(v.toFixed(2), bx + bw / 2, by - 4);
    ctx.textAlign = 'left';
  });
}
