// Dependency-free SVG charts. Every chart returns an SVG string with
// viewBox-based sizing (responsive) and data-tip attributes for tooltips.
import { esc } from "./util.js";

const f1 = (n) => (n * 100).toFixed(1);

/** Horizontal bars. items: [{label, value, color, tip, segments?}] */
export function hbars(items, { max, log = false, fmt = (v) => v, labelW = 92, rowH = 30 } = {}) {
  const W = 560, pad = 8, barX = labelW + pad, barW = W - barX - 64;
  const top = Math.max(...items.map((i) => i.value), 1);
  const scale = (v) => {
    const m = max ?? top;
    return log ? Math.log10(v + 1) / Math.log10(m + 1) : v / m;
  };
  const H = items.length * rowH + 4;
  const rows = items.map((it, i) => {
    const y = i * rowH + 2;
    const w = Math.max(2, scale(it.value) * barW);
    const tip = esc(it.tip || `${it.label}: ${fmt(it.value)}`);
    let bar;
    if (it.segments) {
      const total = it.segments.reduce((a, s) => a + s.value, 0) || 1;
      let x = barX;
      bar = it.segments.map((s) => {
        const sw = (s.value / total) * w; const r = `<rect x="${x}" y="${y + 6}" width="${sw}" height="${rowH - 12}" fill="${s.color}" data-tip="${esc(s.tip)}" tabindex="0"/>`;
        x += sw; return r;
      }).join("");
    } else {
      bar = `<rect x="${barX}" y="${y + 6}" width="${w}" height="${rowH - 12}" rx="4" fill="${it.color || "var(--accent-2)"}"/>`;
    }
    return `<g data-tip="${tip}" tabindex="0" role="img" aria-label="${tip}">
      <text x="${labelW}" y="${y + rowH / 2 + 4}" text-anchor="end" class="axis-label">${esc(it.label)}</text>
      <rect x="${barX}" y="${y + 6}" width="${barW}" height="${rowH - 12}" rx="4" fill="var(--surface-3)" opacity=".5"/>
      ${bar}
      <text x="${barX + w + 8}" y="${y + rowH / 2 + 4}">${fmt(it.value)}</text></g>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="group">${rows}</svg>`;
}

/** Row-normalised confusion-matrix heatmap. */
export function confusion(labels, matrix) {
  const n = labels.length;
  const L = 74, T = 70, cell = n > 8 ? 52 : 56;
  const W = L + n * cell + 44, H = T + n * cell + 26;
  let cells = "", xl = "", yl = "";
  labels.forEach((lab, j) => {
    xl += `<text transform="translate(${L + j * cell + cell / 2},${T - 8}) rotate(-40)" text-anchor="start">${esc(lab)}</text>`;
  });
  matrix.forEach((row, i) => {
    const rowSum = row.reduce((a, b) => a + b, 0) || 1;
    yl += `<text x="${L - 8}" y="${T + i * cell + cell / 2 + 4}" text-anchor="end" class="axis-label">${esc(labels[i])}</text>`;
    row.forEach((v, j) => {
      const share = v / rowSum;
      const diag = i === j;
      const base = diag ? "53,208,196" : "255,122,47";
      const a = v === 0 ? 0.03 : 0.12 + Math.min(1, share * (diag ? 1 : 2.4)) * 0.85;
      const tip = `True ${labels[i]} → predicted ${labels[j]}: ${v} (${f1(share)}% of true ${labels[i]})`;
      cells += `<g data-tip="${esc(tip)}" tabindex="0" role="img" aria-label="${esc(tip)}">
        <rect class="heat-cell" x="${L + j * cell + 1}" y="${T + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="5" fill="rgba(${base},${a.toFixed(3)})"/>
        <text x="${L + j * cell + cell / 2}" y="${T + i * cell + cell / 2 + 4}" text-anchor="middle" style="fill:${a > 0.5 ? "#fff" : "var(--text-2)"}; pointer-events:none">${v}</text></g>`;
    });
  });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="group" aria-label="Confusion matrix">
    ${xl}${yl}${cells}
    <text x="${L + (n * cell) / 2}" y="${H - 4}" text-anchor="middle" class="axis-label">Predicted →</text>
    <text transform="translate(12,${T + (n * cell) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">True</text></svg>`;
}

/** Multi-series line chart over epochs. series: [{name,color,values:[y]}], values 0..1 */
export function lines(series, epochs, { yMin = 0, yMax = 1 } = {}) {
  const W = 560, H = 280, L = 44, R = 14, T = 14, B = 34;
  const iw = W - L - R, ih = H - T - B;
  const n = epochs.length;
  const x = (i) => L + (n === 1 ? 0 : (i / (n - 1)) * iw);
  const y = (v) => T + (1 - (v - yMin) / (yMax - yMin)) * ih;
  let grid = "";
  for (let k = 0; k <= 4; k++) {
    const v = yMin + ((yMax - yMin) * k) / 4;
    grid += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${(v * 100).toFixed(0)}%</text>`;
  }
  const step = Math.ceil(n / 8);
  let xt = "";
  epochs.forEach((e, i) => { if (i % step === 0 || i === n - 1) xt += `<text x="${x(i)}" y="${H - 12}" text-anchor="middle">${e}</text>`; });
  const paths = series.map((s) => {
    const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>`;
  }).join("");
  const hit = epochs.map((e, i) => {
    const tip = `Epoch ${e} · ` + series.map((s) => `${s.name} ${f1(s.values[i])}%`).join(" · ");
    const w = n === 1 ? iw : iw / (n - 1);
    return `<rect x="${x(i) - w / 2}" y="${T}" width="${w}" height="${ih}" fill="transparent" data-tip="${esc(tip)}"/>`;
  }).join("");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Training curves">${grid}${xt}
    <text x="${L + iw / 2}" y="${H}" text-anchor="middle" class="axis-label" dy="-0">epoch</text>${paths}${hit}</svg>`;
}

/** Vertical bars 0..1 (e.g. per-event confidence) with a reference line. */
export function vbars(items, { ref, refLabel } = {}) {
  const W = 560, H = 230, L = 40, R = 10, T = 10, B = 26;
  const iw = W - L - R, ih = H - T - B;
  const n = items.length;
  const gap = n > 40 ? 1 : 3;
  const bw = Math.max(2, iw / n - gap);
  const y = (v) => T + (1 - v) * ih;
  let grid = "";
  for (const v of [0, 0.25, 0.5, 0.75, 1]) grid += `<line class="grid-line" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  const bars = items.map((it, i) => {
    const x = L + i * (iw / n) + gap / 2;
    return `<rect class="vbar" x="${x}" y="${y(it.value)}" width="${bw}" height="${ih - (y(it.value) - T)}" rx="2" fill="${it.color}" data-tip="${esc(it.tip)}" data-idx="${i}" tabindex="0" style="cursor:pointer"/>`;
  }).join("");
  const refLine = ref != null ? `<line x1="${L}" x2="${W - R}" y1="${y(ref)}" y2="${y(ref)}" stroke="#fff" stroke-dasharray="4 4" opacity=".55"/><text x="${W - R}" y="${y(ref) - 5}" text-anchor="end" style="fill:#fff;opacity:.7">${esc(refLabel)}</text>` : "";
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="group" aria-label="Confidence per event">${grid}${bars}${refLine}</svg>`;
}

/** Grouped horizontal bars comparing two series (recall per class). */
export function grouped(labels, a, b, { colorA = "var(--text-3)", colorB = "var(--accent)", nameA, nameB } = {}) {
  const W = 560, labelW = 84, rowH = 36, barX = labelW + 8, barW = W - barX - 48;
  const H = labels.length * rowH + 4;
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="group">${labels.map((lab, i) => {
    const y = i * rowH + 2;
    const tip = `${lab} — ${nameA}: ${f1(a[i])}% · ${nameB}: ${f1(b[i])}%`;
    return `<g data-tip="${esc(tip)}" tabindex="0" role="img" aria-label="${esc(tip)}">
      <text x="${labelW}" y="${y + rowH / 2 + 4}" text-anchor="end" class="axis-label">${esc(lab)}</text>
      <rect x="${barX}" y="${y + 4}" width="${a[i] * barW}" height="11" rx="3" fill="${colorA}"/>
      <rect x="${barX}" y="${y + 18}" width="${b[i] * barW}" height="11" rx="3" fill="${colorB}"/>
      <text x="${barX + Math.max(a[i], b[i]) * barW + 8}" y="${y + rowH / 2 + 4}">${f1(b[i])}</text></g>`;
  }).join("")}</svg>`;
}
