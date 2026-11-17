import { classifySample, groupFiles, isTiff, loadClassifier, popFileContent } from './ebi-core.js';

const $ = (id) => document.getElementById(id);
const selects = {
  erythroblasts: $('ch-erythroblasts'),
  macrophages: $('ch-macrophages'),
  neutrophils: $('ch-neutrophils'),
};

let classifier = null;
let files = [];
let running = false;
let cancelled = false;
let popUrl = null;
// Each result: {id, prob, url}. `selected` holds "group:id" keys.
const results = { ebi: [], notEbi: [] };
const selected = new Set();

function setStatus(element, text, kind = '') {
  element.textContent = text;
  element.className = `status ${kind}`.trim();
}

function channelOrder() {
  return [selects.erythroblasts, selects.macrophages, selects.neutrophils].map((s) => Number(s.value));
}

function refreshFiles() {
  const order = channelOrder();
  const summary = $('file-summary');
  if (new Set(order).size !== 3) {
    setStatus(summary, 'Choose three different channels.', 'bad');
    $('run').disabled = true;
    return null;
  }
  const tiffs = files.filter((f) => isTiff(f.name));
  if (!tiffs.length) {
    setStatus(summary, files.length ? 'None of the selected files are TIFF images.' : 'No images selected.',
      files.length ? 'bad' : '');
    $('run').disabled = true;
    return null;
  }
  const groups = groupFiles(tiffs, order);
  let text = `${tiffs.length} TIFF files: ${groups.complete.length} complete triplets`;
  if (groups.incomplete.length) {
    const shown = groups.incomplete.slice(0, 5).join(', ');
    text += `, ${groups.incomplete.length} incomplete and skipped (${shown}${groups.incomplete.length > 5 ? ', …' : ''})`;
  }
  setStatus(summary, `${text}.`, groups.complete.length ? '' : 'bad');
  $('run').disabled = !classifier || running || !groups.complete.length;
  return groups;
}

function setFiles(list) {
  files = Array.from(list);
  refreshFiles();
}

// --- Thumbnails ---------------------------------------------------------------

function thumbnailUrl(input, size) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  const image = context.createImageData(size, size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    // Same conversion as the Python preview: truncate value * 255.
    image.data[i * 4] = input[i] * 255;
    image.data[i * 4 + 1] = input[plane + i] * 255;
    image.data[i * 4 + 2] = input[2 * plane + i] * 255;
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  // toDataURL is synchronous. canvas.toBlob waits for an idle moment, which
  // takes about a second per image when the tab is in the background.
  return canvas.toDataURL('image/png');
}

// --- Results --------------------------------------------------------------------

function renderGallery(group, container) {
  container.replaceChildren(...results[group].map((item) => {
    const key = `${group}:${item.id}`;
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'tile';
    tile.setAttribute('aria-pressed', String(selected.has(key)));
    tile.title = `${item.id} (p = ${item.prob.toFixed(3)})`;
    const image = document.createElement('img');
    image.src = item.url;
    image.alt = '';
    image.loading = 'lazy';
    const label = document.createElement('span');
    label.textContent = `${item.id} · ${item.prob.toFixed(3)}`;
    tile.append(image, label);
    tile.addEventListener('click', () => {
      if (selected.has(key)) selected.delete(key); else selected.add(key);
      tile.setAttribute('aria-pressed', String(selected.has(key)));
      renderSelection();
    });
    return tile;
  }));
}

function renderSelection() {
  const ebi = [...selected].filter((k) => k.startsWith('ebi:')).length;
  const notEbi = selected.size - ebi;
  const parts = [];
  if (ebi) parts.push(`EBI: ${ebi} selected`);
  if (notEbi) parts.push(`NOT EBI: ${notEbi} selected`);
  $('selection-info').textContent = parts.join(' | ') || 'Nothing selected.';
  $('move-to-not-ebi').disabled = !ebi;
  $('move-to-ebi').disabled = !notEbi;
  $('clear-selection').disabled = !selected.size;
}

function renderResults() {
  // Same order as the Python app: most confident first in each group.
  results.ebi.sort((a, b) => b.prob - a.prob);
  results.notEbi.sort((a, b) => a.prob - b.prob);
  renderGallery('ebi', $('gallery-ebi'));
  renderGallery('notEbi', $('gallery-not-ebi'));
  $('count-ebi').textContent = `(${results.ebi.length})`;
  $('count-not-ebi').textContent = `(${results.notEbi.length})`;
  renderSelection();

  if (popUrl) URL.revokeObjectURL(popUrl);
  popUrl = URL.createObjectURL(new Blob([popFileContent(results.ebi.map((r) => r.id))], { type: 'text/plain' }));
  $('download').href = popUrl;
  const total = results.ebi.length + results.notEbi.length;
  $('download').hidden = !total;
  $('review').hidden = !total;
}

function move(from, to) {
  const moving = new Set([...selected].filter((k) => k.startsWith(`${from}:`)).map((k) => k.slice(from.length + 1)));
  if (!moving.size) return;
  results[to].push(...results[from].filter((r) => moving.has(r.id)));
  results[from] = results[from].filter((r) => !moving.has(r.id));
  selected.clear();
  renderResults();
  const total = results.ebi.length + results.notEbi.length;
  setStatus($('run-status'),
    `${total} samples: ${results.ebi.length} EBI, ${results.notEbi.length} NOT EBI (after your corrections).`, 'ok');
}

function clearResults() {
  for (const group of ['ebi', 'notEbi']) {
    results[group] = [];
  }
  selected.clear();
  renderResults();
}

// --- Run --------------------------------------------------------------------------

async function run() {
  const groups = refreshFiles();
  if (!groups || !groups.complete.length || running) return;
  const order = channelOrder();
  running = true;
  cancelled = false;
  clearResults();
  $('run').disabled = true;
  $('cancel').hidden = false;
  $('errors').hidden = true;
  $('errors').replaceChildren();
  const progress = $('run-progress');
  progress.hidden = false;
  progress.value = 0;

  const failures = [];
  const size = classifier.config.image_size;
  const started = performance.now();
  let done = 0;
  for (const sample of groups.complete) {
    if (cancelled) break;
    try {
      const result = await classifySample(classifier, sample, order);
      const url = thumbnailUrl(result.input, size);
      results[result.isEbi ? 'ebi' : 'notEbi'].push({ id: result.id, prob: result.prob, url });
    } catch (error) {
      failures.push(error.message);
    }
    done++;
    progress.value = done / groups.complete.length;
    setStatus($('run-status'), `Classifying ${done} of ${groups.complete.length}…`);
    // Let the browser repaint between samples. A message channel is used
    // because browsers slow timers down to one per second in background tabs.
    await new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = resolve;
      channel.port2.postMessage(null);
    });
  }

  running = false;
  $('cancel').hidden = true;
  progress.hidden = true;
  renderResults();
  const total = results.ebi.length + results.notEbi.length;
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  let text = `${cancelled ? 'Stopped. ' : ''}${total} samples in ${seconds} s: ${results.ebi.length} EBI, ${results.notEbi.length} NOT EBI.`;
  if (failures.length) text += ` ${failures.length} could not be read.`;
  setStatus($('run-status'), text, total ? 'ok' : 'bad');
  if (failures.length) {
    $('errors').hidden = false;
    $('errors').replaceChildren(...failures.slice(0, 20).map((message) => {
      const item = document.createElement('li');
      item.textContent = message;
      return item;
    }));
  }
  refreshFiles();
}

// --- Dropped folders ------------------------------------------------------------------

async function filesFromDrop(dataTransfer) {
  const entries = Array.from(dataTransfer.items || [])
    .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null))
    .filter(Boolean);
  if (!entries.length) return Array.from(dataTransfer.files);
  const found = [];
  async function walk(entry) {
    if (entry.isFile) {
      found.push(await new Promise((resolve, reject) => entry.file(resolve, reject)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  }
  for (const entry of entries) await walk(entry);
  return found;
}

// --- Start ----------------------------------------------------------------------------

async function start() {
  for (const select of Object.values(selects)) {
    for (let channel = 1; channel <= 12; channel++) select.add(new Option(`Ch${channel}`, String(channel)));
    select.addEventListener('change', refreshFiles);
  }
  // Training defaults, shown while the model loads; replaced by the values in
  // model/config.json once it is read.
  selects.erythroblasts.value = '7';
  selects.macrophages.value = '11';
  selects.neutrophils.value = '3';
  $('files').addEventListener('change', (event) => setFiles(event.target.files));
  $('folder').addEventListener('change', (event) => setFiles(event.target.files));
  const drop = $('drop');
  drop.addEventListener('dragover', (event) => { event.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', async (event) => {
    event.preventDefault();
    drop.classList.remove('over');
    setFiles(await filesFromDrop(event.dataTransfer));
  });
  $('run').addEventListener('click', run);
  $('cancel').addEventListener('click', () => { cancelled = true; });
  $('clear-selection').addEventListener('click', () => { selected.clear(); renderResults(); });
  $('move-to-not-ebi').addEventListener('click', () => move('ebi', 'notEbi'));
  $('move-to-ebi').addEventListener('click', () => move('notEbi', 'ebi'));
  renderSelection();

  const status = $('model-status');
  const progress = $('model-progress');
  try {
    classifier = await loadClassifier(globalThis.ort, new URL('./', import.meta.url), (fraction) => {
      progress.value = fraction;
      status.textContent = `Loading the model… ${Math.round(fraction * 100)}%`;
    });
    const defaults = classifier.config.default_channels;
    for (const [name, select] of Object.entries(selects)) select.value = String(defaults[name]);
    setStatus(status, `Model ready${classifier.verified ? ' (file checksum verified)' : ''}.`, 'ok');
  } catch (error) {
    setStatus(status, `The model could not be loaded: ${error.message}`, 'bad');
  }
  progress.hidden = true;
  refreshFiles();
}

start();
