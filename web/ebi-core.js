// EBI Classifier, browser pipeline.
//
// Everything here mirrors dataset.py and app.py so that the browser produces
// the same probabilities as the Python application. web/dev/validate.html
// checks that claim against PyTorch reference values. Nothing in this file
// sends data anywhere: images are read from the user's disk into memory and
// the model runs locally in WebAssembly.

const CHANNEL_RE = /(?:^|[_\-])ch(\d+)(?:\D|$)/i;
const TIFF_SUFFIXES = ['.ome.tif', '.ome.tiff', '.tif', '.tiff'];

export function isTiff(name) {
  const lower = name.toLowerCase();
  return lower.endsWith('.tif') || lower.endsWith('.tiff');
}

export function parseChannel(name) {
  const match = CHANNEL_RE.exec(name);
  return match ? parseInt(match[1], 10) : null;
}

export function parseSampleId(name) {
  const index = name.toLowerCase().indexOf('_ch');
  if (index > 0) return name.slice(0, index);
  let cleaned = name.replace(new RegExp(CHANNEL_RE.source, 'gi'), '_');
  for (const suffix of TIFF_SUFFIXES) {
    if (cleaned.toLowerCase().endsWith(suffix)) {
      cleaned = cleaned.slice(0, cleaned.length - suffix.length);
      break;
    }
  }
  return cleaned.replace(/^[_-]+|[_-]+$/g, '');
}

// Group files (anything with a .name) into {sampleId: {channel: file}},
// keeping only the three requested channels.
export function groupFiles(files, channels) {
  const wanted = new Set(channels);
  const groups = new Map();
  for (const file of files) {
    if (!isTiff(file.name)) continue;
    const channel = parseChannel(file.name);
    if (channel === null || !wanted.has(channel)) continue;
    const id = parseSampleId(file.name);
    if (!groups.has(id)) groups.set(id, {});
    groups.get(id)[channel] = file;
  }
  const complete = [];
  const incomplete = [];
  for (const [id, byChannel] of groups) {
    if (Object.keys(byChannel).length === wanted.size) complete.push({ id, files: byChannel });
    else incomplete.push(id);
  }
  return { complete, incomplete };
}

// --- TIFF -----------------------------------------------------------------
// Minimal reader for single-channel, uncompressed, stripped TIFF (8/16/32-bit
// integer or 32-bit float), which is what ImageStream exports. It reads the
// first image only, as the Python reader does, and refuses anything else with
// a clear message rather than guessing.

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 6: 1, 8: 2, 9: 4, 16: 8 };

export function decodeTiff(buffer) {
  const view = new DataView(buffer);
  if (view.byteLength < 8) throw new Error('file is empty or too small to be a TIFF');
  const order = view.getUint16(0, false);
  if (order !== 0x4949 && order !== 0x4d4d) throw new Error('not a TIFF file');
  const le = order === 0x4949;
  const magic = view.getUint16(2, le);
  if (magic === 43) throw new Error('BigTIFF files are not supported in the browser version');
  if (magic !== 42) throw new Error('not a TIFF file');

  const ifd = view.getUint32(4, le);
  const count = view.getUint16(ifd, le);
  const tags = new Map();
  for (let i = 0; i < count; i++) {
    const entry = ifd + 2 + i * 12;
    const tag = view.getUint16(entry, le);
    const type = view.getUint16(entry + 2, le);
    const n = view.getUint32(entry + 4, le);
    const size = TYPE_SIZE[type];
    if (!size) continue;
    const at = size * n <= 4 ? entry + 8 : view.getUint32(entry + 8, le);
    const values = [];
    for (let k = 0; k < n && k < 1 << 20; k++) {
      const p = at + k * size;
      if (size === 1) values.push(view.getUint8(p));
      else if (size === 2) values.push(view.getUint16(p, le));
      else if (size === 4) values.push(view.getUint32(p, le));
      else values.push(Number(view.getBigUint64(p, le)));
    }
    tags.set(tag, values);
  }
  const one = (tag, fallback) => (tags.has(tag) ? tags.get(tag)[0] : fallback);

  const width = one(256);
  const height = one(257);
  const bits = one(258, 1);
  const compression = one(259, 1);
  const samples = one(277, 1);
  const format = one(339, 1);
  if (!width || !height) throw new Error('TIFF has no image dimensions');
  if (tags.has(322) || tags.has(324)) throw new Error('tiled TIFF files are not supported');
  if (compression !== 1) throw new Error('compressed TIFF files are not supported (export uncompressed TIFF)');
  if (samples !== 1) throw new Error('expected one channel per TIFF file');
  const bytes = bits / 8;
  const supported = (format === 1 || format === 2) ? [1, 2, 4].includes(bytes) : (format === 3 && bytes === 4);
  if (!supported) throw new Error(`unsupported pixel type (${bits}-bit, format ${format})`);

  const offsets = tags.get(273);
  const counts = tags.get(279);
  if (!offsets || !counts || offsets.length !== counts.length) throw new Error('TIFF has no strip data');

  const total = width * height;
  const data = new Float32Array(total);
  let pixel = 0;
  for (let s = 0; s < offsets.length && pixel < total; s++) {
    const end = offsets[s] + counts[s];
    if (end > view.byteLength) throw new Error('TIFF is truncated');
    for (let p = offsets[s]; p + bytes <= end && pixel < total; p += bytes) {
      let v;
      if (format === 3) v = view.getFloat32(p, le);
      else if (format === 2) v = bytes === 1 ? view.getInt8(p) : bytes === 2 ? view.getInt16(p, le) : view.getInt32(p, le);
      else v = bytes === 1 ? view.getUint8(p) : bytes === 2 ? view.getUint16(p, le) : view.getUint32(p, le);
      data[pixel++] = v;
    }
  }
  if (pixel !== total) throw new Error('TIFF is truncated');
  return { width, height, data };
}

// --- Intensity normalisation (dataset._to_float01, "percentile" mode) -------

function percentile(sorted, q) {
  const position = (sorted.length - 1) * (q / 100);
  const low = Math.floor(position);
  const high = Math.min(low + 1, sorted.length - 1);
  const t = position - low;
  return sorted[low] + (sorted[high] - sorted[low]) * t;
}

export function toFloat01(data, percentileLow, percentileHigh) {
  const x = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) x[i] = Math.max(data[i], 0);
  const sorted = Float32Array.from(x).sort();
  const lo = percentile(sorted, percentileLow);
  const hi = percentile(sorted, percentileHigh);
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) {
    const max = sorted[sorted.length - 1];
    if (!(max > 0) || !Number.isFinite(max)) return x.fill(0);
    const scale = Math.fround(Math.max(1e-6, max));
    for (let i = 0; i < x.length; i++) x[i] = Math.min(1, Math.max(0, x[i] / scale));
    return x;
  }
  const lo32 = Math.fround(lo);
  const range = Math.fround(Math.max(1e-6, hi - lo));
  for (let i = 0; i < x.length; i++) {
    x[i] = Math.min(1, Math.max(0, Math.fround(x[i] - lo32) / range));
  }
  return x;
}

// --- Resize (torchvision v2.Resize, bilinear, antialias=True) ---------------

function resizeWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const support = scale >= 1 ? scale : 1;
  const inv = scale >= 1 ? 1 / scale : 1;
  const starts = new Int32Array(outSize);
  const weights = [];
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5);
    const min = Math.max(0, Math.trunc(center - support + 0.5));
    const size = Math.min(inSize, Math.trunc(center + support + 0.5)) - min;
    const w = new Float64Array(size);
    let sum = 0;
    for (let j = 0; j < size; j++) {
      const d = Math.abs((j + min - center + 0.5) * inv);
      w[j] = d < 1 ? 1 - d : 0;
      sum += w[j];
    }
    for (let j = 0; j < size; j++) w[j] /= sum;
    starts[i] = min;
    weights.push(w);
  }
  return { starts, weights };
}

export function resizeBilinear(src, height, width, newHeight, newWidth) {
  const horizontal = resizeWeights(width, newWidth);
  const temp = new Float32Array(height * newWidth);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < newWidth; x++) {
      const w = horizontal.weights[x];
      const start = y * width + horizontal.starts[x];
      let acc = 0;
      for (let j = 0; j < w.length; j++) acc += src[start + j] * w[j];
      temp[y * newWidth + x] = acc;
    }
  }
  const vertical = resizeWeights(height, newHeight);
  const out = new Float32Array(newHeight * newWidth);
  for (let y = 0; y < newHeight; y++) {
    const w = vertical.weights[y];
    const start = vertical.starts[y];
    for (let x = 0; x < newWidth; x++) {
      let acc = 0;
      for (let j = 0; j < w.length; j++) acc += temp[(start + j) * newWidth + x] * w[j];
      out[y * newWidth + x] = acc;
    }
  }
  return out;
}

// Python's round(): halves go to the even neighbour.
function roundHalfEven(value) {
  const floor = Math.floor(value);
  const diff = value - floor;
  if (diff < 0.5) return floor;
  if (diff > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

// dataset.letterbox_resize_pad for one channel: returns size*size floats.
export function letterbox(src, height, width, size) {
  const r = Math.min(size / height, size / width);
  const newHeight = Math.max(1, roundHalfEven(height * r));
  const newWidth = Math.max(1, roundHalfEven(width * r));
  const resized = resizeBilinear(src, height, width, newHeight, newWidth);
  const top = Math.floor((size - newHeight) / 2);
  const left = Math.floor((size - newWidth) / 2);
  const out = new Float32Array(size * size);
  for (let y = 0; y < newHeight; y++) {
    out.set(resized.subarray(y * newWidth, (y + 1) * newWidth), (top + y) * size + left);
  }
  return out;
}

// Build the model input (3 x size x size, channel-first) from three decoded
// images given in (erythroblasts, macrophages, neutrophils) order.
export function prepareInput(images, config) {
  const size = config.image_size;
  const { width, height } = images[0];
  for (const image of images) {
    if (image.width !== width || image.height !== height) {
      throw new Error('the three channel images have different sizes');
    }
  }
  const input = new Float32Array(3 * size * size);
  images.forEach((image, c) => {
    const normalised = toFloat01(image.data, config.percentile_low, config.percentile_high);
    input.set(letterbox(normalised, height, width, size), c * size * size);
  });
  return input;
}

// --- Model ------------------------------------------------------------------

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function fetchWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  const total = Number(response.headers.get('Content-Length')) || 0;
  if (!response.body || !total) return response.arrayBuffer();
  const reader = response.body.getReader();
  const bytes = new Uint8Array(total);
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    bytes.set(value, received);
    received += value.length;
    if (onProgress) onProgress(received / total);
  }
  return bytes.buffer;
}

// `ort` is the global from vendor/ort.wasm.min.js. `base` is the URL of the
// web/ folder, so the same code serves the application and the validation page.
export async function loadClassifier(ort, base, onProgress) {
  const config = await (await fetch(new URL('model/config.json', base))).json();
  if (config.intensity_norm !== 'percentile') {
    throw new Error(`unsupported intensity normalisation: ${config.intensity_norm}`);
  }
  const model = await fetchWithProgress(new URL('model/ebi.onnx', base), onProgress);
  let verified = false;
  if (globalThis.crypto && crypto.subtle) {
    if ((await sha256Hex(model)) !== config.onnx_sha256) {
      throw new Error('the model file does not match its recorded checksum');
    }
    verified = true;
  }
  ort.env.wasm.wasmPaths = new URL('vendor/', base).href;
  const threads = globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.numThreads = threads;
  const session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
  const size = config.image_size;

  async function classify(input) {
    const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
    const { logits } = await session.run({ image: tensor });
    const [a, b] = logits.data;
    const m = Math.max(a, b);
    const ea = Math.exp(a - m);
    const eb = Math.exp(b - m);
    return eb / (ea + eb);
  }

  return { config, classify, threads, verified };
}

export async function classifySample(classifier, sample, channelOrder) {
  const images = [];
  for (const channel of channelOrder) {
    const file = sample.files[channel];
    try {
      images.push(decodeTiff(await file.arrayBuffer()));
    } catch (error) {
      throw new Error(`${file.name}: ${error.message}`);
    }
  }
  const input = prepareInput(images, classifier.config);
  const prob = await classifier.classify(input);
  return { id: sample.id, prob, isEbi: prob >= classifier.config.threshold, input };
}

export function popFileContent(ids) {
  return ids.map((id) => `${id},`).join('\n');
}

// CSV of the objects the user moved to the other group. It holds file names and
// numbers only, no image data, and is meant to be sent back with the TIFF files
// for retraining (see scripts/import_corrections.py).
// Each item: {id, prob, predicted: 'ebi'|'notEbi', corrected: 'EBI'|'NOT EBI', files}.
export function correctionsCsv(items, config) {
  const quote = (value) => {
    const text = String(value);
    return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  const header = ['object_id', 'predicted_class', 'probability_ebi', 'corrected_class', 'files', 'model_sha256'];
  const rows = items.map((item) => [
    item.id,
    item.predicted === 'ebi' ? 'EBI' : 'NOT EBI',
    item.prob.toFixed(4),
    item.corrected,
    item.files.join(';'),
    config.onnx_sha256 || '',
  ]);
  return [header, ...rows].map((row) => row.map(quote).join(',')).join('\r\n') + '\r\n';
}
