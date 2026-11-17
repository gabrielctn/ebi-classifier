import { classifySample, letterbox, loadClassifier } from '../ebi-core.js';

const log = document.getElementById('log');
const lines = [];
const say = (text) => { lines.push(text); log.textContent = lines.join('\n'); };

async function run() {
  const web = new URL('../', import.meta.url);
  const root = new URL('../../', import.meta.url);
  const result = { done: false };
  window.__result = result;

  // 1. Resize, including the antialiased downscaling path.
  const resize = await (await fetch(new URL('resize_case.json', import.meta.url))).json();
  const pattern = new Float32Array(resize.height * resize.width);
  for (let y = 0; y < resize.height; y++) {
    for (let x = 0; x < resize.width; x++) {
      pattern[y * resize.width + x] = Math.fround(((y * 7 + x * 13 + ((y * x) % 31)) % 256) / 255);
    }
  }
  const resized = letterbox(pattern, resize.height, resize.width, resize.size);
  let resizeDiff = 0;
  for (let i = 0; i < resized.length; i++) {
    resizeDiff = Math.max(resizeDiff, Math.abs(resized[i] - resize.expected[i]));
  }
  result.resizeMaxDiff = resizeDiff;
  say(`Resize (300x260 downscale) max pixel difference: ${resizeDiff.toExponential(2)}`);

  // 2. End-to-end probabilities on the test sets.
  const classifier = await loadClassifier(globalThis.ort, web);
  say(`Model loaded (checksum verified: ${classifier.verified}, threads: ${classifier.threads})`);
  const reference = await (await fetch(new URL('reference.json', import.meta.url))).json();
  const limit = Number(new URLSearchParams(location.search).get('limit')) || reference.samples.length;
  const step = Math.max(1, Math.floor(reference.samples.length / limit));
  const samples = reference.samples.filter((_, i) => i % step === 0).slice(0, limit);

  let maxDiff = 0;
  let labelMismatches = 0;
  let worst = null;
  const started = performance.now();
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    const files = {};
    reference.channels.forEach((channel, k) => {
      const name = sample.files[k];
      files[channel] = {
        name,
        arrayBuffer: async () => (await fetch(new URL(`${sample.folder}/${name}`, root))).arrayBuffer(),
      };
    });
    const { prob } = await classifySample(classifier, { id: sample.id, files }, reference.channels);
    const diff = Math.abs(prob - sample.prob);
    if (diff > maxDiff) { maxDiff = diff; worst = `${sample.folder}/${sample.id}`; }
    if ((prob >= reference.threshold) !== (sample.prob >= reference.threshold)) labelMismatches++;
    if (i % 10 === 9 || i === samples.length - 1) {
      log.textContent = lines.join('\n') + `\n${i + 1}/${samples.length} samples…`;
    }
  }
  const seconds = (performance.now() - started) / 1000;
  Object.assign(result, {
    samples: samples.length, maxProbDiff: maxDiff, worst, labelMismatches,
    secondsPerSample: seconds / samples.length, threads: classifier.threads, done: true,
  });
  say(`${samples.length} samples in ${seconds.toFixed(1)} s (${(seconds / samples.length).toFixed(2)} s each)`);
  say(`Max probability difference vs PyTorch: ${maxDiff.toExponential(2)} (${worst})`);
  say(`Label mismatches at threshold ${reference.threshold}: ${labelMismatches}`);
  say(labelMismatches === 0 && maxDiff < 1e-3 && resizeDiff < 1e-4 ? 'PASS' : 'FAIL');
}

run().catch((error) => {
  window.__result = { done: true, error: String(error && error.stack || error) };
  say(`ERROR: ${error.message}`);
});
