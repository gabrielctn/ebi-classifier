# EBI Classifier, browser version (prototype)

A version of the EBI Classifier that runs entirely in a web browser. Users
open a web page; nothing is installed on their computer and no Python is
involved. The images they select are read and analysed inside the browser tab
and are not uploaded: the page is a set of static files, and its security
policy forbids it from contacting any server other than the one it was loaded
from.

It uses the same trained model as the desktop packages, converted to ONNX, and
reproduces the same preprocessing in JavaScript.

A public instance runs at https://dsimb.inserm.fr/EBI/.

## Try it on this computer

The model file is not stored in git (112 MB). Download `ebi.onnx` from the
project's [release page](https://github.com/gabrielctn/ebi-classifier/releases) or [Zenodo record](https://doi.org/10.5281/zenodo.24324392) into `web/model/` first; see
[model/LICENSE.md](model/LICENSE.md).

    python3 web/serve.py

Then open http://127.0.0.1:8765/ in Chrome, Edge, Firefox or Safari.

## What is in this folder

| File | Purpose |
| --- | --- |
| `index.html`, `style.css`, `app.js` | The page and its interface |
| `ebi-core.js` | TIFF reading, intensity normalisation, resizing, inference |
| `model/ebi.onnx`, `model/config.json` | The model (112 MB) and its settings, with SHA-256 checksums |
| `vendor/` | onnxruntime-web 1.31.1 (MIT), unmodified, with checksums in `VERSION.txt` |
| `serve.py` | Local test server |
| `dev/` | Validation against PyTorch. Not needed on the hosted site |
| `examples/` | Optional example triplets and `manifest.json`, shown as "Run the examples". Not stored in git; the box is hidden when the folder is absent |

There is no build step and no other dependency. The page loads no fonts,
scripts or analytics from third parties.

## Hosting it

Copy everything in this folder except `dev/` and `serve.py` to any static web
host served over HTTPS. Points to settle first:

- **Who can reach it.** Anyone who can open the page can download the model
  file. The base weights are licensed CC BY-NC 4.0 (see
  `IT_INFORMATION.md`), so decide whether the page should be public
  or restricted, for example behind institutional login.
- **File size.** The model is 112 MB. Some hosts limit file size (GitHub Pages:
  100 MB per file), so check before choosing one.
- **Speed.** With the two headers below the browser uses up to four CPU threads;
  without them it uses one and each sample takes several times longer. The
  results are the same.

      Cross-Origin-Opener-Policy: same-origin
      Cross-Origin-Embedder-Policy: require-corp

- **Content security policy.** The policy is set inside `index.html`. Sending
  the same policy as an HTTP header is stronger and is recommended if the host
  allows it.

[apache-example.conf](apache-example.conf) is a complete Apache configuration
with these headers, a deny-by-default file whitelist and read-only access.

## Validation

`scripts/export_onnx.py --reference` exports the model and records PyTorch's
probability for every triplet in `EBI_TEST_SET` and `NOT_EBI_TEST_SET`.
`dev/validate.html` then classifies the same files in the browser and reports
the largest difference:

    python3 web/serve.py --root .
    # open http://127.0.0.1:8765/web/dev/validate.html

`reference.json` is produced from your own test images and is not stored in
git. On the project's 465 test triplets the browser gave the same label as
PyTorch for every sample, with a largest probability difference of 5e-6.

Repeat this whenever the model, `ebi-core.js` or onnxruntime-web changes.

## Differences from the desktop application

- Reads uncompressed, single-channel TIFF files (8, 16 or 32-bit), which is what
  ImageStream exports. Compressed, tiled or BigTIFF files are refused with a
  message instead of being read.
- Runs on the CPU through WebAssembly. Classification is slower while the tab is
  in the background, because browsers give background tabs less CPU time.
- The model is downloaded when the page opens (the browser then caches it), so
  the first visit needs a connection fast enough for 112 MB.
- Results exist only in the open tab. Closing or reloading the page discards
  them; download the `.pop` file first.
