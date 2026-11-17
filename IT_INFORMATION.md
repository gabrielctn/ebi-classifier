# EBI Classifier: information for IT and legal review

Prepared 17 November 2026 for the updated Windows, Mac and Linux offline packages.
This document describes a technical review; it is not a legal determination,
a security certification, or an approval for clinical use.

## Purpose and operation

The EBI Classifier is a research tool that classifies three-channel ImageStream
OME-TIFF images and exports sample identifiers as a `.pop` file. The packages
contain the inference code, trained model, Python interpreter, installer and
Python dependencies. They do not contain the project's training or test images.
Scientific suitability for the recipient's experiment must be evaluated. The
workspace does not establish clinical validation or regulatory clearance.

The launcher installs the bundled wheels without reaching an online package
index. The interface binds to `127.0.0.1`, Gradio public sharing is disabled,
and Gradio analytics and Hugging Face online access are disabled by the
launcher. Normal inference is intended to work offline. A browser is required.
Some installations may require an operating-system runtime or IT approval.

Images and results may be written to local temporary folders, including Gradio
upload caches and a temporary `.pop` output. They are not guaranteed to be
automatically erased on exit. IT should apply its usual data handling,
retention and deletion rules, especially if files or identifiers contain
patient or other sensitive information. Localhost binding is not an access
control boundary against other processes on the same computer.

## Project and model licensing

- **Application code:** released under the MIT licence (see `LICENSE` in the
  source repository, https://github.com/gabrielctn/ebi-classifier). Packages
  built before 17 November 2026 predate this licence and carry an earlier
  version of this paragraph.
- **Model provenance:** the recorded training configuration uses pretrained
  `convnextv2_tiny.fcmae_ft_in22k_in1k` weights (`no_pretrained: false`), which
  were subsequently trained on EBI images. The upstream model card identifies
  that pretrained model as **CC BY-NC 4.0**:
  https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k
  The licence requires attribution and restricts commercial purposes:
  https://creativecommons.org/licenses/by-nc/4.0/legalcode.en
  The implications for these fine-tuned weights and the recipient's intended
  activities need legal review. A nonprofit or research setting is not by
  itself a determination that a particular use is noncommercial. The
  fine-tuned weights are distributed under the same CC BY-NC 4.0 licence.
- **Training material:** rights, consent, institutional agreements and any
  restrictions associated with the training images or funding have not been
  verified in this review. No training images are included in the distribution.
- **Dependencies:** see `THIRD_PARTY_PACKAGES.md`, `THIRD_PARTY_LICENSES/`, the
  wheels in `runtime/wheels/`, and the Python and uv licence files inside
  `runtime/`. The inventory is assembled from wheel metadata and available
  notices; it is a starting point, not a completed audit of all transitive or
  bundled native-code licences. Different components retain their own terms.

## Python versions and known security limitations

The updated package sets were prepared against published releases available on
17 November 2026:

| Target | Python | Important version constraint |
| --- | --- | --- |
| 64-bit Windows | 3.14.10 | Python 3.15 has no compatible Windows wheel for a required dependency, PyYAML, in the current resolved stack. |
| Apple Silicon, macOS 13+ | 3.14.10 | PyTorch 2.11.0 is retained for macOS 13 compatibility. |
| Intel Mac, macOS 13+ | 3.11.18 | PyTorch 2.2.2 is the last available Intel Mac wheel series; it has no newer-Python wheel. |
| 64-bit Intel/AMD or ARM Linux, glibc 2.28+ | 3.14.10 | The package includes CPU-only PyTorch 2.15.0; its wheel requires glibc 2.28 or newer. |

These are the latest security releases of their respective Python branches as
of the review date. Python 3.11.18 is a security-only branch; upstream support
is scheduled to end in October 2027. Python's 26 October security release
details include fixes for CVE-2026-115557, CVE-2026-114583 and other issues:
https://blog.python.org/2026/10/python-3152-31410-31318-31216-31118/

The updated package sets include Brotli 1.2.1, which fixes CVE-2026-111497,
and certifi 2026.10.23, which removes the affected eMudhra/emSign root
certificate. The previous October ZIPs included Brotli 1.2.0 and certifi
2026.7.22. Relevant advisories:
https://github.com/advisories/GHSA-9v8h-qqgm-4m35
https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2

Changes since the 3 November 2026 ZIPs:

- Gradio is updated from 6.31.0 to 6.32.0, which accepts Pillow 13. Pillow is
  therefore updated from 12.3.0 to 13.0.0, which fixes the published Pillow
  advisories that the 3 November packages retained.
  https://github.com/python-pillow/Pillow/security/advisories
- pip and setuptools are removed from the bundled Python interpreter. The
  launcher installs the bundled wheels with uv and never used them. The
  interpreter's pip 26.2.2 carried its own copy of certifi 2026.10.19, which
  still contained the affected emSign root certificate; that copy is no longer
  present. The only certifi in the package is the 2026.10.23 wheel.

**Residual findings require IT review:**

- Apple Silicon's PyTorch 2.11.0 has an advisory involving `torch.jit.script`.
  This app does not explicitly call that function. Intel Mac's PyTorch 2.2.2
  has additional advisories, including checkpoint deserialization issues.
  The updated inference bundle contains model weights in safetensors and JSON
  instead of a pickle-based PyTorch checkpoint, reducing exposure to checkpoint
  deserialization. Do not load untrusted replacement models. This does not
  remove other vulnerabilities in the older PyTorch library.
  https://github.com/advisories/GHSA-rrmf-rvhw-rf47
  https://github.com/advisories/GHSA-53q9-r3pm-6pq6
- The Apple Silicon PyTorch requirement also constrains setuptools to 81.0.0,
  which has published advisories concerning source distribution handling.
  The user package installs prebuilt wheels and does not build source packages
  during normal setup.

An OSV query of the explicitly pinned Python package/version pairs was performed
on 17 November 2026. For the Windows and Linux packages it reported no
advisories; the Mac findings are those listed above. This is a point-in-time dependency check, not a complete
assessment of all code paths, bundled native libraries, operating-system
components or future advisories. Verify the actual package checksum and repeat
the review before distribution or subsequent use.
