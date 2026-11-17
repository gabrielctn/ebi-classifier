#!/usr/bin/env python3
"""Export the release model to ONNX for the browser-only classifier in web/.

Writes web/model/ebi.onnx and web/model/config.json. With --reference it also
writes web/dev/reference.json (PyTorch probabilities for the local test sets)
and web/dev/resize_case.json, which web/dev/validate.html uses to prove that
the browser reproduces the Python pipeline.

Needs onnx, onnxscript and onnxruntime in addition to the development
environment, for example:

    uv run --no-project --python .venv-secure/bin/python \
        --with onnx --with onnxscript --with onnxruntime \
        scripts/export_onnx.py --reference
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import timm  # noqa: E402

from dataset import _parse_channel_id, _parse_sample_id, letterbox_resize_pad  # noqa: E402
import app  # noqa: E402

DEFAULT_MODEL = ROOT / "runs/20260109-182504/final_model/checkpoints/final.pt"
# app.py falls back to this value when the checkpoint has no val_metrics.
APP_DEFAULT_THRESHOLD = 0.67


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(source: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = timm.create_model(config["model"], pretrained=False, num_classes=2,
                              drop_path_rate=config.get("drop_path_rate", 0.2))
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), config


def triplets(folder: Path, channels: tuple[int, int, int]) -> list[tuple[str, dict[int, Path]]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(folder.glob("*.tif")):
        channel = _parse_channel_id(path.name)
        if channel in channels:
            groups.setdefault(_parse_sample_id(path.name), {})[channel] = path
    return [(sid, chans) for sid, chans in sorted(groups.items()) if set(chans) == set(channels)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "web")
    parser.add_argument("--reference", action="store_true",
                        help="Also write the validation references from the local test sets")
    args = parser.parse_args()

    model, config = load(args.model)
    size = int(config.get("image_size", 224))
    model_dir = args.output / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "ebi.onnx"

    torch.onnx.export(
        model, (torch.zeros(1, 3, size, size),), str(onnx_path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    threshold = config.get("val_metrics", {}).get("threshold", APP_DEFAULT_THRESHOLD)
    if not isinstance(threshold, float):
        threshold = APP_DEFAULT_THRESHOLD
    channels = app.TRAINING_RGB_ORDER
    (model_dir / "config.json").write_text(json.dumps({
        "model": config["model"],
        "image_size": size,
        "intensity_norm": config.get("intensity_norm", "percentile"),
        "percentile_low": config.get("percentile_low", 1.0),
        "percentile_high": config.get("percentile_high", 99.0),
        "threshold": threshold,
        "default_channels": {"erythroblasts": channels[0], "macrophages": channels[1],
                             "neutrophils": channels[2]},
        "onnx_sha256": sha256(onnx_path),
        "source_checkpoint_sha256": sha256(args.model),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if not args.reference:
        probe = torch.rand(2, 3, size, size)
        with torch.no_grad():
            expected = model(probe).numpy()
        got = session.run(None, {"image": probe.numpy()})[0]
        print(f"ONNX vs PyTorch, max logit difference: {np.abs(expected - got).max():.2e}")
        return

    dev = args.output / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    samples, worst = [], 0.0
    for folder in ("EBI_TEST_SET", "NOT_EBI_TEST_SET"):
        for sid, chans in triplets(ROOT / folder, channels):
            image = app.prepare_image(
                chans, rgb_order=channels, image_size=size,
                intensity_norm=config.get("intensity_norm", "percentile"),
                percentile_low=config.get("percentile_low", 1.0),
                percentile_high=config.get("percentile_high", 99.0),
            ).unsqueeze(0)
            with torch.no_grad():
                logits = model(image)
            prob = float(torch.softmax(logits, dim=1)[0, 1])
            onnx_logits = torch.from_numpy(session.run(None, {"image": image.numpy()})[0])
            worst = max(worst, abs(prob - float(torch.softmax(onnx_logits, dim=1)[0, 1])))
            samples.append({
                "folder": folder, "id": sid, "prob": prob,
                "files": [chans[c].name for c in channels],
            })
    (dev / "reference.json").write_text(json.dumps({
        "threshold": threshold, "channels": list(channels), "samples": samples,
    }) + "\n", encoding="utf-8")
    print(f"{len(samples)} reference samples; ONNX vs PyTorch max probability difference: {worst:.2e}")

    # A synthetic image larger than the model input exercises the antialiased
    # downscaling path, which the 120-pixel-wide ImageStream crops never reach.
    height, width = 300, 260
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    pattern = (((yy * 7 + xx * 13 + (yy * xx) % 31) % 256) / 255.0).astype(np.float32)
    resized = letterbox_resize_pad(torch.from_numpy(pattern)[None], size=size)[0]
    (dev / "resize_case.json").write_text(json.dumps({
        "height": height, "width": width, "size": size,
        "expected": [round(float(v), 6) for v in resized.flatten()],
    }) + "\n", encoding="utf-8")
    print("Wrote validation references to", dev)


if __name__ == "__main__":
    main()
