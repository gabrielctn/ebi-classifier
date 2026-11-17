#!/usr/bin/env python3
"""Build the offline Windows inference ZIP with torch, safetensors and pip installed."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file as save_safetensors

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "runs/20260109-182504/final_model/checkpoints/final.pt"
PYTHON_URL = (
    "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
    "cpython-3.14.10%2B20261026-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
)
PYTHON_SHA256 = "c1876a9795d3a94ee49a1499a69cd91e26ca3e2829593f93fa6e4db90d05402b"
UV_VERSION = "0.13.7"
UV_URL = (
    "https://files.pythonhosted.org/packages/8e/4f/041b59d2a38807a3e31ef41064cf3e45d2752fe0409d89040d52778049ae/"
    "uv-0.13.7-py3-none-win_amd64.whl"
)
UV_SHA256 = "2835e2ab397d0db86323b9975dce75ecd76aa9cfa0d6a2aea84734aae7f8c9b3"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path, expected: str) -> None:
    if target.exists() and checksum(target) == expected:
        return
    print(f"Downloading {target.name}...", flush=True)
    temporary = target.with_suffix(target.suffix + ".download")
    try:
        with urllib.request.urlopen(url) as source, temporary.open("wb") as dest:
            shutil.copyfileobj(source, dest)
        if checksum(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch for {target.name}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def strip_installer_tools(runtime: Path) -> None:
    """Remove pip and setuptools from an extracted Python runtime.

    The launchers install the bundled wheels with uv, so the interpreter's own
    pip is never used. Removing it also removes pip's vendored copies of
    certifi, urllib3 and other libraries, which would otherwise be flagged by
    file scanners independently of the versions the application uses.
    """
    names = ("pip", "setuptools", "pkg_resources", "_distutils_hack")
    for site in runtime.rglob("site-packages"):
        for entry in site.iterdir():
            base = entry.name.split("-")[0].lower()
            if base in names and (entry.is_dir() or entry.suffix == ".dist-info"):
                shutil.rmtree(entry)
            elif entry.name == "distutils-precedence.pth":
                entry.unlink()
    for bundled in runtime.rglob("ensurepip/_bundled/*.whl"):
        bundled.unlink()
    for folder in ("bin", "Scripts"):
        for script in runtime.glob(f"python/{folder}/pip*"):
            script.unlink()
    leftovers = [str(p.relative_to(runtime)) for p in runtime.rglob("*")
                 if p.name == "cacert.pem" or p.name.lower().startswith("pip-")]
    if leftovers:
        raise ValueError(f"Installer files remain in the runtime: {leftovers}")


def prepare_wheels(cache: Path, requirements: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
        "--no-deps", "--only-binary=:all:", "--platform", "win_amd64",
        "--python-version", "314", "--implementation", "cp", "--abi", "cp314",
        "-r", str(requirements), "-d", str(cache),
    ]
    # Reuse a complete wheel cache without network access where possible.
    cached = subprocess.run(command + ["--no-index", "--find-links", str(cache)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cached.returncode:
        subprocess.run(command + ["--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                       check=True)


def create_release_model(source: Path, package: Path) -> None:
    # The input is the trusted training checkpoint. The distributed model uses
    # safetensors and JSON, so the recipient never needs to unpickle a checkpoint.
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state and config")
    model_path = package / "runs/release/final.safetensors"
    model_path.parent.mkdir(parents=True)
    tensors = {name: tensor.contiguous() for name, tensor in checkpoint["model_state"].items()}
    save_safetensors(tensors, str(model_path))
    model_path.with_suffix(".json").write_text(
        json.dumps(checkpoint["config"], indent=2) + "\n", encoding="utf-8"
    )
    (package / "MODEL_INFO.txt").write_text(
        "Inference checkpoint source: " + str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source.name)
        + "\nSource SHA-256: " + checksum(source)
        + "\nModel weights are distributed in safetensors format; configuration is JSON.\n",
        encoding="utf-8",
    )


def write_third_party_notices(wheels: Path, package: Path) -> None:
    """Expose wheel license metadata and accompanying notices for IT review."""
    notices = package / "THIRD_PARTY_LICENSES"
    notices.mkdir()
    lines = [
        "# Bundled Python packages", "",
        "These entries come from the bundled wheel metadata. License files "
        "found in each wheel's dist-info directory are copied into this folder. "
        "Review each distribution and its bundled native components separately.", "",
    ]
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            metadata_names = [name for name in archive.namelist()
                              if name.endswith(".dist-info/METADATA") and name.count("/") == 1]
            if len(metadata_names) != 1:
                raise ValueError(f"Missing or ambiguous wheel metadata: {wheel.name}")
            metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]), headersonly=True)
            license_name = str(metadata.get("License-Expression") or metadata.get("License") or "Not declared in metadata")
            if "\n" in license_name or len(license_name) > 200:
                license_name = "See accompanying license files and wheel metadata"
            lines.append(f"- {wheel.name}: {license_name.strip()}")
            dist_info = metadata_names[0].rsplit("/", 1)[0] + "/"
            for name in archive.namelist():
                if not name.startswith(dist_info) or name.endswith("/"):
                    continue
                relative = Path(name[len(dist_info):])
                if ".." in relative.parts:
                    raise ValueError(f"Unsafe wheel license path: {name}")
                if "licenses" not in relative.parts and not relative.name.upper().startswith(
                    ("LICENSE", "LICENCE", "COPYING", "NOTICE", "AUTHORS")
                ):
                    continue
                target = notices / wheel.stem / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    (package / "THIRD_PARTY_PACKAGES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    python_archive = args.cache_dir / "python.tar.gz"
    uv_wheel = args.cache_dir / f"uv-{UV_VERSION}-py3-none-win_amd64.whl"
    fetch(PYTHON_URL, python_archive, PYTHON_SHA256)
    fetch(UV_URL, uv_wheel, UV_SHA256)
    wheels = args.cache_dir / "wheels"
    prepare_wheels(wheels, ROOT / "windows/requirements.txt")

    with tempfile.TemporaryDirectory(prefix="ebi-windows-") as temp:
        package = Path(temp) / "EBI_Classifier_Windows"
        runtime = package / "runtime"
        runtime.mkdir(parents=True)
        with tarfile.open(python_archive, "r:gz") as archive:
            archive.extractall(runtime, filter="data")
        strip_installer_tools(runtime)
        with zipfile.ZipFile(uv_wheel) as archive:
            (runtime / "uv.exe").write_bytes(
                archive.read(f"uv-{UV_VERSION}.data/scripts/uv.exe")
            )
            licenses = runtime / "uv-licenses"
            licenses.mkdir()
            for name in ("LICENSE-APACHE", "LICENSE-MIT"):
                (licenses / name).write_bytes(
                    archive.read(f"uv-{UV_VERSION}.dist-info/licenses/{name}")
                )
        shutil.copytree(wheels, runtime / "wheels")
        for source, destination in (
            (ROOT / "app.py", package / "app.py"),
            (ROOT / "dataset.py", package / "dataset.py"),
            (ROOT / "GUIDE_UTILISATEUR.md", package / "GUIDE_UTILISATEUR.md"),
            (ROOT / "IT_INFORMATION.md", package / "IT_INFORMATION.md"),
            (ROOT / "windows/START_EBI.bat", package / "START_EBI.bat"),
            (ROOT / "windows/START_HERE.txt", package / "START_HERE.txt"),
            (ROOT / "windows/requirements.txt", package / "windows/requirements.txt"),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        create_release_model(args.model, package)
        write_third_party_notices(runtime / "wheels", package)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".zip.tmp", delete=False) as tmp:
            temporary_zip = Path(tmp.name)
        try:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for path in sorted(package.rglob("*")):
                    if path.is_file():
                        # Wheels are already compressed.
                        method = zipfile.ZIP_STORED if path.suffix == ".whl" else zipfile.ZIP_DEFLATED
                        archive.write(path, path.relative_to(package.parent), compress_type=method)
            temporary_zip.replace(args.output)
        finally:
            temporary_zip.unlink(missing_ok=True)
    print(f"Created {args.output} ({args.output.stat().st_size / (1024 * 1024):.1f} MiB)")
    print(f"SHA-256: {checksum(args.output)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="Trusted trained checkpoint (defaults to the existing V1 release model)")
    parser.add_argument("--output", type=Path, default=ROOT / "EBI_Classifier_Windows.zip")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "ebi-windows-downloads")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    build(args)


if __name__ == "__main__":
    main()
