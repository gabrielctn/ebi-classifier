#!/usr/bin/env python3
"""Build an offline inference ZIP for Apple Silicon and Intel Macs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from build_windows_package import (
    DEFAULT_MODEL, ROOT, UV_VERSION, checksum, create_release_model, fetch,
    strip_installer_tools, write_third_party_notices,
)

PYTHON_RELEASE = "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/"
TARGETS = {
    "arm64": {
        "python_url": PYTHON_RELEASE + "cpython-3.14.10%2B20261026-aarch64-apple-darwin-install_only_stripped.tar.gz",
        "python_sha": "95eb8c87d16af37c54e9765e31d8c56c0d8ffa9968ea0f704f73a9b83f84e9f9",
        "python_version": "314",
        "uv_url": "https://files.pythonhosted.org/packages/40/a5/8756c38ace102fa13fd4e653e8bc40e1f65924f1c0a0d23350339dac97e5/uv-0.13.7-py3-none-macosx_11_0_arm64.whl",
        "uv_sha": "f91a229093163eefca89ea1b220a221d64c3beade05d03519a6584d1797a5a5f",
        "pip_platform": "macosx_13_0_arm64",
    },
    "intel": {
        "python_url": PYTHON_RELEASE + "cpython-3.11.18%2B20261026-x86_64-apple-darwin-install_only_stripped.tar.gz",
        "python_sha": "d38edc2df25e7e7f886c802bf2fa552422b097c509dbdcbae828827470356095",
        "python_version": "311",
        "uv_url": "https://files.pythonhosted.org/packages/3d/5b/e0379e7b172659420b1ec99a8e8229830aa85ea59684847c80b1c959a056/uv-0.13.7-py3-none-macosx_10_12_x86_64.whl",
        "uv_sha": "fd04bf01ba5f2829f4f94ca8f0c24a743f21e73d75045613ed6fc2fd66fc7bda",
        "pip_platform": "macosx_13_0_x86_64",
    },
}


def prepare_wheels(cache: Path, requirements: Path, platform: str, python_version: str) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
        "--no-deps", "--only-binary=:all:", "--platform", platform,
        "--python-version", python_version, "--implementation", "cp", "--abi", f"cp{python_version}",
        "-r", str(requirements), "-d", str(cache),
    ]
    cached = subprocess.run(command + ["--no-index", "--find-links", str(cache)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cached.returncode:
        subprocess.run(command, check=True)


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for platform, options in TARGETS.items():
        fetch(options["python_url"], args.cache_dir / f"python-{platform}.tar.gz", options["python_sha"])
        fetch(options["uv_url"], args.cache_dir / f"uv-{platform}.whl", options["uv_sha"])
        prepare_wheels(args.cache_dir / f"wheels-{platform}",
                       ROOT / f"macos/requirements-{platform}.txt", options["pip_platform"],
                       options["python_version"])

    with tempfile.TemporaryDirectory(prefix="ebi-mac-") as temp:
        package = Path(temp) / "EBI_Classifier_Mac"
        wheels = package / "runtime/wheels"
        wheels.mkdir(parents=True)
        for platform in TARGETS:
            runtime = package / "runtime" / platform
            runtime.mkdir()
            with tarfile.open(args.cache_dir / f"python-{platform}.tar.gz", "r:gz") as archive:
                archive.extractall(runtime, filter="data")
            strip_installer_tools(runtime)
            with zipfile.ZipFile(args.cache_dir / f"uv-{platform}.whl") as archive:
                uv = runtime / "uv"
                uv.write_bytes(archive.read(f"uv-{UV_VERSION}.data/scripts/uv"))
                uv.chmod(0o755)
                licenses = runtime / "uv-licenses"
                licenses.mkdir()
                for name in ("LICENSE-APACHE", "LICENSE-MIT"):
                    (licenses / name).write_bytes(
                        archive.read(f"uv-{UV_VERSION}.dist-info/licenses/{name}")
                    )
            for wheel in (args.cache_dir / f"wheels-{platform}").glob("*.whl"):
                destination = wheels / wheel.name
                if destination.exists():
                    if checksum(wheel) != checksum(destination):
                        raise ValueError(f"Conflicting wheels named {wheel.name}")
                else:
                    shutil.copy2(wheel, destination)
        for source, destination in (
            (ROOT / "app.py", package / "app.py"),
            (ROOT / "dataset.py", package / "dataset.py"),
            (ROOT / "GUIDE_UTILISATEUR.md", package / "GUIDE_UTILISATEUR.md"),
            (ROOT / "IT_INFORMATION.md", package / "IT_INFORMATION.md"),
            (ROOT / "macos/START_EBI.command", package / "START_EBI.command"),
            (ROOT / "macos/START_HERE.txt", package / "START_HERE.txt"),
            (ROOT / "macos/requirements-arm64.txt", package / "macos/requirements-arm64.txt"),
            (ROOT / "macos/requirements-intel.txt", package / "macos/requirements-intel.txt"),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        create_release_model(args.model, package)
        write_third_party_notices(wheels, package)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, suffix=".zip.tmp", delete=False) as tmp:
            temporary_zip = Path(tmp.name)
        try:
            with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED,
                                 compresslevel=6, allowZip64=True) as archive:
                for path in sorted(package.rglob("*")):
                    if path.is_file():
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
    parser.add_argument("--output", type=Path, default=ROOT / "EBI_Classifier_Mac.zip")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "ebi-mac-downloads")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    build(args)


if __name__ == "__main__":
    main()
