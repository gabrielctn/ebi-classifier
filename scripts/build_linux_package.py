#!/usr/bin/env python3
"""Build an offline Linux inference ZIP for x86-64 and ARM64 computers."""

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
    "x86_64": {
        "python_url": PYTHON_RELEASE + "cpython-3.14.10%2B20261026-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "python_sha": "4d3c16c59ed0802a3d1d34d8bce504b15ceeaff39d62dadecc6a9fa803412aed",
        "uv_url": "https://files.pythonhosted.org/packages/b9/93/0e053a2ca4523ee9825ccbedeb7121470d0a188933f11688b6d90d7f5c08/uv-0.13.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "uv_sha": "2c9bdc4a872b60a93c5f26c0791089ea11585a4f9d3b3c017d78eb8487505e25",
    },
    "aarch64": {
        "python_url": PYTHON_RELEASE + "cpython-3.14.10%2B20261026-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz",
        "python_sha": "90f48337bc3dd4ae6eda81e001e4c981236d403b6b5a07778bbdf94ee423dc00",
        "uv_url": "https://files.pythonhosted.org/packages/96/72/da1119f94034d7173e6c8e2cd77a6299744458f0af3cb8e3a04d5cd419d1/uv-0.13.7-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl",
        "uv_sha": "a90f5d95716ab83091f0f73a8e7ec64e557abf2a28537fcd719b2ac88e6f725a",
    },
}


def prepare_wheels(cache: Path, destination: Path, arch: str) -> None:
    """Download exactly the locked wheels, compatible with glibc 2.28."""
    cache.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True)
    # pip's --platform does not automatically add all older manylinux tags.
    tags = [f"manylinux_2_{minor}_{arch}" for minor in range(28, 4, -1)]
    tags += [f"manylinux2014_{arch}"]
    if arch == "x86_64":
        tags += [f"manylinux2010_{arch}", f"manylinux1_{arch}"]
    command = [
        sys.executable, "-m", "pip", "download", "--disable-pip-version-check",
        "--no-deps", "--only-binary=:all:", "--python-version", "314",
        "--implementation", "cp", "--abi", "cp314",
        "-r", str(ROOT / "linux/requirements.txt"), "-d", str(destination),
    ]
    for tag in tags:
        command += ["--platform", tag]
    cached = subprocess.run(command + ["--no-index", "--find-links", str(cache)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cached.returncode:
        subprocess.run(command + ["--find-links", str(cache), "--extra-index-url",
                                  "https://download.pytorch.org/whl/cpu"], check=True)
    for wheel in destination.glob("*.whl"):
        cached_wheel = cache / wheel.name
        if cached_wheel.exists() and checksum(cached_wheel) != checksum(wheel):
            raise ValueError(f"Conflicting cached wheel: {wheel.name}")
        if not cached_wheel.exists():
            shutil.copy2(wheel, cached_wheel)


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for arch, options in TARGETS.items():
        fetch(options["python_url"], args.cache_dir / f"python-{arch}.tar.gz", options["python_sha"])
        fetch(options["uv_url"], args.cache_dir / f"uv-{arch}.whl", options["uv_sha"])

    with tempfile.TemporaryDirectory(prefix="ebi-linux-") as temp:
        temp_dir = Path(temp)
        package = temp_dir / "EBI_Classifier_Linux"
        wheels = package / "runtime/wheels"
        wheels.mkdir(parents=True)
        for arch in TARGETS:
            runtime = package / "runtime" / arch
            runtime.mkdir()
            with tarfile.open(args.cache_dir / f"python-{arch}.tar.gz", "r:gz") as archive:
                # The bundled terminfo database contains names that differ only
                # by case, which collide on common Mac build filesystems. The
                # classifier does not use curses; Linux provides its own database.
                members = (item for item in archive
                           if item.name != "python/share/terminfo"
                           and not item.name.startswith("python/share/terminfo/"))
                archive.extractall(runtime, members=members, filter="data")
            strip_installer_tools(runtime)
            with zipfile.ZipFile(args.cache_dir / f"uv-{arch}.whl") as archive:
                uv = runtime / "uv"
                uv.write_bytes(archive.read(f"uv-{UV_VERSION}.data/scripts/uv"))
                uv.chmod(0o755)
                licenses = runtime / "uv-licenses"
                licenses.mkdir()
                for name in ("LICENSE-APACHE", "LICENSE-MIT"):
                    (licenses / name).write_bytes(
                        archive.read(f"uv-{UV_VERSION}.dist-info/licenses/{name}")
                    )
            selected = temp_dir / f"selected-{arch}"
            prepare_wheels(args.cache_dir / f"wheels-{arch}", selected, arch)
            for wheel in selected.glob("*.whl"):
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
            (ROOT / "linux/START_EBI.sh", package / "START_EBI.sh"),
            (ROOT / "linux/EBI_Classifier.desktop", package / "EBI_Classifier.desktop"),
            (ROOT / "linux/START_HERE.txt", package / "START_HERE.txt"),
            (ROOT / "linux/requirements.txt", package / "linux/requirements.txt"),
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
    parser.add_argument("--output", type=Path, default=ROOT / "EBI_Classifier_Linux.zip")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(tempfile.gettempdir()) / "ebi-linux-downloads")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.cache_dir = args.cache_dir.resolve()
    build(args)


if __name__ == "__main__":
    main()
