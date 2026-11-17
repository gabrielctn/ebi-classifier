"""Add corrected objects to the training folders.

A user of the classifier who disagrees with some predictions moves those
objects to the other group in the review step and downloads
`EBI_corrections.csv`. They send that file together with the TIFF files of the
corrected objects. This script copies each corrected triplet into `EBI/` or
`NOT_EBI/` according to the corrected class, ready for `train.py`.

ImageStream numbers objects from zero in every acquisition, so the same object
ID appears in many experiments. Every copied file name is therefore given a
prefix (`--prefix`) that names the batch, for example `lab2-2026-11`, so
`36100_Ch7.ome.tif` becomes `lab2-2026-11-36100_Ch7.ome.tif`. Use a new prefix
for every batch you receive.

Nothing is overwritten. Run with `--dry-run` first to see what would be copied.

    python scripts/import_corrections.py EBI_corrections.csv \\
        --images /path/to/received/tiffs --prefix lab2-2026-11 --dry-run

Only the Python standard library is needed.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

CLASS_FOLDERS = {"EBI": "EBI", "NOT EBI": "NOT_EBI", "NOT_EBI": "NOT_EBI"}
PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")


def read_corrections(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"object_id", "corrected_class", "files"}
    if not rows:
        raise SystemExit(f"{path}: no corrections in this file.")
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"{path}: missing column(s) {', '.join(sorted(missing))}. Is this an EBI_corrections.csv?")
    return rows


def index_images(folder: Path) -> dict[str, list[Path]]:
    """Map every TIFF file name below `folder` to its path(s)."""
    found: dict[str, list[Path]] = {}
    for path in folder.rglob("*"):
        if path.is_file() and path.name.lower().endswith((".tif", ".tiff")):
            found.setdefault(path.name, []).append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corrections", type=Path, help="EBI_corrections.csv downloaded from the classifier")
    parser.add_argument("--images", type=Path, required=True, help="Folder holding the TIFF files that were sent (searched recursively)")
    parser.add_argument("--prefix", required=True, help="Batch name added in front of every file name, e.g. lab2-2026-11")
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Folder containing EBI/ and NOT_EBI/ (default: current folder)")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be copied")
    args = parser.parse_args()

    if not PREFIX_RE.match(args.prefix) or "_ch" in args.prefix.lower():
        parser.error("--prefix may only contain letters, digits, '.' and '-', and must not contain '_Ch'.")
    if not args.images.is_dir():
        parser.error(f"--images: {args.images} is not a folder.")

    rows = read_corrections(args.corrections)
    images = index_images(args.images)
    plan: list[tuple[Path, Path]] = []
    problems: list[str] = []
    counts = {"EBI": 0, "NOT_EBI": 0}

    for line, row in enumerate(rows, start=2):
        object_id = row["object_id"].strip()
        folder = CLASS_FOLDERS.get(row["corrected_class"].strip().upper())
        if folder is None:
            problems.append(f"line {line} ({object_id}): unknown class {row['corrected_class']!r}")
            continue
        names = [name.strip() for name in row["files"].split(";") if name.strip()]
        if len(names) != 3:
            problems.append(f"line {line} ({object_id}): expected 3 file names, found {len(names)}")
            continue
        sources = []
        for name in names:
            matches = images.get(name, [])
            if len(matches) != 1:
                what = "not found" if not matches else f"found {len(matches)} times; keep one batch per folder"
                problems.append(f"line {line} ({object_id}): {name} {what}")
                break
            sources.append(matches[0])
        else:
            for source in sources:
                target = args.data_root / folder / f"{args.prefix}-{source.name}"
                if target.exists():
                    problems.append(f"line {line} ({object_id}): {target} already exists; use another --prefix")
                    break
            else:
                plan.extend((source, args.data_root / folder / f"{args.prefix}-{source.name}") for source in sources)
                counts[folder] += 1

    for message in problems:
        print(f"Skipped: {message}", file=sys.stderr)
    for source, target in plan:
        print(f"{'Would copy' if args.dry_run else 'Copy'} {source} -> {target}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    verb = "would be added" if args.dry_run else "added"
    print(f"\n{counts['EBI']} triplet(s) {verb} to EBI/, {counts['NOT_EBI']} to NOT_EBI/. {len(problems)} skipped.")
    if plan and not args.dry_run:
        print("Retrain with train.py, then compare the new model with the current one on "
              "EBI_TEST_SET/ and NOT_EBI_TEST_SET/ before publishing it.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
