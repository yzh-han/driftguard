#!/usr/bin/env python3

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


# get all dataset directories under datasets_root
def iter_datasets(datasets_root: Path) -> Iterable[Path]:
    for path in sorted(datasets_root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            yield path


# get all image files under a class directory
def iter_image_files(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


# build metadata for a dataset directory
def build_meta(dataset_dir: Path) -> dict | None:
    # domain -> list of (relative file path, label)
    domain_to_label_files: dict[str, list[tuple[str, str]]] = {}
    labels: set[str] = set()

    domain_to_label_count: dict[str, dict[str, int]] = {}

    # traverse domain/class/image structure
    for domain_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        domain = domain_dir.name
        for class_dir in sorted(p for p in domain_dir.iterdir() if p.is_dir()):
            label = str(class_dir.name)
            for image_path in iter_image_files(class_dir):
                rel_path = image_path.relative_to(dataset_dir).as_posix()
                domain_to_label_files.setdefault(domain, []).append((rel_path, label))
                labels.add(label)

                # domain -> label -> count
                domain_to_label_count.setdefault(domain, {}).setdefault(label, 0)
                domain_to_label_count[domain][label] += 1

    if not domain_to_label_files:
        return None

    label_list = sorted(labels)
    label_to_idx = {label: idx for idx, label in enumerate(label_list)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    domain_to_files: dict[str, list[tuple[str, int]]] = {}

    for domain, files in domain_to_label_files.items():
        domain_to_files[domain] = [
            (path, label_to_idx[label]) for path, label in files
        ]

    return {
        # list of domains
        "domains": sorted(domain_to_files.keys()),
        # list of labels
        "labels": label_list,
        # label -> idx
        "label_to_idx": label_to_idx,
        # idx -> label
        "idx_to_label": idx_to_label,
        # domain -> list of [relative file path, label idx]
        "domain_to_files": domain_to_files,
        # domain -> label -> count
        "domain_to_label_count": domain_to_label_count,
    }


def write_meta(dataset_dir: Path, meta: dict) -> Path:
    meta_path = dataset_dir / "_meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    )
    return meta_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build _meta.json for each dataset under datasets/."
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("datasets"),
        help="Root directory containing datasets (default: datasets).",
    )
    args = parser.parse_args()

    datasets_root = args.datasets_root
    if not datasets_root.exists():
        raise SystemExit(f"datasets root not found: {datasets_root}")

    updated = 0
    skipped = 0
    for dataset_dir in iter_datasets(datasets_root):
        meta = build_meta(dataset_dir)
        if meta is None:
            skipped += 1
            continue
        meta_path = write_meta(dataset_dir, meta)
        updated += 1
        print(f"wrote {meta_path}")

    print(f"done: {updated} datasets updated, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
