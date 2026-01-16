# Datasets Guidelines

## Directory Layout
- Each dataset lives in its own folder under `datasets/`.
- Required structure: `datasets/<dataset_name>/<domain>/<class>/*.jpg`

## Metadata
- Run `python scripts/build_dataset_meta.py` from the repo root.
- This generates `datasets/<dataset_name>/_meta.json` with domains, labels,
  and `domain_to_files` (paths are relative to the dataset root).

## Notes
- Keep filenames and folder names ASCII when possible.
- Avoid committing large or sensitive datasets unless approved.
