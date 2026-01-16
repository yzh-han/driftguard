"""Domain-level dataset reader backed by precomputed metadata."""

import json
from collections import deque
from collections.abc import Callable
from pathlib import Path
import random


class DomainDataset:
    """Serve samples by domain using precomputed metadata.

    Attributes:
        meta_path: Path to the dataset metadata file.
        dataset_root: Root directory of the dataset (parent of meta_path).
        buffer_size: Number of samples buffered per domain.
        transform: Optional callable applied to raw bytes before returning.
        domains: Ordered list of available domains.
        domain_to_files: Mapping of domain -> deque of (relative path, label idx).
    """

    def __init__(
        self,
        meta_path: Path | str,
        transform: Callable[[bytes], bytes] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the dataset with metadata and buffering.

        Args:
            meta_path: Path to `datasets/<name>/_meta.json`.
            buffer_size: Samples buffered per domain for lightweight sampling.
            transform: Optional transform applied to raw bytes.
        """
        self.meta_path = Path(meta_path)
        self.dataset_root = self.meta_path.parent
        self.transform = transform

        rng = random.Random(seed)
        meta = self._load_meta(self.meta_path)
        self.domains: list[str] = list(meta["domains"])
        domain_to_files: dict[str, list[tuple[Path, int]]] \
            = {domain: [(self.dataset_root / rel_path, label_idx) 
                 for rel_path, label_idx in entries]
               for domain, entries in meta["domain_to_files"].items()}
        for entries in domain_to_files.values():
            rng.shuffle(entries)
        """Mapping of domain -> list of (path, label idx)."""
        self.buffer: dict[str, deque[tuple[Path, int]]] \
            = {domain: deque(entries) for domain, entries in domain_to_files.items()}
        """Mapping of domain -> deque of (path, label idx)."""


    def get_one(self, domain: str) -> tuple[bytes, int]:
        """Return a single sample for the requested domain.

        Args:
            domain: Domain name to sample from.

        Returns:
            Tuple of (raw bytes, label_idx) or None if unavailable.
        """
        if domain not in self.buffer:
            raise ValueError(f"Unknown domain: {domain}")
        
        # buffer of (path, label_idx)
        entries: deque[tuple[Path, int]] = self.buffer[domain]
        if not entries:
            raise ValueError(f"No samples available for domain: {domain}")
        
        path, label_idx = entries.popleft()
        data = path.read_bytes()
        if self.transform:
            data = self.transform(data)
        return data, label_idx

    @staticmethod
    def _load_meta(meta_path: Path) -> dict:
        """Load metadata content from disk.

        Args:
            meta_path: Path to `_meta.json`.

        Returns:
            Parsed metadata dictionary.
        """
        return json.loads(meta_path.read_text())
