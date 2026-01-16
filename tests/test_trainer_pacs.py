"""Smoke tests for the minimal trainer with folder datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from driftguard.model.training import TrainConfig, Trainer
from driftguard.model.vit import ViTArgs, VisonTransformer


@dataclass(frozen=True)
class Sample:
    """Represents a single image sample on disk.

    Attributes:
        path: Path to the image file.
        label: Integer class label.
    """

    path: Path
    label: int


class FolderDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset that loads images from class folders.

    Attributes:
        samples: Collected image samples.
        class_names: Sorted class names for the dataset.
    """

    def __init__(self, root: Path, max_samples: int) -> None:
        """Initialize the dataset from a folder of class subdirectories.

        Args:
            root: Root folder containing class subdirectories.
            max_samples: Maximum number of samples to load.
        """

        self.samples, self.class_names = self._collect_samples(root, max_samples)

    def __len__(self) -> int:
        """Return the number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an image tensor and label.

        Args:
            index: Sample index.

        Returns:
            Tuple of (image tensor, label tensor).
        """

        image_module = pytest.importorskip("PIL.Image")
        sample = self.samples[index]
        image = image_module.open(sample.path).convert("RGB")
        image = image.resize((224, 224))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        label = torch.tensor(sample.label, dtype=torch.long)
        return tensor, label

    @staticmethod
    def _collect_samples(root: Path, max_samples: int) -> tuple[list[Sample], list[str]]:
        """Collect image paths and labels from the root folder.

        Args:
            root: Root folder containing class subdirectories.
            max_samples: Maximum number of samples to collect.

        Returns:
            List of samples.
        """

        samples: list[Sample] = []
        class_dirs = [item for item in sorted(root.iterdir()) if item.is_dir()]
        class_names = [item.name for item in class_dirs]
        for idx, class_dir in enumerate(class_dirs):
            try:
                label = int(class_dir.name)
            except ValueError:
                label = idx
            for image_path in sorted(class_dir.glob("*.jpg")):
                samples.append(Sample(path=image_path, label=label))
                if len(samples) >= max_samples:
                    return samples, class_names
        return samples, class_names


def test_trainer_runs_on_pacs_subset() -> None:
    """Train a small ViT on a tiny PACS subset without errors."""

    data_root = Path("datasets/pacs/photo")
    if not data_root.exists():
        pytest.skip("PACS dataset not found in datasets/pacs/photo")

    dataset = FolderDataset(data_root, max_samples=100)

    if len(dataset) == 0:
        pytest.skip("PACS dataset folder is empty")

    num_classes = len({sample.label for sample in dataset.samples})
    if num_classes == 0:
        pytest.skip("No classes found in PACS dataset")

    train_size = max(1, int(len(dataset) * 0.8))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)

    args = ViTArgs(
        embed_dim=64,
        embed_image_size=224,
        embed_patch_size=16,
        encoder_depth=2,
        mha_num_heads=4,
        repr_dim=64,
        head_num_classes=num_classes,
    )
    model = VisonTransformer(args)

    trainer = Trainer(
        model,
        loss_fn=nn.CrossEntropyLoss(),
        config=TrainConfig(epochs=10, device="cpu", amp=False),
    )
    history = trainer.fit(train_loader, val_loader)

    assert len(history) == 10
    assert "train_loss" in history[0]
    if val_size:
        assert "val_loss" in history[0]

    for record in history:
        print(record)

if __name__ == "__main__":
    test_trainer_runs_on_pacs_subset()
