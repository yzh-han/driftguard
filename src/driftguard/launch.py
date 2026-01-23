"""Launch a minimal federated run with data service, server, and clients."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18

from driftguard.config import get_logger
from driftguard.federate.client.client import FedClient, FedClientArgs
from driftguard.model.training.trainer import TrainConfig, Trainer
from driftguard.rpc.proxy import DataServiceProxy, ServerProxy
from driftguard.rpc.rpc import Node

logger = get_logger("launch")


@dataclass
class LaunchConfig:
    """Configuration for the local federated launch."""

    meta_path: Path = Path("datasets/dg5/_meta.json")
    num_clients: int = 30
    batch_size: int = 4
    data_port: int = 12099
    server_port: int = 12000
    total_rounds: int = 10
    img_size: int = 28
    cluster_thr: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        self.num_classes: int = self.load_num_classes(self.meta_path)

    @staticmethod
    def load_num_classes(meta_path: Path) -> int:
        """Load the number of classes from a dataset meta file."""
        meta = json.loads(meta_path.read_text())
        labels = meta.get("labels")
        if labels:
            return len(labels)
        return len(meta.get("label_to_idx", {}))


def build_client(cid: int, cfg: LaunchConfig, build_model: Callable) -> FedClient:
    """Construct a single FedClient with a ResNet18 model and trainer."""
    model = build_model(cfg.num_classes)
    trainer = Trainer(
        model,
        nn.CrossEntropyLoss(),
        config=TrainConfig(epochs=1),
    )
    args = FedClientArgs()
    args.cid = cid
    args.model = model
    args.img_size = cfg.img_size
    args.d_proxy = DataServiceProxy(Node("http://127.0.0.1", cfg.data_port))
    args.s_proxy = ServerProxy(Node("http://127.0.0.1", cfg.server_port))
    client = FedClient(args)
    client._trainer = trainer
    return client

def build_resnet18(num_classes: int) -> nn.Module:
    """Build a ResNet18 instance."""
    model = resnet18(num_classes=num_classes)
    return model