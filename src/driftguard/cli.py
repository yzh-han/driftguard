from dataclasses import dataclass
import json
from pathlib import Path

import torch
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.federate.server.retrain_strategy import Driftguard
from driftguard.federate.server.fed_server import FedServerArgs, FedServer, start_fed_server
from driftguard.federate.client.client import FedClient, FedClientArgs
from driftguard.data.service import DataServiceArgs, start_data_service
from driftguard.data.drift_simulation import DriftEventArgs
from driftguard.model.training.trainer import TrainConfig, Trainer
from driftguard.rpc.proxy import DataServiceProxy, Node, ServerProxy
from driftguard.rpc.rpc import ThreadedXMLRPCServer
from driftguard.config import get_logger
from driftguard.federate.server.cluster import ClusterArgs
import threading
from typing import Callable, List

logger = get_logger("launch")

@dataclass
class LaunchConfig:
    """Configuration for the local federated launch."""
    
    batch_size: int = 6
    num_clients: int = 3
    img_size: int = 28

    meta_path: Path = Path("datasets/dg5/_meta.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_port: int = 12099
    server_port: int = 12000

    total_rounds: int = 10
    cluster_thr: float = 0.5
    seed: int = 42

    def __post_init__(self) -> None:
        self.num_classes: int = self.load_num_classes(self.meta_path)
        self.model_fn: Callable = self.build_resnet18

    @staticmethod
    def load_num_classes(meta_path: Path) -> int:
        """Load the number of classes from a dataset meta file."""
        meta = json.loads(meta_path.read_text())
        labels = meta.get("labels")
        if labels:
            return len(labels)
        return len(meta.get("label_to_idx", {}))


    def build_resnet18(self) -> Callable:
        """Build a ResNet18 model."""
        return get_cresnet(num_classes=self.num_classes)
    
    def build_cvit(self) -> Callable:
        """Build a Cvit model."""
        return get_cvit(num_classes=self.num_classes, image_size=self.img_size)


def build_client(cid: int, cfg: LaunchConfig, build_model: Callable) -> FedClient:
    """Construct a single FedClient with a ResNet18 model and trainer."""

    args = FedClientArgs(
        cid=cid,
        d_proxy=DataServiceProxy(Node("http://127.0.0.1", cfg.data_port)),
        s_proxy=ServerProxy(Node("http://127.0.0.1", cfg.server_port)),
        trainer=Trainer(
            cfg.model_fn(),
            config=TrainConfig(epochs=1, device=cfg.device, lr=0.001, accumulate_steps=1),
        ),
        total_steps=cfg.total_rounds,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
    )
    client = FedClient(args)
    return client

def main() -> None:
    """Start the local data service, server, and clients."""
    cfg = LaunchConfig(
        meta_path = Path("datasets/drift_domain_net/_meta.json"), # dg5, pacs, drift_domain_net
        num_clients = 1,
        batch_size = 4,
        data_port = 12099,
        server_port = 12000,
        total_rounds = 10,
        img_size = 28,
        cluster_thr = 0.5,
        seed = 42,
    )

    event_args = DriftEventArgs(
        n_time_steps=cfg.total_rounds,
        n_clients=cfg.num_clients,
        n_sudden=3,
        n_gradual=3,
        n_stage=1,
        aff_client_ratio_range=(0.1, 0.15),
        start=0.05,
        end=0.8,
        dist_range=(1, 3),
        gradual_duration_ratio=0.15,
        seed=cfg.seed,
    )

    data_args = DataServiceArgs(
        meta_path=cfg.meta_path,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
        drift_event_args=event_args,
        seed=cfg.seed,
    )

    data_server, _data_service = start_data_service(
        Node("0.0.0.0", cfg.data_port), data_args
    )

    fed_server, _fed_service = start_fed_server(
        Node("0.0.0.0", cfg.server_port),
        FedServerArgs(
            data_service_proxy=DataServiceProxy(
                Node("http://127.0.0.1", cfg.data_port)
            ),
            num_clients=cfg.num_clients,
            rt_round=2,
            retrain_strategy=Driftguard(thr_reliance=0.1, thr_group_acc=0.8),
            clu_args=ClusterArgs(thr=cfg.cluster_thr, min_group_size=3, w_size=3),
        ),
    )

    clients = [build_client(cid, cfg, cfg.model_fn) for cid in range(cfg.num_clients)]
    threads = [
        threading.Thread(target=client.run, args=(), daemon=True)
        for client in clients
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    fed_server.shutdown()
    data_server.shutdown()
    logger.info("Launch finished.")

if __name__ == "__main__":
    main()