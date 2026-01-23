from dataclasses import dataclass
import json
from pathlib import Path
from time import sleep

import torch
from driftguard.exp import DATASET, MODEL, Exps
from driftguard.model.c_resnet.model import get_cresnet
from driftguard.model.c_vit.model import get_cvit
from driftguard.federate.server.retrain_strategy import Driftguard, RetrainStrategy
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
    
    # data service
    sample_size_per_step: int 
    dataset: DATASET

    # client
    total_rounds: int
    batch_size: int
    device: str
    model: MODEL
    num_clients: int
    

    # server
    strategy: RetrainStrategy
    cluster_thr: float = 0.2
    min_group_size: int = 3
    w_size: int = 3
    thr_reliance: float = 0.1
    thr_group_acc: float = 0.8

    # ports
    data_port: int = 12099
    server_port: int = 12000

    seed: int = 42


def build_client(cid: int, cfg: LaunchConfig) -> FedClient:
    """Construct a single FedClient with a ResNet18 model and trainer."""

    args = FedClientArgs(
        cid=cid,
        d_proxy=DataServiceProxy(Node("http://127.0.0.1", cfg.data_port)),
        s_proxy=ServerProxy(Node("http://127.0.0.1", cfg.server_port)),
        trainer=Trainer(
            cfg.model.fn(cfg.dataset.num_classes),
            config=TrainConfig(
                epochs=1,
                device=cfg.device,
                lr=0.001,
                accumulate_steps=1,
                early_stop=True,
            ),
        ),
        total_steps=cfg.total_rounds,
        batch_size=cfg.batch_size,
        img_size=cfg.dataset.img_size,
    )
    client = FedClient(args)
    return client

#######################################
# Main Launching Code
#######################################
exps = Exps(
    datasets=[DATASET.DG5, DATASET.PACS, DATASET.DDN],
    models=[MODEL.CRST_S, MODEL.CRST_M, MODEL.CVIT],
    strategies=[Driftguard()],
    device = "cuda" if torch.cuda.is_available() else "cpu",

).exps
def main() -> None:
    """Start the local data service, server, and clients."""
    for exp in exps:
        logger.info(f"[Experiment]: {exp.name}, Dataset: {exp.dataset.name}, Model: {exp.model.value}, Strategy: {exp.strategy.name}")
        cfg = LaunchConfig(
            # data service
            sample_size_per_step = 30,
            dataset = exp.dataset,
            # client
            total_rounds = 20,
            batch_size = 8,
            num_clients=30,
            model = exp.model,
            device = exp.device,
            # server
            strategy= exp.strategy,
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

        #  ## data service
        data_args = DataServiceArgs(
            meta_path=cfg.dataset.path,
            num_clients=cfg.num_clients,
            sample_size=cfg.sample_size_per_step,
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
                retrain_strategy=Driftguard(thr_reliance=cfg.thr_reliance, thr_group_acc=cfg.thr_group_acc),
                clu_args=ClusterArgs(thr=cfg.cluster_thr, min_group_size=cfg.min_group_size, w_size=cfg.w_size),
            ),
        )

        sleep(5)  

        clients = [build_client(cid, cfg) for cid in range(cfg.num_clients)]
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