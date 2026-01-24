from dataclasses import dataclass
import json
from pathlib import Path
from time import sleep

import torch
from driftguard.exp import DATASET, MODEL, Exps
from driftguard.federate.server.retrain_strategy import (
    Driftguard,
    Never,
    AveTrig,
    PerCTrig,
    MoEAve,
    MoEPerC,
    Cluster,
    RetrainStrategy,
)
from driftguard.federate.server.fed_server import FedServerArgs, FedServer, start_fed_server
from driftguard.federate.client.client import FedClient, FedClientArgs
from driftguard.data.service import DataServiceArgs, start_data_service
from driftguard.data.drift_simulation import DriftEventArgs
from driftguard.model.training.trainer import TrainConfig, Trainer
from driftguard.rpc.proxy import DataServiceProxy, Node, ServerProxy
from driftguard.config import get_logger
from driftguard.federate.server.cluster import ClusterArgs
import threading
from typing import Callable, List

logger = get_logger("launch")

@dataclass
class LaunchConfig:
    """Configuration for the local federated launch."""
    exp_name: str

    # data service
    sample_size_per_step: int 
    dataset: DATASET

    # client
    total_steps: int
    batch_size: int
    device: str
    model: MODEL
    num_clients: int
    epochs: int
    

    # server
    rt_round: int
    strategy: RetrainStrategy
    cluster_thr: float = 0.2
    min_group_size: int = 3
    w_size: int = 3

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
                epochs=cfg.epochs,
                device=cfg.device,
                lr=0.001,
                accumulate_steps=1,
                early_stop=True,
                cp_name=f"{cfg.dataset.name}-{cfg.model.value}"
            ),
        ),
        total_steps=cfg.total_steps,
        batch_size=cfg.batch_size,
        img_size=cfg.dataset.img_size,
        exp_name=cfg.exp_name
    )
    client = FedClient(args)
    client._trainer.load()  # load pretrained weights
    return client

#######################################
# Main Launching Code
#######################################
exps = Exps(
    datasets=[
        DATASET.DG5, 
        DATASET.PACS, 
        DATASET.DDN
    ],
    models=[
        MODEL.CRST_S, 
        MODEL.CRST_M, 
        MODEL.CVIT
    ],
    strategies=[
        Never(),
        # AveTrig(thr_acc=0.7),
        # PerCTrig(thr_acc=0.7),
        # MoEAve(thr_acc=0.7),
        MoEPerC(thr_acc=0.7),
        Cluster(thr_acc=0.7),
        # Driftguard(thr_reliance=0.2, thr_group_acc=0.7)
    ],
    device = "cuda" if torch.cuda.is_available() else "cpu",
).exps

def main() -> None:
    """Start the local data service, server, and clients."""
    for exp in exps:
        print("\n\n")
        logger.info(f"[Experiment]: {exp.name}, Dataset: {exp.dataset.name}, Model: {exp.model.value}, Strategy: {exp.strategy.name}")
        cfg = LaunchConfig(
            exp_name=exp.name,
            # data service
            sample_size_per_step = 30,
            dataset = exp.dataset,
            # client
            total_steps = 30, # <--------------------
            batch_size = 8,
            num_clients=20,
            model = exp.model,
            device = exp.device,
            epochs=20, # <--------------------
            # server
            rt_round=5, # communication rounds <--------------------
            strategy= exp.strategy,
            cluster_thr = 0.3,  # <--------------------
            data_port=12001, # <--------------------
            server_port=12002 # <--------------------
        )

        event_args = DriftEventArgs(
            n_time_steps=cfg.total_steps,
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
                rt_round=cfg.rt_round,
                retrain_strategy=cfg.strategy,
                clu_args=ClusterArgs(thr=cfg.cluster_thr, min_group_size=cfg.min_group_size, w_size=cfg.w_size),
            ),
        )

        sleep(3)  

        clients = [build_client(cid, cfg) for cid in range(cfg.num_clients)]
        threads = [
            threading.Thread(target=client.run, args=(), daemon=True)
            for client in clients
        ]

        sleep(2)  # wait for all clients to be ready
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 强制关闭 socket，线程会自动退出
        fed_server.server_close()
        data_server.server_close()

        logger.info("Launch finished.")

        sleep(1)  # 短暂等待即可

if __name__ == "__main__":
    main()