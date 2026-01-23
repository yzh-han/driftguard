from pathlib import Path
from driftguard.federate.server.retrain_strategy import Driftguard
from driftguard.launch import (
    LaunchConfig,
    build_client,
    build_resnet18,
)
from driftguard.federate.server.fed_server import FedServerArgs, FedServer, start_fed_server
from driftguard.federate.client.client import FedClient, FedClientArgs
from driftguard.data.service import DataServiceArgs, start_data_service
from driftguard.data.drift_simulation import DriftEventArgs
from driftguard.rpc.proxy import DataServiceProxy, Node
from driftguard.rpc.rpc import ThreadedXMLRPCServer
from driftguard.config import get_logger
from driftguard.federate.server.cluster import ClusterArgs
import threading
from typing import List

logger = get_logger("launch")

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

    clients = [build_client(cid, cfg, build_resnet18) for cid in range(cfg.num_clients)]
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