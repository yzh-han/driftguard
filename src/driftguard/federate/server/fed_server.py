from dataclasses import dataclass
import threading
from typing import List, Tuple

from driftguard.federate.observation import Observation
from driftguard.federate.params import FedParam, ParamType, Params
from driftguard.federate.retrain_config import RetrainConfig
from driftguard.federate.server.state import ReqState, RetrainState
from driftguard.federate.server.retrain_strategy import (
    RetrainStrategy,
)
from driftguard.rpc.proxy import DataServiceProxy
from driftguard.rpc.rpc import Node, ThreadedXMLRPCServer, server_func
from driftguard.config import get_logger
from driftguard.federate.server.sync import ServerSyncCoordinator
from driftguard.federate.server.cluster import ClusterArgs, Group, GroupState

logger = get_logger("fed_server")

@dataclass
class FedServerArgs:
    """Arguments for constructing a federated server.

    Attributes:
        data_service_proxy: Proxy for remote data service calls.
        num_clients: Total number of clients in the federation.
        retrain_rounds: Default number of rounds for retraining.
        retrain_strategy: Optional strategy to handle retraining triggers.
    """

    data_service_proxy: DataServiceProxy
    num_clients: int
    rt_round: int # default communication rounds
    retrain_strategy: RetrainStrategy
    clu_args: ClusterArgs
    

class FedServer:
    def __init__(
        self,
        args: FedServerArgs,
    ):
        # STATES
        self.param_state: FedParam = FedParam([],[],[])
        self.grp_state: GroupState = GroupState(args.num_clients, args.clu_args)
        self.rt_state: RetrainState = RetrainState(_rt_round=args.rt_round)

        self.rt_strategy: RetrainStrategy = args.retrain_strategy
        
        # runtime
        self._time_step: int = 0
        self._data_service_proxy: DataServiceProxy = args.data_service_proxy
        self._sync: ServerSyncCoordinator = ServerSyncCoordinator(
            ReqState(args.num_clients)
        )

        logger.info(f"FedServer starting ...")



    def _attach_server(self, server: ThreadedXMLRPCServer) -> None:
        self._server = server

    def stop(self) -> bool:
        """Shutdown the XML-RPC server if attached.

        Returns:
            True if shutdown was triggered, False otherwise.
        """
        # Allow remote shutdown via RPC.
        if self._server is None:
            return False

        self._data_service_proxy.stop()

        logger.info("Shutting down the FedServer...")
        threading.Thread(target=self._server.shutdown, daemon=True).start()
        return True

    @server_func
    def req_adv_step(self, args: Tuple[int,]) -> Tuple[int,]:
        (cid,) = args

        # perform once
        def on_step():
            self._time_step += 1
            logger.info(f"[Time step] advanced to [{self._time_step}]")

        self._sync.await_adv_step(cid, on_step)
        return (self._time_step,)

    @server_func
    def req_upload_obs(self, args: Tuple[int, Observation]) -> Tuple[Params, ParamType]:
        """Return group parameters, or empty if no groups."""
        cid, obs = args
        # store observation

        def on_obs(obs_list: List[Observation], grp_state: GroupState, rt_state: RetrainState) -> None:
            self.rt_strategy.on_obs(obs_list, grp_state, rt_state)
            logger.info(f"[Ave Acc] {sum(o.accuracy for o in obs_list)/len(obs_list):.4f}")
            logger.info(f"[Groups] {Observation.group_ave_acc(obs_list, grp_state.groups)}")
        self._sync.await_upload_obs(cid, on_obs, obs, self.grp_state, self.rt_state)
        
        params, param_type = [], ParamType.NONE
        if self.grp_state.groups: # has groups
            params = self.grp_state.get_group(cid).params
            param_type = ParamType.CLUSTER if self.rt_state.is_cluster else ParamType.LOCAL

        return params, param_type

    @server_func
    def req_trig(self, args: Tuple[int, Observation, Params]) -> Tuple[Params, RetrainConfig]:
        cid, obs, params = args

        def on_trig(
            obs_list: List[Observation],
            params_list: List[Params],
            rt_state: RetrainState,
            grp_state: GroupState,
            param_state: FedParam,
        ) -> None:
            self.rt_strategy.on_trig(obs_list, params_list, rt_state, grp_state, param_state)
            logger.info(f"[Retrain Trigger] rt_cfg: {rt_state.rt_cfg}")

        self._sync.await_trig(cid, on_trig, obs, params, self.rt_state, self.grp_state, self.param_state)

        params = []
        if self.rt_state.rt_cfg.param_type == ParamType.SHARED:
            params = self.param_state.shared
        elif (
            self.rt_state.rt_cfg.param_type == ParamType.LOCAL
            or self.rt_state.rt_cfg.param_type == ParamType.CLUSTER
        ):
            params = self.grp_state.get_group(cid).params
        elif (
            self.rt_state.rt_cfg.param_type == ParamType.FULL
            or self.rt_state.rt_cfg.param_type == ParamType.MOE
        ):
            params = self.param_state.full
        else:
            pass
        return params, self.rt_state.rt_cfg  # placeholder
 
def start_fed_server(
    node: Node, args: FedServerArgs
) -> Tuple[ThreadedXMLRPCServer, FedServer]:
    """Start the federated server in a background thread."""
    server = ThreadedXMLRPCServer((node.host, node.port), logRequests=False, allow_none=True)
    service = FedServer(args)
    service._attach_server(server)
    server.register_instance(service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, service
