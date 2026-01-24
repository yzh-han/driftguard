from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List

from driftguard.config import get_logger
from driftguard.federate.observation import Observation
from driftguard.federate.params import FedParam, ParamType, Params, aggregate_params
from driftguard.federate.retrain_config import RetrainConfig
from driftguard.federate.server.cluster import Group, GroupState
from driftguard.federate.server.state import RetrainState
from statistics import mean

logger = get_logger("retrain_strategy")

class RetrainStrategy(ABC):
    """Pluggable retraining strategy for trigger and aggregation behavior."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the retraining strategy."""

    @abstractmethod
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        """
        Handle uploaded observations before retraining decisions.
        """

    @abstractmethod
    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """
        Handle retraining trigger and aggregation logic.
        """

class Never(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    @property
    def name(self) -> str:
        return "never"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
        logger.debug("Retraining never triggered.")

class AveTrig(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    def __init__(self, thr_acc: float):
        self.thr_acc = thr_acc
    @property
    def name(self) -> str:
        return "average"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        ave_acc = mean([obs.accuracy for obs in obs_list])
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if ave_acc < self.thr_acc:
                # - 全局重训练
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.FULL)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")

class PerCTrig(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    def __init__(self, thr_acc: float):
        self.thr_acc = thr_acc
    @property
    def name(self) -> str:
        return "per_client"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        drop_clients = [
            c for c in range(len(obs_list)) if obs_list[c].accuracy < self.thr_acc
        ]
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if drop_clients:
                # True, Drop clients, FULL
                rt_state.rt_cfg = RetrainConfig(True, drop_clients, ParamType.FULL)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
                sel_clients=rt_state.rt_cfg.selection,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
                sel_clients=rt_state.rt_cfg.selection,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")

class MoEAve(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    def __init__(self, thr_acc: float):
        self.thr_acc = thr_acc
    @property
    def name(self) -> str:
        return "moe_ave"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        ave_acc = mean([obs.accuracy for obs in obs_list])
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if ave_acc < self.thr_acc:
                # - 全局重训练
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.MOE)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")

class MoEPerC(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    def __init__(self, thr_acc: float):
        self.thr_acc = thr_acc
    @property
    def name(self) -> str:
        return "moe_perC"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        pass

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        drop_clients = [
            c for c in range(len(obs_list)) if obs_list[c].accuracy < self.thr_acc
        ]
        
        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if drop_clients:
                # True, Drop clients, FULL
                rt_state.rt_cfg = RetrainConfig(True, drop_clients, ParamType.MOE)
                rt_state.remain_round = rt_state._rt_round
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
                sel_clients=rt_state.rt_cfg.selection,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
                sel_clients=rt_state.rt_cfg.selection,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")
      


class Cluster(RetrainStrategy):
    """A retraining strategy that never triggers retraining."""
    def __init__(self, thr_acc: float):
        self.thr_acc = thr_acc

    @property
    def name(self) -> str:
        return "cluster"
    
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        rt_state.is_cluster = True

        fps = [obs.fingerprint for obs in obs_list if obs.fingerprint is not None]
        # AgglomerativeClustering requires at least 2 samples.
        if len(fps) >= 2:
            grp_state.update(fps)
        logger.info(f"[Updated groups]: {grp_state.groups}")

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """Apply the default retraining logic to the current round.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.
            params_list: Parameters uploaded by all clients.

        Returns:
            None.
        """
        group_accs = Observation.group_ave_acc(obs_list, grp_state.groups)
        grps = [g for g, acc in group_accs if acc < self.thr_acc]

        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if grps:
                # - 分组重训练
                rt_state.rt_cfg = RetrainConfig(
                    True, [c for g in grps for c in g.clients], ParamType.CLUSTER
                )
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")



class Driftguard(RetrainStrategy):
    """Default retraining strategy based on reliance and group accuracy."""
    def __init__(self, thr_reliance: float = 0.1, thr_group_acc: float = 0.8):
        self.thr_reliance = thr_reliance
        self.thr_group_acc = thr_group_acc

    @property
    def name(self) -> str:
        return "driftguard"
    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
        rt_state: RetrainState,
    ) -> None:
        fps = [obs.fingerprint for obs in obs_list if obs.fingerprint is not None]
        # AgglomerativeClustering requires at least 2 samples.
        if len(fps) >= 2:
            grp_state.update(fps)
        logger.info(f"[Updated groups]: {grp_state.groups}")

    def on_trig(
        self,
        obs_list: List[Observation],
        params_list: List[Params],
        rt_state: RetrainState,
        grp_state: GroupState,
        param_state: FedParam,
    ) -> None:
        """Apply the default retraining logic to the current round.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.
            params_list: Parameters uploaded by all clients.

        Returns:
            None.
        """
        rt_state = rt_state
        reliance = Observation.ave_reliance(obs_list)
        group_accs = Observation.group_ave_acc(obs_list, grp_state.groups)
        grps = [g for g, acc in group_accs if acc < self.thr_group_acc]

        logger.debug(f"Reliance: {reliance}, Group Accuracies: {group_accs}")


        if rt_state.stage == RetrainState.Stage.IDLE:
            # 1. 开始
            if reliance < self.thr_reliance:
                # - 全局重训练
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.SHARED)
                
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
                
            elif grps:
                # - 分组重训练
                rt_state.rt_cfg = RetrainConfig(
                    True, [c for g in grps for c in g.clients], ParamType.LOCAL
                )
                rt_state.remain_round = rt_state._rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
            else:
                # - 不重训练 keep cfg
                rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")

        elif rt_state.stage == RetrainState.Stage.ONGOING:
            # 2. 继续训练
            # 2.1 聚合参数
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )

            rt_state.remain_round -= 1

        elif rt_state.stage == RetrainState.Stage.COMPLETED:
            aggregate(
                params_list,
                rt_state.rt_cfg,
                grp_state,
                param_state,
            )
            rt_state.rt_cfg.trigger = False
            logger.debug("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")
    
def aggregate(
    params_list: List[Params],
    rt_cfg: RetrainConfig,
    grp_state: GroupState,
    param_state: FedParam,
    sel_clients: List[int] | None = None,
) -> None:
    # 2.1 聚合参数
    if rt_cfg.param_type == ParamType.LOCAL or rt_cfg.param_type == ParamType.CLUSTER:
        assert rt_cfg.selection is not None or [], "need selection"
        grps = grp_state.unique_groups(
            rt_cfg.selection
        )
        for g in grps:
            # 更新组内参数
            g.params = aggregate_params(
                [params_list[c] for c in g.clients if c in rt_cfg.selection]
            )
    # 2.2 全局训练
    elif rt_cfg.param_type == ParamType.SHARED:
        params = aggregate_params(params_list)
        param_state.shared = params
    elif rt_cfg.param_type == ParamType.FULL:
        if sel_clients is not None:
            params = aggregate_params(
                [params_list[i] for i in range(len(params_list)) if i in sel_clients]
            )
        else:
            params = aggregate_params(params_list)
        param_state.full = params
    else:
        raise ValueError(f"Unknown param type: {rt_cfg.param_type}")
        