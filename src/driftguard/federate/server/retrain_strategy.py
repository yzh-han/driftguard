from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from turtle import rt
from typing import Callable, List

from driftguard.config import get_logger
from driftguard.federate.observation import Observation
from driftguard.federate.params import FedParam, ParamType, Params, aggregate_params
from driftguard.federate.retrain_config import RetrainConfig
from driftguard.federate.server.cluster import Group, GroupState
from driftguard.federate.server.state import RetrainState

logger = get_logger("retrain_strategy")

@dataclass
class RetrainContext:
    """Runtime context required by a retraining strategy.

    Attributes:
        retrain_state: Mutable retrain state for remaining rounds and config.
        group_state: Cluster grouping state for clients.
        retrain_rounds: Default number of rounds for a new retraining session.
        set_shared_params: Callback to update shared parameters.
        set_gate_params: Callback to update gate parameters.
    """

    retrain_state: RetrainState
    group_state: GroupState
    retrain_rounds: int
    set_shared_params: Callable[[Params], None]
    set_gate_params: Callable[[Params], None]


class RetrainStrategy(ABC):
    """Pluggable retraining strategy for trigger and aggregation behavior."""

    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
    ) -> None:
        """Handle uploaded observations before retraining decisions.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.

        Returns:
            None.
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
        """Handle a trigger round and mutate context.

        Args:
            context: Runtime retrain context to mutate.
            obs_list: Observations from all clients for the round.
            params_list: Parameters uploaded by all clients.

        Returns:
            None.
        """


class Driftguard(RetrainStrategy):
    """Default retraining strategy based on reliance and group accuracy."""
    def __init__(self, thr_reliance: float = 0.1, thr_group_acc: float = 0.8):
        self.thr_reliance = thr_reliance
        self.thr_group_acc = thr_group_acc

    def on_obs(
        self,
        obs_list: List[Observation],
        grp_state: GroupState,
    ) -> None:
        fps = [obs.fingerprint for obs in obs_list]
        if fps:
            grp_state.update(fps)

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
                rt_state.rt_cfg = RetrainConfig(True, grp_state.all_clients, ParamType.FULL)
                
                rt_state.remain_round = rt_state.rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
                
            elif grps:
                # - 分组重训练
                rt_state.rt_cfg = RetrainConfig(
                    True, [c for g in grps for c in g.clients], ParamType.LOCAL
                )
                rt_state.remain_round = rt_state.rt_round
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")
            else:
                # - 不重训练 keep cfg
                logger.debug(f"Rt Cfg: {rt_state.rt_cfg}")

        elif rt.remain_round > 0:
            # 2. 继续训练
            # 2.1 组内训练
            if rt_state.rt_cfg.param_type == ParamType.LOCAL:
                grps = grp_state.unique_groups(
                    rt_state.rt_cfg.selection
                )
                for g in grps:
                    # 更新组内参数
                    g.params = aggregate_params(
                        [
                            params_list[c]
                            for c in g.clients
                            if c in rt_state.rt_cfg.selection
                        ]
                    )
            # 2.2 全局训练
            else:
                params = aggregate_params(params_list)
                param_state.shared = params

            logger.info(
                f"Retraining round completed. {rt_state.remain_round - 1} rounds remaining."
            )
            rt_state.remain_round -= 1

        elif rt_state.remain_round <= 0 and rt_state.rt_cfg.trigger:
            # 3. 结束
            rt_state.rt_cfg = RetrainConfig(False, [], ParamType.NONE)
            logger.info("Retraining ended.")
        else:
            raise ValueError("Inconsistent retrain state.")
    
