

from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader

from driftguard.recorder import Recorder
from driftguard.federate.observation import Fp, Observation
from driftguard.federate.params import FedParam, ParamType, Params
from driftguard.model.dataset import ListDataset, get_inference_transform, get_train_transform
from driftguard.model.training.trainer import Trainer
from driftguard.model.utils import freeze_layer, get_trainable_params
from driftguard.rpc.proxy import DataServiceProxy, ServerProxy

from typing import Any, Callable, Optional, Tuple, List, Dict, Deque
from driftguard.config import get_logger

logger = get_logger("fedclient")

@dataclass
class FedClientArgs:
    cid: int
    d_proxy: DataServiceProxy
    s_proxy: ServerProxy
    trainer: Trainer
    total_steps: int = 20
    batch_size: int = 6

    img_size: int = 28 # 28, 224 ,224

    exp_name: str = "exp"
    exp_root: str = "exp"

class FedClient:
    """client"""
    
    def __init__(
        self,
        args: FedClientArgs,
    ):
        self.cid = args.cid
        self.img_size: int = args.img_size
        self.d_proxy: DataServiceProxy = args.d_proxy
        self.s_proxy: ServerProxy = args.s_proxy
        self.total_steps: int = args.total_steps
        self.batch_size: int = args.batch_size
        self.model: nn.Module = args.trainer.model
        self._trainer: Trainer = args.trainer
        self._buffer: List =[]
        self.recorder = Recorder(args.exp_name, args.exp_root)

    def run(self):
        """perform one round of client operations"""
        time_step = 1
        while time_step <= self.total_steps:
            # step 1. inference
            time_step, = self.s_proxy.req_adv_step((self.cid,))
            samples = self.d_proxy.get_data((self.cid, time_step)) 
            logger.debug(f"c_{self.cid}: time_step {time_step}, samples {len(samples)}")           

            # step 2. upload observations, update local params
            obs = self.inference(samples)
            _, = self.s_proxy.req_upload_obs((self.cid, obs))
            # set params
            # fed_params.set(self.model)

            self.recorder.update_acc(time_step, obs.accuracy) 
            # step 3. trigger retrain if needed
            # obs = self.inference(samples) #
            # self.recorder.update_acc(time_step, obs.accuracy)    

            train_sets, val_sets = [*self._buffer, *samples[:-10]], samples[-10:]
            while True:
                FedParam.unfreeze(self.model)
                # request 3 req_trig
                fed_params, rt_cfg = self.s_proxy.req_trig(
                    (self.cid, obs, FedParam.get(self.model))
                )
                
                # stop
                if not rt_cfg.trigger:
                    if rt_cfg.param_type != ParamType.NONE:
                        fed_params.set(self.model) # n. last params update
                    break
                
                # 1. 准备更新参数
                fed_params.set(self.model) 

                # 2. no params need to retrain
                if not fed_params.gate:
                    freeze_layer(self.model, include_names=["gate"])
                if not fed_params.local:
                    freeze_layer(self.model, include_names=["local"])
                if not fed_params.other:
                    freeze_layer(self.model, include_names=["local", "gate"], exclude=True)
                # 3. 进入训练, 可以没有参数 e.g. 新组 使用原有参数
                if get_trainable_params(self.model) == 0:
                    logger.debug(f"{self.cid} No parameters to retrain, skip training.")
                    continue
                                
                self.train(train_sets, val_sets, time_step)

            # one step done, update buffer
            self._buffer = samples[-10:]
        # all steps done
        self.recorder.record(self.cid)

    def inference(self, samples: List[Tuple[bytes, int]]) -> Observation:
        loader = DataLoader(
            ListDataset(samples, get_inference_transform(self.img_size)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        metrix, l1_w, l2_w, softs = self._trainer.inference(loader)
        obs = Observation(
            accuracy=metrix.accuracy,  
            reliance=l1_w.mean(dim=[0,1])[0].item(), 
            fingerprint=Fp.build(
                out_softs=softs.cpu().numpy(),
                gate_activations=l2_w.cpu().numpy(),
                w_size=3
            )  
        )
        return obs
    
    def train(self, train_sets: List[Tuple[bytes, int]], val_sets: List[Tuple[bytes, int]], time_step: int) -> None:
        
        train_loader1, train_loader2, val_loader = (
            DataLoader(
                ListDataset(train_sets, get_inference_transform(self.img_size)),
                batch_size=self.batch_size,
                shuffle=True,
            ),
            DataLoader(
                ListDataset(train_sets, get_train_transform(self.img_size)),
                batch_size=self.batch_size,
                shuffle=True,
            ),
            DataLoader(
                ListDataset(val_sets, get_inference_transform(self.img_size)),
                batch_size=self.batch_size,
                shuffle=False,
            ),
        )

        # 2 stage train, origin -> 增强
        history_1 = self._trainer.fit(train_loader1, val_loader)
        history_2 = self._trainer.fit(train_loader2, val_loader)

        self.recorder.update_cost(
            time_step, get_trainable_params(self.model), len(history_1) + len(history_2)
        )