

from dataclasses import dataclass
import torch.nn as nn
from torch.utils.data import DataLoader

from driftguard.exp import Exp
from driftguard.federate.observation import Fp, Observation
from driftguard.federate.params import FedParam, ParamType, Params
from driftguard.model.dataset import ListDataset, get_inference_transform, get_train_transform
from driftguard.model.training.trainer import Trainer
from driftguard.model.utils import get_trainable_params
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
        self.exp = Exp(args.exp_name)
    def run(self):
        """perform one round of client operations"""
        time_step = 1
        while time_step <= self.total_steps:
            # step 1. inference
            time_step, = self.s_proxy.req_adv_step((self.cid,))
            samples = self.d_proxy.get_data((self.cid, time_step)) 
            logger.debug(f"c_{self.cid}: time_step {time_step}, samples {len(samples)}")           

            # step 2. upload observations
            obs = self.inference(samples)
            params, param_type = self.s_proxy.req_upload_obs((self.cid, obs))
            if params:
                FedParam.set(self.model, params, param_type)

            # step 3. trigger retrain if needed
            obs = self.inference(samples) #
            self.exp.update_acc(time_step, obs.accuracy)    

            params = []
            while True:
                params, rt_cfg = self.s_proxy.req_trig((self.cid, obs, params))
                
                if not rt_cfg.trigger or self.cid not in rt_cfg.selection:
                    continue

                if params: 
                    FedParam.set(self.model, params, rt_cfg.param_type)
                
                FedParam.freeze_exclude(self.model, rt_cfg.param_type)
                FedParam.unfreeze(self.model, ["gate"])
                
                self.train(samples)
                params = FedParam.get(self.model, rt_cfg.param_type)
                self.exp.update_cost(time_step, get_trainable_params(self.model))
                FedParam.unfreeze(self.model)
        self.exp.record(self.cid)

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
    
    def train(self, samples: List[Tuple[bytes, int]]) -> None:
        loader = DataLoader(
            ListDataset(samples, get_train_transform(self.img_size)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self._trainer.fit(loader)
