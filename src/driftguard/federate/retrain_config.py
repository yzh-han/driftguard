
from dataclasses import dataclass
from typing import List

from driftguard.federate.params import ParamType

@dataclass
class RetrainConfig:
    trigger: bool
    selection: List[int] # selected clients
    param_type: ParamType
