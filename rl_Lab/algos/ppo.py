import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

import itertools as it

from typing import Any
from .base import Algo
from ..nets import MLP, CNN
from ..memory import RolloutBuffer

class PPO(Algo):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def select_action(self, obs_np : np.ndarray, eval_mode: bool = False)->  Any:
        pass 

    def observe(self, transition: dict[str, Any])-> None:
        pass 

    def update(self,global_step)-> dict[str, float]| None:
        return
    
    def on_episode_end(self,)-> None:
        pass 

    def on_train_start(self, cfg):
        pass 

    def on_train_end(self, global_step):
        pass 

    def state_dict(self,)-> dict[str, Any]|None:
        pass 

    def load_state_dict(self, state: dict[str, Any])-> None:
        return 
    
        
 