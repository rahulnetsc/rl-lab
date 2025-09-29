import torch 
import torch.nn as nn
import numpy as np

import itertools as it

from typing import Any
from .base import Algo
from ..nets import MLP, CNN
from ..memory import RolloutBuffer

class A2C(Algo):
    def __init__(self, obs_dim, obs_shape, state_dim, action_dim, cfg, device) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        if self.action_dim is None:
            raise ValueError("A2C supports only discrete action spaces. Got: "
                            f"{type(self.action_dim)}")
        
        self.batch_size = cfg['batch_size']
        self.cfg = cfg
        
        
        # if not atari env or env requiring otherwise cnn, need a way to autmate this
        if len(self.obs_shape) == 1:
            # vector_obs -> MLP
            self.feature_layer = MLP(in_dim= obs_dim, out_dim= state_dim).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.rollout = RolloutBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_ep_info = True)
        else:
            # CNN size = (channels,H,W) 
            self.feature_layer = CNN(in_channels= obs_shape[0], out_channels= state_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.replay = RolloutBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_uint8 = True,            store_ep_info = True)

        self.init_model_weights()
 
        self.rng = np.random.default_rng(seed=cfg['seed'])
        
        self.act_rng = np.random.default_rng()
        self.batch_sampler = np.random.default_rng()
        self.loss_fn = nn.SmoothL1Loss(reduction='none')

        self.device = device
        self.lr = float(cfg.get('lr', 0.01))
        self.optimizer = None 

    def init_model_weights(self):
        def init_fn(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # leave activations/dropout alone
        self.feature_layer.apply(init_fn)
        self.policy_head.apply(init_fn)
        self.value_head.apply(init_fn)

    def select_action(self, obs_np: np.ndarray, eval_mode: bool = False):
        '''
        Given the current observation, return an action to take in the environment.
        - Convert obs_np into a tensor in the correct device, run the policy or Qnet 
            and obtain an environment compatible action
        - Handle exploration/exploitation with eval_mode
        - MUST NOT call env.step() here (or anywhere in the algo). The trainer owns env stepping.
        '''
        obs_tensor = torch.as_tensor(obs_np, device= self.device)
        logits = self.policy_head(obs_tensor)
        action = self.act_rng.choice(self.action_dim, p= logits) 
        return int(action)

    def observe(self, transition: dict[str, Any]) -> None:
        '''
        A per step-hook called after env.step() 
        transition (minimum)= 
            {
            "obs" : np.ndarray,                 # s_t
            "action" : Any,                     # a_t
            "reward" : float,                   # r_t
            "next_obs" : np.ndarray,            # s_{t+1}
            "done" : bool,                      # episode terminal flag
            "step" : int,                       # global step
            "value": value_fn,                  # output of Value head
            "logits": logits                    # output of Policy head
            }
        Typical usage:
            - Off policy: push to replay buffer
            - On policy: append to rollout buffer
        '''
        # obs

        self.rollout.push(transition= transition) 
    
    def update(self, global_step) -> dict[str, float] | None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).
        '''
        pass 
    

    def on_episode_end(self,)-> None:
        '''
        Optional per episode hook
        Examples:
          - finalize trajectory advantages/returns,
          - update exploration schedules,
          - reset episode-specific accumulators.
        '''
        trajectory = self.rollout.sample()
        

    def state_dict(self,):
        '''
        Return model parameters
        '''    
        pass 
    
    def load_state_dict(self, state: dict[str, Any])-> None:

        pass 
    
    def on_train_start(self, cfg):
        '''
        For initialization during training
        '''
        pass 
    
    def on_train_end(self,global_step ):
        '''
        Optional: train logic end
        '''
        pass 