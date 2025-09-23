import torch 
import torch.nn as nn

from torch.distributions import Categorical

import numpy as np
import gymnasium as gym

from collections import defaultdict
from typing import Any

from .base import Algo
from ..nets import MLP, CNN

from ..memory import ReplayBuffer
import ipdb 

class DQN(Algo):
    def __init__(self, obs_dim, obs_shape, action_dim, cfg, device) -> None:
        super().__init__()

        self.obs_dim = obs_dim
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        if self.action_dim is None:
            raise ValueError("DQN supports only discrete action spaces. Got: "
                            f"{type(self.action_dim)}")
        
        
        self.batch_size = cfg['batch_size']
        self.cfg = cfg
        self.eps = cfg['epsilon']
        self.gamma = cfg['gamma']
        self.reset_gamma = cfg['gamma']
        self.max_norm = cfg['max_norm']
        self.warmup_steps = cfg['warmup_steps']

        # if not atari env or env requiring otherwise cnn, need a way to autmate this
        if len(self.obs_shape) == 1:
            # vector_obs -> MLP
            self.online_net = MLP(in_dim= obs_dim, out_dim= action_dim).to(device=device)
            self.target_net = MLP(in_dim= obs_dim, out_dim= action_dim).to(device=device)
            self.replay = ReplayBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_ep_info = True)
        else:
            # CNN size = (channels,H,W) 
            self.online_net = CNN(in_channels= obs_shape[0], out_channels= action_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.target_net = CNN(in_channels= obs_shape[0], out_channels= action_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.replay = ReplayBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_uint8 = True,  store_ep_info = True)

        self.init_model_weights()

    
        self.rng = np.random.default_rng(seed=cfg['seed'])
        
        self.act_rng = np.random.default_rng()
        self.batch_sampler = np.random.default_rng()
        self.loss_fn = nn.MSELoss()

        self.device = device
        self.lr = float(cfg.get('lr', 0.01))
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.online_dict = None    
        self.target_dict =  None
        self.optimizer_dict = None
    
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

        self.online_net.apply(init_fn)
        # Start target identical to online
        self.target_net.load_state_dict(self.online_net.state_dict())
    
    def select_action(self, obs_np : np.ndarray, eval_mode: bool = False)->  Any:
        '''
        Given the current observation, return an action to take in the environment.
        - Convert obs_np into a tensor in the correct device, run the policy or Qnet 
            and obtain an environment compatible action
        - Handle exploration/exploitation with eval_mode
        - MUST NOT call env.step() here (or anywhere in the algo). The trainer owns env stepping.
        '''
        if not eval_mode and (self.rng.random() < self.eps):
            # exploration
            return self.rng.integers(low= 0, high= self.action_dim)
        
        obs_tensor = torch.tensor(obs_np, device= self.device)
        logits = self.online_net(obs_tensor)
        action = torch.argmax(logits, dim= -1).item()
        return action


    def observe(self, transition: dict[str, Any])-> None:
        '''
        A per step-hook called after env.step() 
        transition (minimum)= 
            {
            "obs" : np.ndarray,                 # s_t
            "action" : Any,                     # a_t
            "reward" : float,                   # r_t
            "next_obs" : np.ndarray,            # s_{t+1}
            "done" : bool,                      # episode terminal flag
            "step" : int                        # global step
            }
        Typical usage:
            - Off policy: push to replay buffer
            - On policy: append to rollout buffer
        '''
        self.replay.push(transition=transition)
        return None 
    
    def update(self,global_step)-> dict[str, float]| None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).

        '''
        if self.replay.can_sample(batch_size=self.batch_size):
            sample = self.replay.sample(batch_size= self.batch_size, rng= self.batch_sampler)
            # ipdb.set_trace()
            obs = sample['obs']
            obs = torch.as_tensor(obs, device= self.device)
            action = sample['action']
            action = torch.as_tensor(action, device= self.device).unsqueeze(1)
            reward = sample['reward']
            reward = torch.as_tensor(reward, device= self.device).unsqueeze(1)
            next_obs = sample['next_obs']
            next_obs = torch.as_tensor(next_obs, device= self.device)
            done = sample['done']
            done = torch.as_tensor(done, device= self.device, dtype=torch.float32).unsqueeze(1)

            
            with torch.no_grad():
                next_state_q_values = self.target_net(next_obs)
                max_next_q_values = next_state_q_values.max(dim=-1, keepdim=True).values
                target_q = reward + self.gamma * max_next_q_values * (1- done)
            
            q_tensor = self.online_net(obs)
            q_values = q_tensor.gather(1, action)
            loss = self.loss_fn(q_values, target_q) 
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.max_norm)
            self.optimizer.step()
            mean_loss = torch.mean(loss).item()

            if global_step % self.cfg['target_update_every'] == 0:
                self.target_net.load_state_dict(self.online_net.state_dict())

            return {'loss': mean_loss}
                
        return None
    
    def on_episode_end(self,)-> None:
        '''
        Optional per episode hook
        Examples:
          - finalize trajectory advantages/returns,
          - update exploration schedules,
          - reset episode-specific accumulators.
        '''
        self.gamma = self.reset_gamma 

        # self.eps = max(self.cfg.get('eps_min', 0.05),
        #            self.eps * self.cfg.get('eps_decay', 0.995))

        return

    def state_dict(self,)-> dict[str, Any]:
        '''
        Return model parameters
        '''    
        self.online_dict = self.online_net.state_dict()    
        self.target_dict =  self.target_net.state_dict()
        self.optimizer_dict = self.optimizer.state_dict()
        return {'online': self.online_dict, 'target': self.target_dict, 'optimizer': self.optimizer_dict}
    
    def load_state_dict(self, state: dict[str, Any])-> None:

        assert state['online'], 'Online net model parameters not found'
        self.online_net.load_state_dict(state_dict= state['online']) 
        assert state['target'], 'Target net model parameters not found'
        self.target_net.load_state_dict(state_dict= state['target']) 
        assert state['optimizer'], 'Optimizer parameters not found'
        self.optimizer.load_state_dict(state_dict= state['optimizer']) 

        return 
    
    def on_train_start(self, cfg):
        '''
        For initialization during training
        '''
        self.online_net.train()
        self.target_net.eval()
        self.warmup_steps = int(self.cfg.get('warmup_steps', 1_000))  # steps before any training/decay

        return
    
    def on_train_end(self,global_step ):
        '''
        Optional: train logic end
        '''
        if global_step >= self.warmup_steps:
            denom = max(1, self.cfg['eps_decay_steps'])
            frac = min(1.0, (global_step - self.warmup_steps) / denom)
            self.eps = float(self.cfg['eps_start'] + frac * (self.cfg['eps_end'] - self.cfg['eps_start']))

        return
    