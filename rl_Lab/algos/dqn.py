import torch 
import torch.nn as nn

from torch.distributions import Categorical

import numpy as np
import gymnasium as gym

from collections import defaultdict
from typing import Any

from .base import Algo
from ..nets import MLP, CNN


class DQN(Algo):
    def __init__(self, obs_dim, num_actions, env: gym.Env, device : torch.device, seed = 42, cfg = None) -> None:
        super().__init__()
        self.online_net = MLP(in_dim= obs_dim, out_dim= num_actions).to(device=device)
        self.target_net = MLP(in_dim= obs_dim, out_dim= num_actions).to(device=device)

        self.replay_buffer = []

        self.init_model_weights()
        self.env = env 
    
        self.rng = np.random.default_rng(seed=seed)
        self.num_actions = num_actions
        self.act_rng = np.random.default_rng()
        self.batch_sampler = np.random.default_rng()
        self.loss_fn = nn.MSELoss()

        self.max_norm = 10
        self.device = device
        self.lr = 0.01

    def init_model_weights(self,):
        # To be implemented
        # self.online_net.apply(), self.target_net.apply()
        pass
    
    def select_action(self, obs_np : np.ndarray, eval_mode: bool = False)->  Any:
        '''
        Given the current observation, return an action to take in the environment.
        - Convert obs_np into a tensor in the correct device, run the policy or Qnet 
            and obtain an environment compatible action
        - Handle exploration/exploitation with eval_mode
        - MUST NOT call env.step() here (or anywhere in the algo). The trainer owns env stepping.
        '''
        raise NotImplementedError

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
        return None 
    
    def update(self,)-> dict[str, float]| None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).
        '''
        return None
    
    def on_episode_end(self,)-> None:
        '''
        Optional per episode hook
        Examples:
          - finalize trajectory advantages/returns,
          - update exploration schedules,
          - reset episode-specific accumulators.
        '''
        return

    def state_dict(self,)-> dict[str, Any]:
        '''
        Return model parameters
        '''        
        return {}
    
    def load_state_dict(self, state: dict[str, Any])-> None:
        return 
    
    def act(self, obs_np: np.ndarray, gamma: float, eps: float = 0.1, eval_mode: bool = False):
         
        
        if self.rng.random() < eps:
            action = self.act_rng.integers(low=0, high= self.num_actions)
        else:
            obs_tensor = torch.tensor(obs_np, device= self.device)
            logits = self.online_net(obs_tensor)
            policy_distribution = Categorical(logits= logits)
            action = policy_distribution.sample()

        done = False
        next_obs, rew, terminated, truncated, info = self.env.step(action)
        Q_s = self.target_net(obs_tensor)

        if terminated or truncated:
            done = True
        
        transition = {
            "obs": obs_np,
            "act": action,
            "next_obs": next_obs,
            "rew": rew,
            "gamma": 0 if done else gamma,
            "done": done,
            "Q_s": Q_s
        }

        self.step_update(transition=transition)
        return action
              
    def step_update(self,transition: dict):
        self.replay_buffer.append(transition)
        
    def episode_update(self, episode: int):
        pass 
    
    def algo_train(self, batch_size: int = 64, target_update: bool = False):

        optimizer = torch.optim.Adam(self.online_net.parameters(), lr= self.lr)
        loss = 0.0
        buffer_size = len(self.replay_buffer)
        batch_idxs = self.batch_sampler.integers(low= 0, high= buffer_size, size= batch_size)

        # Stepping one by one through the batch, inefficient needs proper batch creation
        for idx in batch_idxs:
            r_i = self.replay_buffer[idx]["rew"]
            a_i = self.replay_buffer[idx]["act"]
            s_i = self.replay_buffer[idx]["obs"]
            gamma_i = self.replay_buffer[idx]["gamma"]
            done = self.replay_buffer[idx]["done"]

            max_Q = torch.max(self.replay_buffer[idx]["Q_s"], dim= -1)
            y_i = r_i + gamma_i * max_Q * (1-done)
            Q_pred = self.online_net(s_i)[a_i]
            
            loss += self.loss_fn(Q_pred, y_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if target_update:
            self.target_net.load_state_dict(self.online_net.state_dict())


        


        