import gymnasium as gym
import torch
import numpy as np
from collections import defaultdict
from torch.distributions import Categorical

from .base import Algo
from ..nets import MLP

class A2C(Algo):
    def __init__(self, state_dim, num_actions, env: gym.Env, device : torch.device, seed = 42) -> None:
        super().__init__()
        self.device = device 
        self.actor = MLP(in_dim=state_dim, out_dim= num_actions).to(device)
        self.critic = MLP(in_dim=state_dim, out_dim= 1).to(device)
        self.initialize_model_weights(self.actor)
        self.initialize_model_weights(self.critic)
        self._episode_buffer = defaultdict(list)
        self._replay_buffer = defaultdict(lambda: defaultdict(list))
        self.env = env
    def initialize_model_weights(self, mlp):
        # initialize model parameters : to be implemented
        pass

    def act(self, obs_np: np.ndarray, eval_mode: bool = False, done: bool = False, transition : dict= {}):
        if not eval_mode:
            obs_tensor = torch.tensor(obs_np,device=self.device)
            logits = self.actor(obs_tensor)
            policy_distribution = Categorical(logits=logits)
            action = policy_distribution.sample()
            value_function = self.critic(obs_tensor)
        else:
            # To be implemented
            pass 

        next_obs, rew, terminated, truncated, info = self.env.step(action=action)
        if terminated or truncated:
            done = True
        
        transition["obs"] = obs_np
        transition["act"] = action
        transition["rew"] = rew
        transition["next_obs"] = next_obs
        transition["value"] = value_function 
        self.step_update(transition=transition)
        return action 
    
    def step_update(self, transition: dict) -> dict | None:
        '''
          On policy: store transitions
          transition = 
            {
            "obs" : obs_np,
            "act" : action, 
            "rew" : np.ndarray,
            "next_obs" : next_obs_np,
            "value" : value
            } 
        '''
        self._episode_buffer["obs"].append(transition["obs"])
        self._episode_buffer["act"].append(transition["act"])
        self._episode_buffer["rew"].append(transition["rew"])
        self._episode_buffer["next_obs"].append(transition["next_obs"])
        self._episode_buffer["value"].append(transition["value"])
        return self._episode_buffer
    
    def episode_update(self, episode : int, trajectory: defaultdict) -> dict | None:
        '''
           trajectory = 
            {
            "obs" :     [obs_t, ...],
            "act" :     [act_t, ...],
            "rew" :     [rew_t, ...],
            "next_obs": [next_obs_t, ...],
            } 
        '''
        # Store episode trajectory
        self._replay_buffer[episode] = trajectory 

        # Update 
        self.train()
        return None
    
    def train(self,):
        pass

    def train_actor(self,):
        pass 

        
        