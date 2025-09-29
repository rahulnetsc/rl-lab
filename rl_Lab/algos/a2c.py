import torch 
import torch.nn as nn
from torch.distributions import Categorical
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
        
        self.gamma = cfg['gamma']
        # if not atari env or env requiring otherwise cnn, need a way to autmate this
        if len(self.obs_shape) == 1:
            # vector_obs -> MLP
            self.feature_layer = MLP(in_dim= obs_dim, out_dim= state_dim).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.rollout = RolloutBuffer(capacity= cfg["rollout_size"],obs_shape= self.obs_shape, action_dim= action_dim, store_ep_info = True)
        else:
            # CNN size = (channels,H,W) 
            self.feature_layer = CNN(in_channels= obs_shape[0], out_channels= state_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.rollout = RolloutBuffer(capacity= cfg["rollout_size"],obs_shape= self.obs_shape, action_dim= action_dim, store_uint8 = True,            store_ep_info = True)

        self.init_model_weights()
 
        self.rng = np.random.default_rng(seed=cfg['seed'])
        
        self.act_rng = np.random.default_rng()
        self.batch_sampler = np.random.default_rng()

        self.value_loss_fn = nn.MSELoss()
        
        self.device = device
        self.lr = float(cfg.get('lr', 0.01))
        self.max_norm = cfg['max_norm']

        self.optimizer = torch.optim.Adam(it.chain(self.feature_layer.parameters(),
                                                   self.policy_head.parameters(),
                                                   self.value_head.parameters()),
                                            lr= self.lr) 
        self.feature_dict = None
        self.value_dict = None
        self.policy_dict = None
        self.optimizer_dict = None 
        self._cache = None 

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
        if eval_mode:
            return int(torch.argmax(logits, dim =-1).item())
        else:
            action = self.act_rng.choice(self.action_dim, p= logits) 
            value = self.value_head(obs_tensor)
            self._cache = {
                'logits': logits,
                'value': value,
            }
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
        if self._cache:
            transition['value'] = self._cache['value']
            transition['logit'] = self._cache['logits']
        self.rollout.push(transition= transition) 
    
    def update(self, global_step) -> dict[str, float] | None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).
        '''
        pass 
    

    def on_episode_end(self,)-> dict[str, float]:
        '''
        Optional per episode hook
        Examples:
          - finalize trajectory advantages/returns,
          - update exploration schedules,
          - reset episode-specific accumulators.
        '''
        trajectory = self.rollout.sample()
        assert trajectory, 'EMPTY BUFFER'
        obs_tensor = torch.as_tensor(trajectory['obs'],device= self.device)
        _value = trajectory['value']
        _reward = trajectory['reward']

        advantage, net_return = self.compute_adv_ret(values = _value, rewards = _reward)
        value_new = self.value_head(obs_tensor)
        logit_new = self.policy_head(obs_tensor)
        value_loss = self.value_loss_fn(value_new,net_return)
        adv_loss = -(advantage* logit_new).mean()
        entropy_loss = -Categorical(logits=logit_new).entropy()
        loss = value_loss + self.cfg['value_coef'] * adv_loss + self.cfg['entropy_coef'] * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(it.chain(self.feature_layer.parameters(),
                                                   self.policy_head.parameters(),
                                                   self.value_head.parameters()), 
                                                   self.max_norm)
        self.optimizer.step()
        metrics = {
                'val_loss': value_loss,
                'adv_loss': adv_loss,
                'logits_max': float(logit_new.max().item()),
            }
        return metrics

    def state_dict(self,)-> dict[str, Any]:
        '''
        Return model parameters
        '''    
        self.feature_dict = self.feature_layer.state_dict()
        self.value_dict = self.value_head.state_dict()
        self.policy_dict = self.policy_head.state_dict()
        self.optimizer_dict = self.optimizer.state_dict() 
        return {'feature': self.feature_dict,
                'value': self.value_dict,
                'policy': self.policy_dict
                }
    
    def load_state_dict(self, state: dict[str, Any])-> None:
            assert state['feature'] , "Feature parameters not found"
            assert state['value'] , "Value head parameters not found"
            assert state['policy'] , "Policy head parameters not found"

            self.feature_layer.load_state_dict(state_dict= state['feature'])
            self.value_head.load_state_dict(state_dict= state['value'])
            self.policy_head.load_state_dict(state_dict= state['policy'])
            return
    
    def on_train_start(self, cfg):
        '''
        For initialization during training
        '''
        self.feature_layer.train()
        self.value_head.train()
        self.policy_head.train()
        
        self.warmup_steps = int(self.cfg.get('warmup_steps', 1_000))  # steps before any training/decay 
    
    def on_train_end(self,global_step ):
        '''
        Optional: train logic end
        '''
        pass 
    
    def compute_adv_ret(self, values, rewards):
        with torch.no_grad():
            advs = torch.zeros_like(rewards,device=self.device)
            
            last_v = 0.0
            next_v = last_v
            adv_next = 0.0
            for value, reward, adv in reversed(zip(values,rewards, advs)):
                adv = reward + self.gamma * next_v - value + self.gamma * adv_next
                adv_next = adv 
                next_v = value 
            net_return = advs + values 

        return advs, net_return

