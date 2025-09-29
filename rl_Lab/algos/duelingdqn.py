import torch 
import torch.nn as nn
import numpy as np

import itertools as it

from typing import Any
from .base import Algo
from ..nets import MLP, CNN
from ..memory import ReplayBuffer

class DuelingDQN(Algo):
    def __init__(self, obs_dim, obs_shape, state_dim, action_dim, cfg, device) -> None:
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
        self.tau = cfg['tau']
        self.polyak = cfg['polyak']
        self.target_update_every = cfg['target_update_every']

        # if not atari env or env requiring otherwise cnn, need a way to autmate this
        if len(self.obs_shape) == 1:
            # vector_obs -> MLP
            self.target_feature_layer = MLP(in_dim= obs_dim, out_dim= state_dim).to(device=device)
            self.online_feature_layer = MLP(in_dim= obs_dim, out_dim= state_dim).to(device=device)
            
            self.online_value_fn = MLP(in_dim= state_dim, out_dim= 1).to(device= device)
            self.online_adv = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)

            self.target_value_fn = MLP(in_dim= state_dim, out_dim= 1).to(device= device)
            self.target_adv = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            
            self.replay = ReplayBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_ep_info = True)
        else:
            # CNN size = (channels,H,W) 
            self.online_feature_layer = CNN(in_channels= obs_shape[0], out_channels= state_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.target_feature_layer = CNN(in_channels= obs_shape[0], out_channels= state_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)

            self.online_value_fn = MLP(in_dim= state_dim, out_dim= 1).to(device= device)
            self.online_adv = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)

            self.target_value_fn = MLP(in_dim= state_dim, out_dim= 1).to(device= device)
            self.target_adv = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            
            self.replay = ReplayBuffer(capacity= cfg["buffer_size"],obs_shape= self.obs_shape, store_uint8 = True,  store_ep_info = True)

        self.init_model_weights()
 
        self.rng = np.random.default_rng(seed=cfg['seed'])
        
        self.act_rng = np.random.default_rng()
        self.batch_sampler = np.random.default_rng()
        self.loss_fn = nn.SmoothL1Loss()

        self.device = device
        self.lr = float(cfg.get('lr', 0.01))
        self.optimizer = torch.optim.Adam(it.chain(self.online_feature_layer.parameters(),
                                                   self.online_value_fn.parameters(),
                                                   self.online_adv.parameters()), 
                                            lr=self.lr)
        self.online_dict = None    
        self.target_dict =  None
        self.optimizer_dict = None

    @staticmethod
    def combine_dueling(v, a): 
        return v + (a - a.mean(dim=-1, keepdim=True))

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

        self.online_feature_layer.apply(init_fn)
        self.online_value_fn.apply(init_fn)
        self.online_adv.apply(init_fn)

        # Start target identical to online
        self.target_feature_layer.load_state_dict(self.online_feature_layer.state_dict())
        self.target_value_fn.load_state_dict(self.online_value_fn.state_dict())
        self.target_adv.load_state_dict(self.online_adv.state_dict())

    def select_action(self, obs_np: np.ndarray, eval_mode: bool = False):
        '''
        Given the current observation, return an action to take in the environment.
        - Convert obs_np into a tensor in the correct device, run the policy or Qnet 
            and obtain an environment compatible action
        - Handle exploration/exploitation with eval_mode
        - MUST NOT call env.step() here (or anywhere in the algo). The trainer owns env stepping.
        '''
        if not eval_mode and self.rng.random() < self.eps:
            return self.act_rng.integers(low=0, high= self.action_dim)
        
        obs_tensor = torch.as_tensor(obs_np, device= self.device).unsqueeze(0)
        phi = self.online_feature_layer(obs_tensor)
        value = self.online_value_fn(phi)
        adv = self.online_adv(phi)
        q = DuelingDQN.combine_dueling(value,adv)
        action =  int(torch.argmax(q, dim=-1).item())
        return action
    
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
            "step" : int                        # global step
            }
        Typical usage:
            - Off policy: push to replay buffer
            - On policy: append to rollout buffer
        '''
        self.replay.push(transition= transition)
        return None
    
    def update(self, global_step) -> dict[str, float] | None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).
        '''
        if self.replay.can_sample(batch_size=self.batch_size):
            sample = self.replay.sample(batch_size=self.batch_size, rng= self.batch_sampler)
            
            obs      = torch.as_tensor(sample['obs'], device=self.device)
            action   = torch.as_tensor(sample['action'], device=self.device, dtype=torch.long).unsqueeze(1)
            reward   = torch.as_tensor(sample['reward'], device=self.device, dtype=torch.float32).unsqueeze(1)
            next_obs = torch.as_tensor(sample['next_obs'], device=self.device)
            done     = torch.as_tensor(sample['done'], device=self.device, dtype=torch.float32).unsqueeze(1)


            with torch.no_grad():
                # 1) action selection by ONLINE net (indices, not values)
                next_phi_online = self.online_feature_layer(next_obs)
                next_q_online_val = self.online_value_fn(next_phi_online)
                next_q_online_adv = self.online_adv(next_phi_online)
                next_q_online = DuelingDQN.combine_dueling(next_q_online_val,next_q_online_adv) 
                next_actions = next_q_online.argmax(dim =-1, keepdim=True)

                # 2) Action evaluation by TARGET net
                next_phi_target = self.target_feature_layer(next_obs)
                next_q_target_val = self.target_value_fn(next_phi_target)
                next_q_target_adv = self.target_adv(next_phi_target)
                next_q_target = DuelingDQN.combine_dueling(next_q_target_val, next_q_target_adv)
                max_next_q_values = next_q_target.gather(1,next_actions)
                
                target_q = reward + self.gamma * max_next_q_values * (1.0 - done)
            
            q_phi = self.online_feature_layer(obs)
            q_val = self.online_value_fn(q_phi)
            q_adv = self.online_adv(q_phi)
            q_tensor = DuelingDQN.combine_dueling(q_val, q_adv)
            q_values = q_tensor.gather(1, action)
            loss = self.loss_fn(q_values, target_q)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(it.chain(self.online_feature_layer.parameters(),self.online_value_fn.parameters(), self.online_adv.parameters()), self.max_norm)
            self.optimizer.step()
            mean_loss = torch.mean(loss).item()

            # Polyak soft target updates            
            if self.polyak:     
                with torch.no_grad():
                    for t,p in zip(it.chain(self.target_feature_layer.parameters(),self.target_value_fn.parameters(), self.target_adv.parameters()), 
                                   it.chain(self.online_feature_layer.parameters(),self.online_value_fn.parameters(),self.online_adv.parameters())):
                        t.data.mul_(1 - self.tau).add_(self.tau * p.data)
            else: 
                if global_step % self.target_update_every == 0:
                    self.target_feature_layer.load_state_dict(state_dict= self.online_feature_layer.state_dict())
                    self.target_value_fn.load_state_dict(state_dict= self.online_value_fn.state_dict())
                    self.target_adv.load_state_dict(state_dict= self.online_adv.state_dict())

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
        self.online_feature_layer_dict = self.online_feature_layer.state_dict()
        self.online_val_fn_dict = self.online_value_fn.state_dict()    
        self.online_adv_dict = self.online_adv.state_dict()    

        self.target_feature_layer_dict =  self.target_feature_layer.state_dict()
        self.target_val_fn_dict =  self.target_value_fn.state_dict()
        self.target_adv_dict = self.target_adv.state_dict()
        self.optimizer_dict = self.optimizer.state_dict()
        return {'online_feat': self.online_feature_layer_dict, 
                'online_val': self.online_val_fn_dict, 
                'online_adv': self.online_adv_dict,
                'target_feat': self.target_feature_layer_dict,
                'target_val': self.target_val_fn_dict, 
                'target_adv': self.target_adv_dict,
                'optimizer': self.optimizer_dict}
    
    def load_state_dict(self, state: dict[str, Any])-> None:

        assert state['online_feat'], 'Online Feature layer net model parameters not found'
        assert state['online_val'], 'Online Value Fn net model parameters not found'
        assert state['online_adv'], 'Online Adv net model parameters not found'
        self.online_feature_layer.load_state_dict(state_dict= state['online_feat'])
        self.online_value_fn.load_state_dict(state_dict= state['online_val'])
        self.online_adv.load_state_dict(state_dict= state['online_adv'])

        assert state['target_feat'], 'Target Feature layer net model parameters not found'
        assert state['target_val'], 'Target Value Fn net model parameters not found'
        assert state['target_adv'], 'Target Adv net model parameters not found'
        self.target_feature_layer.load_state_dict(state_dict= state['target_feat'])
        self.target_value_fn.load_state_dict(state_dict= state['target_val'])
        self.target_adv.load_state_dict(state_dict= state['target_adv'])

        assert state['optimizer'], 'Optimizer parameters not found'
        self.optimizer.load_state_dict(state_dict= state['optimizer']) 

        return 
    
    def on_train_start(self, cfg):
        '''
        For initialization during training
        '''
        self.online_feature_layer.train()
        self.online_value_fn.train()
        self.online_adv.train()
        self.target_feature_layer.eval()
        self.target_value_fn.eval()
        self.target_adv.eval()

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