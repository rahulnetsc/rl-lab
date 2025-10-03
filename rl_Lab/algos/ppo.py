import torch 
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

import itertools as it

from typing import Any
from .base import Algo
from ..nets import MLP, CNN
from ..memory import RolloutBuffer

class PPO(Algo):
    def __init__(self, obs_dim, obs_shape, state_dim, action_dim, cfg, device) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.obs_shape = obs_shape
        self.action_dim = action_dim

        if self.action_dim is None:
            raise ValueError("PPO supports only discrete action spaces. Got: "
                            f"{type(self.action_dim)}")
        
        self.cfg = cfg
        self.train_epochs = cfg['train_epochs']
        self.minibatch_size = cfg['minibatch_size']
        
        self.gamma = cfg['gamma']
        self.td_lambda = cfg['td_lambda']
        self.max_steps = cfg['max_steps']
        self.clip_eps =cfg['clip_eps']
        
        # if not atari env or env requiring otherwise cnn, need a way to autmate this
        if len(self.obs_shape) == 1:
            # vector_obs -> MLP
            self.feature_layer = MLP(in_dim= obs_dim, out_dim= state_dim).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.rollout = RolloutBuffer(max_steps= self.max_steps,obs_shape= self.obs_shape, action_dim= action_dim, store_ep_info = True)
        else:
            # CNN size = (channels,H,W) 
            self.feature_layer = CNN(in_channels= obs_shape[0], out_channels= state_dim, img_height= obs_shape[1], img_width= obs_shape[2]).to(device=device)
            self.policy_head = MLP(in_dim= state_dim, out_dim= action_dim).to(device=device)
            self.value_head = MLP(in_dim= state_dim, out_dim= 1).to(device=device)
            self.rollout = RolloutBuffer(max_steps= self.max_steps,obs_shape= self.obs_shape, action_dim= action_dim, store_uint8 = True,            store_ep_info = True)

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
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_np, device= self.device, dtype= torch.float32).unsqueeze(0)
            out_feat = self.feature_layer(obs_tensor)
            logits = self.policy_head(out_feat)
            dist = Categorical(logits= logits)
            action = torch.argmax(logits, -1) if eval_mode else dist.sample()
            log_p = dist.log_prob(action)
            value = self.value_head(out_feat).squeeze(-1)
            self._cache = {'log_p': float(log_p.item()), 'value': float(value.item()), 'action': int(action.item())}

        return int(action.item())

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
            transition['action'] = self._cache['action']
            transition['log_p'] = self._cache['log_p']
            transition['last_step'] = bool(transition['done'] or transition['truncated'])
        self.rollout.push(transition)
    
    def update(self, global_step) -> dict[str, float] | None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'train/loss': 0.123}) or None
        - Off policy (e.g., DQN):  usually learn every step after warmup by sampling from replay.
        - On policy (e.g., PPO, A2C): learn only when a rollout/episode is complete (or length T reached).
        '''
        if self.rollout.buffer_overflow() or self.rollout.last_step:
            trajectory = self.rollout.sample()
            
            assert trajectory, 'EMPTY BUFFER'

            _values = torch.as_tensor(trajectory['value'], device= self.device, dtype= torch.float32)
            _reward = torch.as_tensor(trajectory['reward'], device= self.device, dtype= torch.float32)
            _done = torch.as_tensor(trajectory['done'], device= self.device)

            _obs = torch.as_tensor(trajectory['obs'], device= self.device, dtype= torch.float32)
            _act = torch.as_tensor(trajectory['action'], device= self.device, dtype= torch.long)
            _log_p_old = torch.as_tensor(trajectory['log_p'], device= self.device, dtype= torch.float32)
            
            last_done  = bool(trajectory['done'][-1])        # true terminal
            last_trunc = bool(trajectory['truncated'][-1])   # time-limit
            if last_done and not last_trunc:
                last_v = 0.0
            else:
                no = torch.as_tensor(trajectory['next_obs'][-1], device=self.device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    h_no = self.feature_layer(no)
                    last_v = float(self.value_head(h_no).item())
            advantage, net_return = self.compute_adv_ret(values = _values, rewards = _reward, done= _done, last_v= last_v)

            if self.cfg.get('normalize_adv', True):
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            num_samples = len(_reward)
            indices = np.arange(num_samples)
            log_policy_loss, log_value_loss, log_entropy, logger_loss = [], [], [], []
            eps_v = float(self.cfg.get('value_clip_eps', 0.0))  # if 0, you can skip clipping
            self.target_kl = self.cfg.get('target_kl', None)

            for _ in range(self.train_epochs):

                epoch_kls = []
                self.rng.shuffle(indices) 

                for start in range(0, num_samples, self.minibatch_size):
                    mb = indices[start:start + self.minibatch_size]
                    mb_idx = torch.as_tensor(mb, device=self.device, dtype=torch.long)

                    obs_mb = _obs[mb_idx]
                    act_mb = _act[mb_idx]
                    old_log_p_ = _log_p_old[mb_idx]
                    adv_mb =  advantage[mb_idx]
                    ret_mb = net_return[mb_idx]
                    v_old_mb = _values[mb_idx]

                    # Forward pass with current policy
                    h = self.feature_layer(obs_mb)
                    logits = self.policy_head(h)
                    dist = Categorical(logits=logits)
                    new_log_p = dist.log_prob(act_mb)

                    ratio = torch.exp(new_log_p-old_log_p_)
                    unclipped = ratio * adv_mb
                    clipped = torch.clamp(ratio, min=1.0-self.clip_eps, max= 1.0 + self.clip_eps)* adv_mb
                    policy_loss = -torch.min( unclipped, clipped).mean()

                    # Clipped value loss
                    # value_loss = self.value_loss_fn(v_new, old_ret)
                    v_new     = self.value_head(h).squeeze(-1)     # current values
                    v_new = self.value_head(h).squeeze(-1)
                    if eps_v > 0.0:
                        v_clip = v_old_mb + (v_new - v_old_mb).clamp(-eps_v, eps_v)
                        v_loss_unclipped = (v_new  - ret_mb).pow(2)
                        v_loss_clipped   = (v_clip - ret_mb).pow(2)
                        value_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                    else:
                        value_loss = self.value_loss_fn(v_new, ret_mb)


                    entropy = dist.entropy().mean()

                    loss = policy_loss + self.cfg['value_coef'] * value_loss - self.cfg['entropy_coef'] * entropy

                    log_policy_loss.append(policy_loss.item())
                    log_value_loss.append(value_loss.item())
                    log_entropy.append(entropy.item())
                    logger_loss.append(loss.item())

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(it.chain(self.feature_layer.parameters(),
                                                    self.policy_head.parameters(),
                                                    self.value_head.parameters()), 
                                                    self.max_norm)
                    
                    self.optimizer.step()
                    with torch.no_grad():
                        approx_kl = (old_log_p_ - new_log_p).mean()
                    epoch_kls.append(approx_kl.item())

                if (self.target_kl is not None) and (np.mean(epoch_kls) >= float(self.target_kl)):
                    break 

            self.rollout.clear()

            metrics = {
                'loss': float(np.mean(logger_loss)),
                'policy_loss': float(np.mean(log_policy_loss)),
                'value_loss': float(np.mean(log_value_loss)),
                'entropy': float(np.mean(log_entropy)),
                'approx_kl': float(np.mean(epoch_kls))
            }


            return metrics
        return None
    

    def on_episode_end(self,):
        '''
        Optional per episode hook
        Examples:
          - finalize trajectory advantages/returns,
          - update exploration schedules,
          - reset episode-specific accumulators.
        '''
        pass 

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
    
    def compute_adv_ret(self, values, rewards, done, last_v ):

        with torch.no_grad():    
            v = torch.as_tensor(values, device=self.device, dtype= torch.float32)
            r = torch.as_tensor(rewards, device=self.device, dtype= torch.float32)
            d = torch.as_tensor(done, device=self.device, dtype= torch.float32)
            advs = torch.zeros_like(r, device=self.device, dtype= torch.float32)
            gamma = self.gamma
            lam = self.td_lambda
            T = len(rewards)
            
            next_v = torch.as_tensor(last_v, device= self.device, dtype= torch.float32)
            adv_next = torch.zeros((), device=self.device, dtype=torch.float32)  
            
            for t in reversed(range(T)):
                m_t = 1.0-d[t].float()
                # delta_t = r_t + gamma * V(s_(t+1)) * (1-done_t+1) - V(s_t)
                delta_t = r[t] + gamma * next_v * m_t - v[t]

                # A_t = delta_t + gamma * lambda * A_t+1 * (1-done_t+1) 
                advs[t] = delta_t + gamma * lam * adv_next * m_t
                adv_next = advs[t]
                next_v = v[t]
                
            net_return = advs + v 
        return advs, net_return
    


