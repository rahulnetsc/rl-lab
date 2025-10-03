from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class Algo(ABC):
    @abstractmethod
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
    
    def update(self,global_step)-> dict[str, float]| None:
        '''
        Perform one step gradient update if ready otherwise no op 
        Returns a metrics dict (e.g. {'loss': 0.123}) or None
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

    def on_train_start(self, cfg):
        '''
        For initialization during training
        '''
        return
    
    def on_train_end(self, global_step):
        '''
        Optional: train logic end
        '''
        return
    
    def state_dict(self,)-> dict[str, Any]| None:
        '''
        Return model parameters
        '''        
        return {}
    
    def load_state_dict(self, state: dict[str, Any])-> None:
        return 
