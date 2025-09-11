from abc import ABC, abstractmethod
import numpy as np

class Algo(ABC):
    @abstractmethod
    def act(self, obs_np : np.ndarray, eval_mode: bool = False):
        '''
        Given the current observation, return an action to take in the environment.
        - Convert obs_np into a tensor in the correct device, run the policy or Qnet 
            and obtain an environment compatible action
        - Handle exploration/exploitation with eval_mode
        - OPTIONAL: Cache log_prob, value etc into a rollout buffer 
        '''
        pass

    def step_update(self, transition) -> dict| None:
        '''
        A per environment hook. Off-policy algos updates here, on-policy algos just stores.

        The trainer calls this after every env.step() with the current transition
        Used to perform training in algos that learn every step and from a replay buffer
        - store (obs, act, rew, next_obs, done) in a buffer 
        - sample a minbatch 
        - perform gradient update 
        e.g. trainer passes 
        transition = 
            {
            "obs" : obs_np,
            "act" : action, 
            "rew" : np.ndarray,
            "next_obs" : next_obs_np,
            "done" : bool(done),
            }

        '''

        return None # Off-policy algos override
    
    def episode_update(self, trajectory) -> dict| None:
        '''
        A per episode/rollout hook. The trainer calls this after every episode to update on-policy algos.  
        e.g.:
        trajectory = 
        {
        "obs" : [obs_t, ...],
        "acts" : [act_t, ...],
        "rews" : [rew_t, ...]
        }

        '''
        return None # On-policy algos override

    
