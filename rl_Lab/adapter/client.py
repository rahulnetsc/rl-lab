import numpy as np

class Client:
    def __init__(self, obs_dim, act_dict: dict = {}) -> None:
        self._t = 0
        self.obs_dim = obs_dim
        self._state = None 
        pass

    def reset(self, options = None):
        self._t = 0
        self._state = np.zeros(self.obs_dim, dtype= np.float32)
        return self._state.copy()
    
    def step(self,action):
        pass
    
    def close(self,):
        pass 



