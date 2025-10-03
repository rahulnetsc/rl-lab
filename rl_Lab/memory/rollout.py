import numpy as np

class RolloutBuffer:
    def __init__(self, max_steps: int, obs_shape, action_dim, dtype = np.float32, store_uint8 = False, store_value_fn: bool = False, store_ep_info: bool = False) -> None:
        
        self.max_steps: int = max_steps
        
        self._size = 0
        self._ptr = 0

        self.obs_shape = tuple(obs_shape)
        
        self.store_value_fn = store_value_fn
        self.store_ep_info = store_ep_info
        
        self.max_steps = max_steps
        
        self.obs_dtype = np.uint8 if store_uint8 else dtype 
        self.obs = np.empty((self.max_steps, *self.obs_shape), dtype= self.obs_dtype)
        self.next_obs = np.empty((self.max_steps, *self.obs_shape), dtype= self.obs_dtype)
        self.action = np.empty(self.max_steps, dtype= np.int64)
        self.reward = np.empty(self.max_steps, dtype= np.float32)
        self.value = np.empty(self.max_steps, dtype= np.float32)
        self.reward = np.empty(self.max_steps, dtype= np.float32)
        self.logp = np.empty(self.max_steps, dtype= np.float32)

        self.last_step = False

        if self.store_ep_info:
            self.done = np.empty(self.max_steps, dtype= np.bool_)
            self.truncated = np.empty(self.max_steps, dtype= np.bool_)
            self.step = np.empty(self.max_steps, dtype= np.int64)
        
    @property
    def capacity(self): return self.max_steps

    def clear(self):
        self._ptr = 0
        self._size = 0
        self.obs = np.empty((self.max_steps, *self.obs_shape), dtype= self.obs_dtype)
        self.next_obs = np.empty((self.max_steps, *self.obs_shape), dtype= self.obs_dtype)
        self.action = np.empty(self.max_steps, dtype= np.int64)
        self.reward = np.empty(self.max_steps, dtype= np.float32)
        self.value = np.empty(self.max_steps, dtype= np.float32)
        self.reward = np.empty(self.max_steps, dtype= np.float32)
        self.logp = np.empty(self.max_steps, dtype= np.float32)

        self.last_step = False

        if self.store_ep_info:
            self.done = np.empty(self.max_steps, dtype= np.bool_)
            self.truncated = np.empty(self.max_steps, dtype= np.bool_)
            self.step = np.empty(self.max_steps, dtype= np.int64)
        

    def push(self, transition: dict):
        assert transition["obs"].shape == self.obs_shape, f"Shape mismatch between transition['obs'].shape={transition['obs'].shape} and self.obs_shape = {self.obs_shape} "
        assert transition["next_obs"].shape == self.obs_shape, f"Shape mismatch between transition['next_obs'].shape={transition['next_obs'].shape} and self.obs_shape = {self.obs_shape} "
        
        self.obs[self._ptr] = transition['obs']
        self.next_obs[self._ptr] = transition['next_obs']
        self.action[self._ptr] = transition['action']
        self.reward[self._ptr] = transition['reward']
        self.value[self._ptr] = transition['value']
        self.logp[self._ptr] = transition['log_p']

        if self.store_ep_info:
            self.done[self._ptr] = transition['done']
            self.truncated[self._ptr] = transition['truncated']
            self.step[self._ptr] = transition['step']
        self.last_step = transition['last_step']
        self._ptr = self._ptr+1
        self._size = min(self._size+1, self.max_steps) 

    def __len__(self,):
        return self._size

    def buffer_overflow(self, ):
        return bool(self._ptr >= self.max_steps)

    def sample(self, ):
        n= self._size
        trajectory = {
            'obs': self.obs[:n], 
            'action': self.action[:n], 
            'reward': self.reward[:n], 
            'next_obs': self.next_obs[:n],
            'value': self.value[:n],
            'log_p': self.logp[:n],
            'done': self.done[:n] if self.store_ep_info else None,
            'truncated': self.truncated[:n] if self.store_ep_info else None,
        }
        return trajectory