import numpy as np

class RolloutBuffer:
    def __init__(self, capacity, obs_shape, action_dim, dtype = np.float32, store_uint8 = False, store_value_fn: bool = False, store_ep_info: bool = False) -> None:
        self._capacity = capacity
        
        self._size = 0
        self._ptr = 0

        self.obs_shape = tuple(obs_shape)
        
        self.store_value_fn = store_value_fn
        self.store_ep_info = store_ep_info
        
        obs_dtype = np.uint8 if store_uint8 else dtype 
        self.obs = np.empty((capacity, *obs_shape), dtype= obs_dtype)
        self.next_obs = np.empty((capacity, *obs_shape), dtype= obs_dtype)
        self.action = np.empty(capacity, dtype= np.int64)
        self.reward = np.empty(capacity, dtype= np.float32)
        self.value = np.empty(capacity, dtype= np.float32)
        self.reward = np.empty(capacity, dtype= np.float32)
        self.logits = np.empty((capacity, action_dim), dtype= np.float32)

        if self.store_ep_info:
            self.done = np.empty(capacity, dtype= np.bool_)
            self.truncated = np.empty(capacity, dtype= np.bool_)
            self.step = np.empty(capacity, dtype= np.int64)

    @property
    def capacity(self): return self._capacity

    def push(self, transition: dict):
        assert transition["obs"].shape == self.obs_shape, f"Shape mismatch between transition['obs'].shape={transition['obs'].shape} and self.obs_shape = {self.obs_shape} "
        assert transition["next_obs"].shape == self.obs_shape, f"Shape mismatch between transition['next_obs'].shape={transition['next_obs'].shape} and self.obs_shape = {self.obs_shape} "
        
        self.obs[self._ptr] = transition['obs']
        self.next_obs[self._ptr] = transition['next_obs']
        self.action[self._ptr] = transition['action']
        self.reward[self._ptr] = transition['reward']
        self.value[self._ptr] = transition['value']
        self.logits[self._ptr] = transition['logits']

        if self.store_ep_info:
            self.done[self._ptr] = transition['done']
            self.truncated[self._ptr] = transition['truncated']
            self.step[self._ptr] = transition['step']
    
        self._ptr += 1
        self._size = min(self._size+1, self._capacity) 

    def __len__(self,):
        return self._size

    def can_sample(self, ):
        return bool(self._ptr < self.capacity)

    def sample(self, ):
        
        trajectory = {
            'obs': self.obs, 
            'action': self.action, 
            'reward': self.reward, 
            'next_obs': self.next_obs,
            'value': self.value,
            'logits': self.logits
        }