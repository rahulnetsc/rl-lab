import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_shape,  dtype = np.float32, store_uint8 = False, store_value_fn: bool = False, store_ep_info: bool = False) -> None:
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

        if self.store_ep_info:
            self.done = np.empty(capacity, dtype= np.bool_)
            self.truncated = np.empty(capacity, dtype= np.bool_)
            self.step = np.empty(capacity, dtype= np.int64)
        
        if store_value_fn:
            self.val_fn = np.empty(capacity, dtype= np.float32)

    @property
    def capacity(self): return self._capacity

    def push(self, transition: dict):

        assert transition["obs"].shape == self.obs_shape, f"Shape mismatch between transition['obs'].shape={transition['obs'].shape} and self.obs_shape = {self.obs_shape} "
        assert transition["next_obs"].shape == self.obs_shape, f"Shape mismatch between transition['next_obs'].shape={transition['next_obs'].shape} and self.obs_shape = {self.obs_shape} "
        
        self.obs[self._ptr] = transition['obs']
        self.next_obs[self._ptr] = transition['next_obs']
        self.action[self._ptr] = transition['action']
        self.reward[self._ptr] = transition['reward']
        if self.store_ep_info:
            self.done[self._ptr] = transition['done']
            self.truncated[self._ptr] = transition['truncated']
            self.step[self._ptr] = transition['step']
        if self.store_value_fn and ('val_fn' in transition.keys()):
            self.val_fn[self._ptr] = transition['val_fn']

        self._ptr = (self._ptr+1) % self._capacity
        self._size = min(self._size+1, self._capacity)

    def __len__(self,):
        return self._size

    def can_sample(self, batch_size):
        return bool(self._size>= batch_size)

    def sample(self, batch_size, rng: np.random.Generator):

        assert self.can_sample(batch_size=batch_size), f'Not enough samples, buffer size = {self._size}'
        
        idxs= rng.integers(low= 0, high= self._size, size= batch_size)
        sample = { 
            'obs': self.obs[idxs], 
            'action': self.action[idxs], 
            'reward': self.reward[idxs], 
            'next_obs': self.next_obs[idxs]
            }
        
        if self.store_ep_info:
            sample.update({'done': self.done[idxs], 'truncated': self.truncated[idxs], 'step': self.step[idxs]})
        
        if self.store_value_fn:
            sample.update({'val_fn': self.val_fn[idxs]})
                
        return sample



        

