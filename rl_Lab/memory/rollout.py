import numpy as np

class RolloutBuffer:
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
        pass 

    def __len__(self,):
        return self._size

    def can_sample(self, batch_size):
        return bool(self._size>= batch_size)

    def sample(self, batch_size, rng: np.random.Generator):
        pass 



        

