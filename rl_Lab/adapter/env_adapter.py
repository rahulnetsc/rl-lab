import gymnasium as gym 
from gymnasium import spaces
import numpy as np
from typing import Any

class EnvAdapter(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, client, obs_shape, act_low, act_high, act_disc_dict: dict ,  episode_horizon = 1000) -> None:
        super().__init__()
        self.client = client
        self.observation_space = spaces.Box(low= -np.inf, high= np.inf, shape = obs_shape, dtype= np.float32)
        self.act_low = act_low
        self.act_high = act_high
        action_spaces = {}
        action_spaces['cts_act1'] = spaces.Box(low= self.act_low, high= self.act_high, dtype= np.float32)
        for key, value in act_disc_dict.items():
            action_spaces[f"disc_{key}"] = spaces.Discrete(value),            
        self.action_space = spaces.Dict(action_spaces)
        
        self._t = 0
        self._horizon = episode_horizon

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        obs = self.client.reset(options or {})
        self._t = 0
        obs = np.asarray(obs, dtype=self.observation_space.dtype)
        return obs, {"info_from_system": 'ok'}
    
    def step(self,action):
        
        cts = np.asarray(action['cts_act1'], dtype= np.float32)
        cts = np.clip(cts, self.act_low, self.act_high)

        disc_parts = {k: int(v) for k,v in action.items() if k.startswith('disc_')}
        payload = {'cts_act1': cts.tolist(), **disc_parts}

        out = self.client.step(payload)
        # expect: out = {"obs": ..., "reward": float, "terminated": bool, "truncated": bool, "info": {...}}
        
        self._t += 1
        obs = np.asarray(out['obs'], dtype= self.observation_space.dtype)
        terminated = bool(out.get('terminated', False))
        truncated = bool(out.get('truncated', False) or self._t >= self._horizon)
        reward = float(out["reward"])
        info = out.get("info", {})
        return obs, reward, terminated, truncated, info
    
    def close(self,):
        self.client.close()
        
