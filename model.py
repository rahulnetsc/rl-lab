import torch.nn as nn
import gymnasium as gym 
import numpy as np
import argparse

class RLAgent(nn.Module):
    def __init__(self, env : gym.Env, model = None):
        super().__init__()
        self.env = env
        self.model = model
        self.actions = None
        self.lr = 0.01

    def init_env(self,):
        _obs, _info = self.env.reset() 
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions = np.arange(self.env.action_space.n)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.actions = None
        else:
            raise NotImplementedError(f"Action space: type(self.env.action_space)={type(self.env.action_space)} not implemented")
        
    def init_model(self,):
        pass

    

if __name__ == '__main__':
    default_env = gym.make('MountainCar-v0')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',help="Open AI Gym env for Environment", default= default_env)
    parser.add_argument('--n_eps', help= "Number of training episodes", default= 1000)
    agent = RLAgent(env= default_env)

        