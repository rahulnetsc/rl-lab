import torch.nn as nn
import gymnasium as gym 
import numpy as np
import argparse

from rl_algorithms import PPO, DQN, REINFORCE

class RLAgent():
    def __init__(self, n_eps, env : gym.Env, algo):
        super().__init__()
        self.env = env
        self.algo = algo(n_eps)
        self.actions = None
        self.lr = 0.01
        self.init_env()
        
    def init_env(self,):
        _obs, _info = self.env.reset() 
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.actions = np.arange(self.env.action_space.n)
        elif isinstance(self.env.action_space, gym.spaces.Box):
            self.actions = None
        else:
            raise NotImplementedError(f"Action space: type(self.env.action_space)={type(self.env.action_space)} not implemented")
        
    def train(self,):
        self.algo.train()

    def eval(self,):
        pass
    
if __name__ == '__main__':

    default_env = gym.make('MountainCar-v0')
    default_algo = PPO
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',help="Open AI Gym env for Environment", default= default_env)
    parser.add_argument('--algo', help= "RL algorithm default= PPO", default= default_algo)
    parser.add_argument('--n_eps', help= "Number of training episodes", default= 1000)
    args = parser.parse_args()

    print(f"Running ")
    agent = RLAgent(n_eps= args.n_eps, env= args.env, algo= args.algo)

        