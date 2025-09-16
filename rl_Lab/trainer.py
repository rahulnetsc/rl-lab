import torch.nn as nn
import gymnasium as gym 
import numpy as np
import argparse

from algos import PPO, DQN, REINFORCE

class RLAgent():
    def __init__(self, env: gym.Env, algo, eval_env: gym.Env|None = None):
        super().__init__()
        self.env = env
        self.algo = algo
        self.eval_env = eval_env or env    
        
    def train(self, total_steps: int):
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
    # agent = RLAgent(n_eps= args.n_eps, env= args.env, algo= args.algo)

        