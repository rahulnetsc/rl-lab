import torch.nn as nn
import gymnasium as gym 
import numpy as np
import argparse
import yaml

from torch.utils.tensorboard import SummaryWriter

from algos import DQN
from algos.base import Algo
from utils import dev_sel

import os 
class trainer:
    def __init__(self, cfg, algo_cfg, args, env = None, eval_env = None):
        super().__init__()

        self.env: gym.Env = env or gym.make(cfg['trainer']['env_id'])
        self.eval_env: gym.Env = env or gym.make(cfg['trainer']['env_id'])
        
        self.obs_dim = int(np.prod(np.array(self.env.observation_space.shape)))
        self.action_dim = self.env.action_space.n if isinstance(self.env.action_space,gym.spaces.Discrete) else None 
        
        if args.algo == 'dqn':
            self.algo = DQN(env = self.env, obs_dim=self.obs_dim, num_actions= self.action_dim, cfg= algo_cfg, device= args.device)
        else: 
            raise NotImplementedError(f"Algo: {args.algo} not implemented")

        os.makedirs(cfg['ckpt_dir'], exist_ok= True)
        os.makedirs(cfg['log_dir'], exist_ok= True)
        self.writer = SummaryWriter(log_dir= cfg['log_dir'])        

        self.global_step = 0
        self.episode_idx = 0
        self.seed = cfg['trainer']['seed']
        self.rng = np.random.default_rng(seed= self.seed)
        self.cfg = cfg

    def train(self, ):
        
        obs, _ = self.env.reset(seed= self.seed)
        ep_ret, ep_len = 0.0, 0
        last_eval = 0
        last_log = 0

        while self.global_step < self.cfg['max_steps']:

            action = self.algo.select_action(obs_np=obs)
            next_obs, reward, terminated, truncated, info= self.env.step(action= action)
            done = bool(terminated or truncated)

            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done,
                'step': self.global_step,
            }
            self.algo.observe(transition=transition)

            metrics = self.algo.update() or {}

            if self.global_step - last_log == self.cfg['log_every']:
                print(f"[Episode: {self.episode_idx}], ep_len: {ep_len}, global_step: {self.global_step}")
                kv = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
                print(kv)
                last_log = self.global_step

            if self.global_step - last_eval == self.cfg['eval_every']:
                eval_metrics: dict = self.eval_policy() or {}
                print(f"Eval at {self.global_step}")
                kv = ", ".join(f"{k}: {v:.3f}" for k, v in eval_metrics.items())
                print(kv)

            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            self.global_step += 1

            if done:
                print(f"[Episode: {self.episode_idx}], episode_len = {ep_len}, return = {ep_ret:.3f}")
                self.algo.on_episode_end()
                ep_ret, ep_len = 0.0, 0
                self.episode_idx += 1

    def eval_policy(self,)-> dict| None:
        return 

if __name__ == '__main__':

    default_env = gym.make('MountainCar-v0')
    default_algo = 'dqn'
    default_cfg_path = 'configs/base.yaml'
    device = dev_sel()

    parser = argparse.ArgumentParser()
    parser.add_argument('--env',help="Open AI Gym env for Environment", default= default_env)
    parser.add_argument('--algo', help= "RL algorithm: default= dqn, choices={'dqn','ppo','ddpg','a2c'}", default= default_algo, choices={'dqn','ppo','ddpg','a2c'})
    args = parser.parse_args()
    args.device = device

    print(f"Running ")
    try:
        with open(default_cfg_path,'r') as f:
            trainer_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file from {default_cfg_path}: {e}")

    if args.algo == 'dqn':
        algo_cfg_path = 'configs/dqn.yaml'
        with open(algo_cfg_path, 'r') as f:
            algo_cfg = yaml.safe_load(f)
        
    else: 
        raise NotImplementedError(f"Algo: {args.algo} not implemented")
    
    agent = trainer(cfg=trainer_config, algo_cfg= algo_cfg, args= args)

        