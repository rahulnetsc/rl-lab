import gymnasium as gym 
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter

from algos import DQN
from utils import dev_sel, load_yaml

import os 

class Trainer:
    def __init__(self, cfg, algo_cfg, args, env = None, eval_env = None):
        super().__init__()

        self.env: gym.Env = env or gym.make(cfg['trainer']['env_id'])
        self.eval_env: gym.Env = eval_env or gym.make(cfg['trainer']['env_id'])

        self.obs_shape = np.array(self.env.observation_space.shape)
        self.obs_dim = int(np.prod(np.array(self.env.observation_space.shape)))
        self.action_dim = self.env.action_space.n if isinstance(self.env.action_space,gym.spaces.Discrete) else None 
        
        # need to implement a better logic for algo selection
        if args.algo == 'dqn':
            self.algo = DQN(obs_dim=self.obs_dim, obs_shape = self.obs_shape, action_dim= self.action_dim, cfg= algo_cfg, device= args.device)
        else: 
            raise NotImplementedError(f"Algo: {args.algo} not implemented")

        os.makedirs(cfg['trainer']['ckpt_dir'], exist_ok= True)
        os.makedirs(cfg['trainer']['log_dir'], exist_ok= True)
        self.writer = SummaryWriter(log_dir= cfg['trainer']['log_dir'])        

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

        while self.global_step < self.cfg['trainer']['max_steps']:

            action = self.algo.select_action(obs_np=obs)
            next_obs, reward, terminated, truncated, _ = self.env.step(action= action)
            done = bool(terminated or truncated)

            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                "done": bool(terminated),          # true terminals only
                "truncated": bool(truncated),      # time-limit cutoffs
                'step': self.global_step,
            }
            self.algo.observe(transition=transition)

            metrics = self.algo.update(global_step=self.global_step) or {}

            self.algo.on_train_end(global_step = self.global_step)

            if last_log==0 or self.global_step - last_log >= self.cfg['trainer']['log_every']:
                print(f"[Episode: {self.episode_idx}], ep_len: {ep_len}, global_step: {self.global_step}")
                kv = ", ".join(f"train/{k}: {v:.3f}" for k, v in metrics.items())
                print(kv)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                last_log = self.global_step

            if last_eval==0 or self.global_step - last_eval >= self.cfg['trainer']['eval_every']:
                eval_metrics: dict = self.eval_policy(self.global_step) or {}
                print(f"Eval at {self.global_step}")
                kv = ", ".join(f"eval/{k}: {v:.3f}" for k, v in eval_metrics.items())
                print(kv)
                for k, v in eval_metrics.items():
                    self.writer.add_scalar(f'eval/{k}', v, self.global_step)
                last_eval = self.global_step

            obs = next_obs
            ep_ret += float(reward)
            ep_len += 1
            self.global_step += 1

            if done:
                print(f"[Episode: {self.episode_idx}], episode_len = {ep_len}, return = {ep_ret:.3f}")
                self.writer.add_scalar('train/episode_return', ep_ret, self.global_step)
                self.writer.add_scalar('train/episode_length', ep_len, self.global_step)
                self.algo.on_episode_end()
                ep_ret, ep_len = 0.0, 0
                self.episode_idx += 1
                obs, _ = self.env.reset()
        
        self.env.close(); self.eval_env.close()
        self.writer.flush(); self.writer.close()

    def eval_policy(self,idx)-> dict| None:
        returns, lengths = [], []
        for i in range(self.cfg['trainer']['eval_episodes']):
            obs, _ = self.eval_env.reset(seed=self.seed+1000+idx+i)
            done = False

            ep_ret, ep_len = 0.0, 0
            while not done:
                action = self.algo.select_action(obs_np=obs, eval_mode= True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action=action)
                done = bool(terminated or truncated)
                if self.cfg['trainer']['render_eval']:
                    self.eval_env.render()
                obs = next_obs
                ep_ret += float(reward)
                ep_len += 1


            returns.append(ep_ret)
            lengths.append(ep_len)

        return {
            'return_mean': np.mean(returns),
            'length_mean': np.mean(lengths)
        }

if __name__ == '__main__':

    device = dev_sel()

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn","ppo","ddpg","a2c"])
    parser.add_argument("--trainer_cfg", type=str, default="configs/base.yaml")
    parser.add_argument("--algo_cfg", type=str, default="configs/dqn.yaml")
    args = parser.parse_args()
    args.device = device

    print(f"Running {args.algo}")
    print(f"Loading Trainer configs from {args.trainer_cfg}")
    trainer_config = load_yaml(args.trainer_cfg)
    print(f"Loading Algorithm configs from {args.algo_cfg}")
    algo_cfg = load_yaml(args.algo_cfg)

    agent = Trainer(cfg=trainer_config, algo_cfg= algo_cfg, args= args)
    agent.train()

        