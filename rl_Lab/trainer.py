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
        self.eval_env: gym.Env = gym.make(cfg['trainer']['env_id'])
        
        self.obs_dim = int(np.prod(np.array(self.env.observation_space.shape)))
        self.action_dim = self.env.action_space.n if isinstance(self.env.action_space,gym.spaces.Discrete) else None 
        
        if args.algo == 'dqn':
            self.algo = DQN(obs_dim=self.obs_dim, num_actions= self.action_dim, cfg= algo_cfg, device= args.device)
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

            if self.global_step - last_log >= self.cfg['log_every']:
                print(f"[Episode: {self.episode_idx}], ep_len: {ep_len}, global_step: {self.global_step}")
                kv = ", ".join(f"{k}: {v:.3f}" for k, v in metrics.items())
                print(kv)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)
                last_log = self.global_step

            if self.global_step - last_eval >= self.cfg['eval_every']:
                eval_metrics: dict = self.eval_policy() or {}
                print(f"Eval at {self.global_step}")
                kv = ", ".join(f"{k}: {v:.3f}" for k, v in eval_metrics.items())
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

    def eval_policy(self,)-> dict| None:
        returns, lengths = [], []
        for _ in range(self.cfg['eval_episodes']):
            obs, _ = self.eval_env.reset(seed=self.seed+1000)
            done = False

            ep_ret, ep_len = 0.0, 0
            while not done:
                action = self.algo.select_action(obs_np=obs, eval_mode= True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action=action)
                done = bool(terminated or truncated)
                ep_ret += float(reward)
                ep_len += 1
                if self.cfg['render_eval']:
                    self.eval_env.render()

            returns.append(ep_ret)
            lengths.append(ep_len)

        return {
            'eval/return_mean': np.mean(returns),
            'eval/length_mean': np.mean(lengths)
        }

if __name__ == '__main__':

    device = dev_sel()

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn","ppo","ddpg","a2c"])
    parser.add_argument("--trainer_cfg", type=str, default="configs/base.yaml")
    parser.add_argument("--algo_cfg", type=str, default="configs/dqn.yaml")
    args = parser.parse_args()
    args.device = device

    print(f"Running ")
    trainer_config = load_yaml(args.trainer_cfg)
    algo_cfg = load_yaml(args.algo_cfg)

    agent = Trainer(cfg=trainer_config, algo_cfg= algo_cfg, args= args)
    agent.train()

        