import gymnasium as gym 
import numpy as np
import argparse

from torch.utils.tensorboard import SummaryWriter
import torch 
from tqdm import tqdm

from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

import os 
from pathlib import Path
import time 

from .algos import DQN, DoubleDQN, DuelingDQN, DuelingDQNPrb, A2C
from .utils import dev_sel, load_yaml

class Trainer:
    def __init__(self, cfg, algo_cfg, args, env = None, eval_env = None):
        super().__init__()
        self.env_id = cfg['trainer']['env_id']
        self.env: gym.Env = env or gym.make(self.env_id)
        if args.only_eval:
            self.eval_env: gym.Env = eval_env or gym.make(self.env_id, render_mode="human")
        else:
            self.eval_env: gym.Env = eval_env or gym.make(self.env_id)

        self.obs_shape = np.array(self.env.observation_space.shape)
        self.obs_dim = int(np.prod(np.array(self.env.observation_space.shape)))
        self.action_dim = self.env.action_space.n if isinstance(self.env.action_space,gym.spaces.Discrete) else None 
        self.device = args.device
        print(f"Initializing Env with obs_dim={self.obs_dim}, obs_shape = {self.obs_shape}, action_dim= {self.action_dim}, cfg= {algo_cfg}, device= {self.device}")
        time.sleep(1)
        # need to implement a better logic for algo selection
        if args.algo == 'dqn':
            self.algo = DQN(obs_dim=self.obs_dim, obs_shape = self.obs_shape, action_dim= self.action_dim, 
                            cfg= algo_cfg, device= self.device)
        elif args.algo == 'doubledqn':
            self.algo = DoubleDQN(obs_dim=self.obs_dim, obs_shape = self.obs_shape, action_dim= self.action_dim, 
                            cfg= algo_cfg, device= self.device)
        elif args.algo == 'duelingdqn':
            self.algo = DuelingDQN(obs_dim=self.obs_dim, obs_shape = self.obs_shape, state_dim= 64, action_dim= self.action_dim, 
                            cfg= algo_cfg, device= self.device)
        elif args.algo == 'prbduelingdqn':
            self.algo = DuelingDQNPrb(obs_dim=self.obs_dim, obs_shape = self.obs_shape, state_dim= 64, action_dim= self.action_dim, cfg= algo_cfg, device= self.device)
        elif args.algo == 'a2c':
            # To be implemented
            pass 
        elif args.algo == 'ppo':
            # To be implemented 
            pass
        else: 
            raise NotImplementedError(f"Algo: {args.algo} not implemented")

        # --- Checkpoints ---
        self.ckpt_dir = Path(cfg['trainer']['ckpt_dir']) / self.env_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.ckpt_dir / f"{args.algo}_{self.env_id}.pt"

        # --- Logs ---
        self.log_dir = Path(cfg['trainer']['log_dir']) / f"{args.algo}_{self.env_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.global_step = 0
        self.episode_idx = 0
        self.seed = cfg['trainer']['seed']
        self.rng = np.random.default_rng(seed= self.seed)
        self.cfg = cfg

        self.best_eval_return = float('-inf')

    def train(self, ):
        
        obs, _ = self.env.reset(seed= self.seed)
        ep_ret, ep_len = 0.0, 0
        last_eval = 0
        last_log = 0
        if self.ckpt_path.exists() and self.cfg['trainer'].get('resume', False):
            state = torch.load(self.ckpt_path, map_location=self.device)
            self.algo.load_state_dict(state)
            print(f"Resumed from {self.ckpt_path}")
            eval_metrics: dict = self.eval_policy(idx=self.global_step) or {}
            self.best_eval_return = eval_metrics['return_mean']
        else:
            print(f"Starting training")
        
        self.algo.on_train_start(self.cfg)

        max_steps = int(self.cfg['trainer']['max_steps'])

        # bar = Progress(
        #     TextColumn("[bold blue]Training"),
        #     BarColumn(),
        #     TaskProgressColumn(),
        #     TextColumn("• step {task.completed}/{task.total}"),
        #     TextColumn("• loss {task.fields[loss]:.3f}"),
        #     TimeRemainingColumn(),
        #     transient=False,  # keep bar after finish; set True to clear
        # )

        # with bar:
        #     task = bar.add_task("train", total=max_steps, loss=0.0, start=True)
        # tqdm bar knows total steps; picks up from current global_step if resuming
        with tqdm(total=max_steps, initial=self.global_step, desc="Training", dynamic_ncols=True) as pbar:
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

                if metrics and (last_log==0 or self.global_step - last_log >= self.cfg['trainer']['log_every']):
                    # print(f"[Episode: {self.episode_idx}], ep_len: {ep_len}, global_step: {self.global_step}")
                    kv = ", ".join(f"train/{k}: {v:.3f}" for k, v in metrics.items())
                    tqdm.write(f"[Episode {self.episode_idx}] step={self.global_step} ep_len={ep_len} | {kv}")
                    # bar.log(f"[Episode {self.episode_idx}] step={self.global_step} ep_len={ep_len} | {kv}")
                
                    # print(kv)
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            self.writer.add_scalar(f'train/{k}', v, self.global_step)
                    last_log = self.global_step

                if last_eval==0 or self.global_step - last_eval >= self.cfg['trainer']['eval_every']:
                    eval_metrics: dict = self.eval_policy(idx=self.global_step) or {}
                    # kv = ", ".join(f"eval/{k}: {v:.3f}" for k, v in eval_metrics.items())
                    for k, v in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{k}', v, self.global_step)

                    if self.best_eval_return < eval_metrics['return_mean']:
                        # print(f"Eval at {self.global_step}")
                        # print(kv)
                    
                        self.best_eval_return = eval_metrics['return_mean']
                        state = self.algo.state_dict()
                        tqdm.write(f"Saving params in {self.ckpt_path} at self.global_step = {self.global_step}\n")
                        torch.save(state, self.ckpt_path)
                        tqdm.write(f"✅ New best at step {self.global_step}: return_mean={self.best_eval_return:.3f}")
                        # bar.log(f"[green]✅ New best at step {self.global_step}: return_mean={self.best_eval_return:.3f}")
                        last_eval = self.global_step

                obs = next_obs
                ep_ret += float(reward)
                ep_len += 1
                self.global_step += 1
                pbar.update(1)
                # advance + update live fields
                # bar.advance(task, 1)
                if metrics:
                    pbar.set_postfix(loss=float(metrics.get('loss', 0.0)), ep_len=ep_len, refresh=False)
                    # bar.update(task, loss=float(metrics.get('loss', 0.0)))

                if done:
                    # print(f"DONE: [Train/Episode: {self.episode_idx}], episode_len = {ep_len}, return = {ep_ret:.3f}")
                    self.writer.add_scalar('train/episode_return', ep_ret, self.global_step)
                    self.writer.add_scalar('train/episode_length', ep_len, self.global_step)
                    self.algo.on_episode_end()
                    ep_ret, ep_len = 0.0, 0
                    self.episode_idx += 1
                    obs, _ = self.env.reset()
        
        self.env.close(); self.eval_env.close()
        self.writer.flush(); self.writer.close()
        # state = self.algo.state_dict()
        # print(f"Saving params in {self.ckpt_dir}")
        # torch.save(state, self.ckpt_path)


    def eval_policy(self, idx=0, only_eval = False)-> dict| None:
        returns, lengths = [], []
        if only_eval:
            print(f"\nONLY EVAL: \nLoading state from {self.ckpt_path}")
            state = torch.load(self.ckpt_path)
            self.algo.load_state_dict(state=state)

        for i in range(self.cfg['eval']['eval_episodes']):
            obs, _ = self.eval_env.reset(seed=self.seed+1000+idx+i)
            done = False

            ep_ret, ep_len = 0.0, 0
            while not done:
                action = self.algo.select_action(obs_np=obs, eval_mode= True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action=action)
                done = bool(terminated or truncated)
                if self.cfg['eval']['render_eval'] and only_eval:
                    self.eval_env.render()
                obs = next_obs
                ep_ret += float(reward)
                ep_len += 1
            if only_eval:    
                print(f"[Eval] episode_len = {ep_len}, episode_return = {ep_ret:.3f}")
                
            returns.append(ep_ret)
            lengths.append(ep_len)

        # if self.cfg['eval']['render_eval']:
        #     self.eval_env.close()
        #     self.eval_env = gym.make(self.self.env_id, render_mode="human")
        # print(f"[Eval] length_mean = {np.mean(lengths)}, return_mean = {np.mean(returns):.3f}")
        return {
            'length_mean': np.mean(lengths),
            'return_mean': np.mean(returns)
        }

if __name__ == '__main__':

    device = dev_sel()

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn","doubledqn","duelingdqn","prbduelingdqn"])
    parser.add_argument("--trainer_cfg", type=str, default="rl_Lab/configs/base_lunar.yaml")
    parser.add_argument("--only_eval", action="store_true")
    args = parser.parse_args()
    args.device = device

    print(f"Running {args.algo} with args: \n{args} ")
    print(f"Loading Trainer configs from {args.trainer_cfg}")
    trainer_config = load_yaml(args.trainer_cfg)

    if args.algo == 'dqn':
        args.algo_cfg = "rl_Lab/configs/dqn.yaml"
    elif args.algo == 'doubledqn':
        args.algo_cfg = "rl_Lab/configs/double_dqn.yaml"
    elif args.algo == 'duelingdqn':
        args.algo_cfg = "rl_Lab/configs/duelingdqn.yaml"
    elif args.algo == 'prbduelingdqn':
        args.algo_cfg = "rl_Lab/configs/prb_duelingdqn.yaml"
    elif args.algo == 'a2c':
        args.algo_cfg = "rl_Lab/configs/a2c.yaml"
    elif args.algo == 'ppo':
        args.algo_cfg = "rl_Lab/configs/ppo.yaml"

    print(f"Loading Algorithm configs from {args.algo_cfg}")
    algo_cfg = load_yaml(args.algo_cfg)
    time.sleep(1)
    
    agent = Trainer(cfg=trainer_config, algo_cfg= algo_cfg, args= args)
    if args.only_eval:
        agent.eval_policy(only_eval=args.only_eval)
    else:
        start_time = time.time()
        agent.train()
        print(f"Full training time = {(time.time()-start_time)/60:.2f} mins")

        