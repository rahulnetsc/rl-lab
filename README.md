# RL-LAB

A minimal extensible learning framework built on OpenAI gymnasium and PyTorch

## Features
- Plug-and-play with any Gymnasium environment
- Modular design for any RL algorithm
- Ready-to-use baselines: `PPO, DDPG` etc
- Clean training loop with logging and evaluation

## Roadmap
- [x] Base framework with Gymnasium
- [ ] A2C
- [ ] DDPG
- [x] DQN
- [x] Double DQN
- [x] Dueling DQN
- [x] Prioritised Replay Buffer DQN
- [x] PPO
- [ ] SAC (continuous control)
- [x] Tensorboard integration
- [x] Weights and biases integration
- [ ] unit tests

## Package structure
```
rl-lab/
├── rl_Lab/
│   ├── __init__.py
│   ├── trainer.py
│   ├── envs.py
│   ├── utils.py
│   ├── checkpoints/
│   ├── adapter
│   │   ├── client.py
│   │   ├── env_adapter.py
│   │
│   └── algos/
│   │   ├── base.py
│   │   ├── dqn.py     
│   │   ├── doubledqn.py     
│   │   ├── duelingdqn.py     
│   │   ├── prb_dueling.py
│   │   ├── a2c.py
│   │   ├── ddpg.py
│   │   └── ppo.py           
│   │
│   ├── configs
│   │   ├── a2c.yaml
│   │   ├── base_acro.yaml
│   │   ├── base_cart.yaml
│   │   ├── base_lunar.yaml
│   │   ├── base_mtn.yaml
│   │   ├── base_taxi.yaml
│   │   ├── dqn.yaml
│   │   ├── double_dqn.yaml
│   │   ├── dueling_dqn.yaml
│   │   ├── prb_duelingdqn.yaml
│   │   ├── ppo.yaml
│   │
│   ├── logs/
│   │
│   ├── memory/
│   │   ├── replay.py
│   │   ├── prb_replay.py
│   │   ├── rollout.py
│   │
│   ├── nets/
│   │   ├── cnn.py
│   │   ├── transformers.py
│   │   └── mlp.py
│    
├── main.py                  (CLI entry point)
├── README.md
├── requirements_linux_rocm.txt
├── requirements_linux.txt
├── requirements_win.txt
└── LICENSE

```

DQN
```
  python -m rl_Lab.trainer --algo dqn
```

Double DQN
```
  python -m rl_Lab.trainer --algo doubledqn
```

Dueling DQN
```
  python -m rl_Lab.trainer --algo duelingdqn
```
    
Dueling DQN with Prioritized Replay Buffer
```
 python -m rl_Lab.trainer --algo prbduelingdqn 
```

A2C
```
 python -m rl_Lab.trainer --algo a2c
```

PPO
```
 python -m rl_Lab.trainer --algo ppo
```
