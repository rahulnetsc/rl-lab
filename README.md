# RL-LAB

A minimal extensible learning framework built on OpenAI gymnasium and PyTorch

## Features
- Plug-and-play with any Gymnasium environment
- Modular design for any RL algorithm
- Ready-to-use baselines: `REINFORCE, DQN`
- Clean training loop with logging and evaluation

## Roadmap
- [x] Base framework with Gymnasium
- [ ] A2C
- [ ] DDPC
- [x] DQN
- [ ] PPO
- [ ] SAC (continuous control)
- [ ] Tensorboard integration
- [ ] Weights and biases integration
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
│   │
│   ├── configs
│   │   ├── base.yaml
│   │   ├── dqn.yaml
│   │
│   ├── logs/
│   │
│   ├── memory/
│   │   ├── replay.py
│   │
│   ├── nets/
│   │   ├── __init__.py
│   │   └── mlp.py
│   └── algos/
│       ├── __init__.py
│       ├── base.py
│       ├── dqn.py     
│       ├── reinforce.py     
│       └── ppo.py           
│    
├── main.py                  (CLI entry point)
├── README.md
├── requirements.txt
└── LICENSE

```

DQN
```
 python -m rl_Lab.trainer 
```