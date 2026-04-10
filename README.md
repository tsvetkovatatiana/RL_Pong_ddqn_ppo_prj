# Atari Pong RL (DDQN + PPO)

University project comparing two reinforcement learning approaches on `ALE/Pong-v5`:

- PPO (`stable-baselines3`)
- DDQN (custom PyTorch implementation)

## Demo

![Demo GIF](/screenshot/pong_ppo_eval-step-0-to-step-10000.gif)

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1) start training (in background)
source venv/bin/activate
nohup python -u ppo_main.py > ppo.out 2>&1 &
nohup python -u dqn_main.py > dqn.out 2>&1 &
```

```bash
# 2) check processes/logs
ps aux | grep _main.py
tensorboard --logdir ppo_logs --bind_all --port 6009
tail -f ppo.out
tail -f dqn.out
```

```bash
# 3) stop training
pkill -f ppo_main.py
pkill -f dqn_main.py
```

```bash
# optional: run PPO evaluation and save video
python -u ppo_test.py
```

## Pong Environment (Short Summary)

- Environment: `ALE/Pong-v5` (Atari Pong).
- Reward logic: `+1` when the agent scores, `-1` when the opponent scores, `0` otherwise.
- Actions: discrete joystick actions (no-op, fire, move up/down, and fire+move variants).

## Monitoring

I use TensorBoard to observe model behavior during training (reward trends, losses, and overall learning progress):

```bash
tensorboard --logdir ppo_logs --bind_all --port 6009
```

## Outputs

- PPO logs/checkpoints: `ppo_logs/`, `ppo_models/`
- DDQN checkpoints/metrics: `models/`, `metrics/`

## Experiment

Feel free to experiment with hyperparameters (timesteps, learning rate, batch size, epsilon decay, and eval frequency) and compare PPO vs DDQN behavior.

## License

For educational use.
