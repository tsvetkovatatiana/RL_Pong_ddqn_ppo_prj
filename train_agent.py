import time
import os
import json
import torch.optim

from dqn import *
from utils import *

os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

def train(environment, name, q_net,
          target_net, optimizer, timesteps,
          replay, metric, train_freq,
          batch, g, C, save_step=85_000):

    loss_f = nn.SmoothL1Loss()
    start_time = time.time()
    episode_count = 0
    best_avg_reward = -float("inf")
    printed_shapes = False

    obs, _ = environment.reset()
    print(obs.shape)

    for step in range(1, timesteps + 1):
        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = np.squeeze(obs, axis=-1)   # [4, 84, 84, 1] -> [4, 84, 84]

        batched_obs = np.expand_dims(obs, axis=0)

        action = q_net.epsilon_greedy(
            torch.as_tensor(batched_obs, dtype=torch.float32, device=device)
        ).cpu().item()

        next_state, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated

        replay.store((obs, action, reward, next_state, done))
        metric.add_step_reward(reward)
        obs = next_state

        if step % train_freq == 0 and len(replay) >= batch:
            observations, actions, rewards, next_states, dones = replay.sample(batch)

            with torch.no_grad():
                q_values_next = target_net(next_states)
                bootstrapped_values = torch.amax(q_values_next, dim=1, keepdim=True)

                rewards = rewards.unsqueeze(1)              # [batch, 1]
                dones = dones.unsqueeze(1)                  # [batch, 1]
                y_true = rewards + g * bootstrapped_values * (1.0 - dones.float())

            actions = actions.unsqueeze(1)                  # [batch, 1]
            y_pred = q_net(observations)
            q_selected = y_pred.gather(1, actions)         # [batch, 1]

            if not printed_shapes:
                print("observations:", observations.shape)
                print("actions:", actions.shape)
                print("rewards:", rewards.shape)
                print("next_states:", next_states.shape)
                print("dones:", dones.shape)
                print("q_selected:", q_selected.shape)
                print("y_true:", y_true.shape)
                printed_shapes = True

            loss = loss_f(q_selected, y_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()

        q_net.epsilon_decay(step)
        target_net.epsilon_decay(step)

        if done:
            elapsed_time = time.time() - start_time
            steps_per_sec = step / elapsed_time
            metric.end_episode()
            episode_count += 1

            obs, _ = environment.reset()

            if metric.avg_reward > best_avg_reward and step > save_step:
                best_avg_reward = metric.avg_reward
                model_path = f"models/{name}_dqn_best_{step}.pth"
                metrics_path = f"metrics/{name}_metrics_{step}.json"

                torch.save({
                    "step": step,
                    "model_state_dict": q_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "avg_reward": metric.avg_reward,
                }, model_path)

                with open(metrics_path, "w") as f:
                    json.dump({
                        "step": step,
                        "avg_reward": metric.avg_reward,
                        "episode_count": episode_count,
                        "epsilon": q_net.epsilon,
                    }, f, indent=4)

            print(
                f"\rStep: {step:,}/{timesteps:,} | "
                f"Episodes: {episode_count} | "
                f"Avg Reward: {metric.avg_reward:.4f} | "
                f"Epsilon: {q_net.epsilon:.4f} | "
                f"Steps/sec: {steps_per_sec:.2f}",
                end="\r"
            )

        if step % C == 0:
            target_net.load_state_dict(q_net.state_dict())