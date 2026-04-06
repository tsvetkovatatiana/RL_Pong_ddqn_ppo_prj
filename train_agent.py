import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("ppo_logs", exist_ok=True)


def get_next_run_id(base_dir, name):
    existing = [
        d for d in os.listdir(base_dir)
        if d.startswith(name + "_")
    ]

    ids = []
    for d in existing:
        try:
            ids.append(int(d.split("_")[-1]))
        except ValueError:
            continue

    next_id = max(ids, default=0) + 1
    return f"{name}_{next_id}"


def evaluate_policy(environment, q_net, episodes=5):
    rewards = []

    for _ in range(episodes):
        obs, _ = environment.reset()

        if obs.ndim == 4 and obs.shape[-1] == 1:
            obs = np.squeeze(obs, axis=-1)

        done = False
        episode_reward = 0.0

        while not done:
            batched_obs = np.expand_dims(obs, axis=0)

            with torch.no_grad():
                q_values = q_net(
                    torch.as_tensor(
                        batched_obs,
                        dtype=torch.float32,
                        device=q_net.device,
                    )
                )
                action = torch.argmax(q_values, dim=1).item()

            next_state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            if next_state.ndim == 4 and next_state.shape[-1] == 1:
                next_state = np.squeeze(next_state, axis=-1)

            obs = next_state
            episode_reward += reward

        rewards.append(episode_reward)

    return float(np.mean(rewards))


def train(
    environment,
    name,
    q_net,
    target_net,
    optimizer,
    timesteps,
    replay,
    metric,
    train_freq,
    batch,
    g,
    C,
    save_step=85_000,
    eval_freq=50_000,
    eval_episodes=5,
):
    loss_f = nn.SmoothL1Loss()
    start_time = time.time()
    episode_count = 0
    best_avg_reward = -float("inf")
    printed_shapes = False

    # SB3-style run naming
    run_name = get_next_run_id("ppo_logs", name)
    log_dir = os.path.join("ppo_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir)

    print("TensorBoard run dir:", log_dir)

    obs, _ = environment.reset()

    if obs.ndim == 4 and obs.shape[-1] == 1:
        obs = np.squeeze(obs, axis=-1)

    print("Initial obs shape:", obs.shape)

    try:
        for step in range(1, timesteps + 1):
            batched_obs = np.expand_dims(obs, axis=0)

            action = q_net.epsilon_greedy(
                torch.as_tensor(batched_obs, dtype=torch.float32, device=q_net.device)
            ).item()

            next_state, reward, terminated, truncated, _ = environment.step(action)
            done = terminated or truncated

            if next_state.ndim == 4 and next_state.shape[-1] == 1:
                next_state = np.squeeze(next_state, axis=-1)

            replay.store((obs, action, reward, next_state, done))
            metric.add_step_reward(reward)
            obs = next_state

            writer.add_scalar("train/step_reward", reward, step)

            if step % train_freq == 0 and len(replay) >= batch:
                observations, actions, rewards, next_states, dones = replay.sample(batch)

                observations = observations.to(q_net.device)
                actions = actions.to(q_net.device)
                rewards = rewards.to(q_net.device)
                next_states = next_states.to(q_net.device)
                dones = dones.to(q_net.device)

                with torch.no_grad():
                    q_values_next = target_net(next_states)
                    bootstrapped_values = torch.amax(q_values_next, dim=1, keepdim=True)

                    rewards = rewards.unsqueeze(1)
                    dones = dones.unsqueeze(1)
                    y_true = rewards + g * bootstrapped_values * (1.0 - dones.float())

                actions = actions.unsqueeze(1)
                y_pred = q_net(observations)
                q_selected = y_pred.gather(1, actions)

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

                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar("train/q_mean", q_selected.mean().item(), step)
                writer.add_scalar("train/target_mean", y_true.mean().item(), step)

            q_net.epsilon_decay(step)
            target_net.epsilon_decay(step)

            writer.add_scalar("train/epsilon", float(q_net.epsilon), step)

            if done:
                elapsed_time = time.time() - start_time
                steps_per_sec = step / max(elapsed_time, 1e-8)

                metric.end_episode()
                episode_count += 1

                writer.add_scalar("rollout/ep_rew_mean", float(metric.avg_reward), step)
                writer.add_scalar("train/episode_count", episode_count, step)
                writer.add_scalar("time/steps_per_sec", steps_per_sec, step)

                obs, _ = environment.reset()
                if obs.ndim == 4 and obs.shape[-1] == 1:
                    obs = np.squeeze(obs, axis=-1)

                if metric.avg_reward > best_avg_reward and step > save_step:
                    best_avg_reward = metric.avg_reward

                    model_path = f"models/{run_name}_best_{step}.pth"
                    metrics_path = f"metrics/{run_name}_{step}.json"

                    torch.save(
                        {
                            "step": step,
                            "model_state_dict": q_net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "avg_reward": metric.avg_reward,
                            "epsilon": float(q_net.epsilon),
                        },
                        model_path,
                    )

                    with open(metrics_path, "w") as f:
                        json.dump(
                            {
                                "step": step,
                                "avg_reward": float(metric.avg_reward),
                                "episode_count": episode_count,
                                "epsilon": float(q_net.epsilon),
                            },
                            f,
                            indent=4,
                        )

                print(
                    f"Step: {step:,}/{timesteps:,} | "
                    f"Episodes: {episode_count} | "
                    f"Avg Reward: {metric.avg_reward:.4f} | "
                    f"Epsilon: {q_net.epsilon:.4f} | "
                    f"Steps/sec: {steps_per_sec:.2f}",
                    end="\r",
                )

            if step % C == 0:
                target_net.load_state_dict(q_net.state_dict())
                writer.add_scalar("train/target_sync", 1, step)

            # Evaluation logging (this fixes missing eval/mean_reward)
            if step % eval_freq == 0:
                eval_mean_reward = evaluate_policy(
                    environment, q_net, episodes=eval_episodes
                )
                writer.add_scalar("eval/mean_reward", eval_mean_reward, step)

                print(
                    f"\nEval @ step {step:,}: mean_reward={eval_mean_reward:.4f}"
                )

        print("\nTraining finished.")

    finally:
        writer.close()