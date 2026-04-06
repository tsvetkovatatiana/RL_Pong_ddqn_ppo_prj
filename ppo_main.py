import os
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


ENV_ID = "ALE/Pong-v5"
RUN_NAME = "pong_ppo"

TOTAL_TIMESTEPS = 25_000_000
N_ENVS = 8
FRAME_STACK = 4
SEED = 42

LOG_DIR = "ppo_logs"
MODEL_DIR = "ppo_models"
BEST_DIR = os.path.join(MODEL_DIR, "best_model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def make_train_env():
    env = make_atari_env(
        ENV_ID,
        n_envs=N_ENVS,
        seed=SEED,
        wrapper_kwargs=dict(
            terminal_on_life_loss=True  # important for training
        )
    )
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def make_eval_env():
    env = make_atari_env(
        ENV_ID,
        n_envs=1,
        seed=SEED + 1,
        wrapper_kwargs=dict(
            terminal_on_life_loss=False
        )
    )
    env = VecFrameStack(env, n_stack=FRAME_STACK)
    return env


def main():
    train_env = make_train_env()
    eval_env = make_eval_env()

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // N_ENVS, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix=RUN_NAME,
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_DIR,
        log_path=LOG_DIR,
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=SEED,
        device="auto",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=RUN_NAME,
    )

    model.save(os.path.join(MODEL_DIR, f"{RUN_NAME}_final"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()