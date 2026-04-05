import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.atari_wrappers import AtariWrapper


def make_env(game, render_mode=None, full_action_space=False):
    env = gym.make(
        game,
        render_mode=render_mode,
        full_action_space=full_action_space,
    )

    # Atari preprocessing:
    # - frame skip
    # - terminal on life loss
    # - resize to 84x84
    # - grayscale
    # - reward clipping
    env = AtariWrapper(env)

    # Stack 4 processed frames -> shape [4, 84, 84]
    env = FrameStackObservation(env, stack_size=4)

    return env