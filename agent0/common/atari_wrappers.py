from collections import deque

import gymnasium as gym
import ale_py

import numpy as np
from gymnasium.core import Env
from gymnasium.wrappers import (AtariPreprocessing, FrameStackObservation,
                                RecordEpisodeStatistics)
gym.register_envs(ale_py)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        for a in range(3):
            obs, _, terminated, _, info = self.env.step(a)
            if terminated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over.
    Done by DeepMind for the DQN and co. since it helps value estimation.
    """

    def step(self, action):
        old_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, trunc, info = super().step(action)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        new_lives = self.env.unwrapped.ale.lives()

        # for Qbert sometimes we stay in lives == 0 condition for a few frames
        # so it's important to keep lives > 0, so that we only reset once
        # the environment advertises done.
        life_loss = old_lives > new_lives > 0
        info["life_loss"] = life_loss
        if life_loss and self.env.unwrapped.get_action_meanings()[1] == "FIRE":
            for a in range(3):
                obs, _, _, _, step_info = self.env.step(a)
            info.update(step_info)
        return obs, reward, done, trunc, info


def make_atari(env_id, num_envs, episode_life=True):
    wrappers = [
        lambda x: AtariPreprocessing(x, terminal_on_life_loss=False),
        lambda x: FrameStackObservation(x, 4),
        lambda x: EpisodicLifeEnv(x) if episode_life else x,
        FireResetEnv,
        RecordEpisodeStatistics,
        ClipRewardEnv,
    ]
    envs = gym.make_vec(f"{env_id}NoFrameskip-v4", num_envs, wrappers=wrappers)
    return envs
