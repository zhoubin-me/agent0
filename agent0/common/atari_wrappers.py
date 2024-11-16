from collections import deque

import gymnasium as gym
import ale_py

import numpy as np
from gymnasium.wrappers import (
    AtariPreprocessing, 
    FrameStackObservation,
    TransformReward,
    RecordEpisodeStatistics)

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
        info["lifeloss"] = life_loss
        if life_loss and self.env.unwrapped.get_action_meanings()[1] == "FIRE":
            for a in range(3):
                obs, _, _, _, step_info = self.env.step(a)
            info.update(step_info)
        return obs, reward, done, trunc, info


def make_atari(env_id: str, num_envs: int, episode_life=True):
    def trunk():
        x = gym.make(f'{env_id.capitalize()}NoFrameskip-v4')
        x = AtariPreprocessing(x, terminal_on_life_loss=False)
        x = FrameStackObservation(x, 4)
        x = EpisodicLifeEnv(x)
        x = FireResetEnv(x)
        x = RecordEpisodeStatistics(x)
        # x = TransformReward(x, lambda r: np.sign(r) * np.log(1 + np.abs(r)))
        x = TransformReward(x, lambda r: np.sign(r))
        return x
    envs = gym.vector.AsyncVectorEnv([lambda: trunk() for _ in range(num_envs)])
    return envs