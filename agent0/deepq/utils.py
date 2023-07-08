from gymnasium.core import Env
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics
import gymnasium as gym
import numpy as np

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
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}



class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        old_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, trunc, info = super().step(action)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        new_lives = self.env.unwrapped.ale.lives()

        if old_lives > new_lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            info['life_loss'] = True
            obs, _, _, _, _ = self.env.step(0)
        else:
            info['life_loss'] = False
        return obs, reward, done, trunc, info



def make_atari(env_id, num_envs):
    wrappers = [
        FireResetEnv,
        lambda x: AtariPreprocessing(x, terminal_on_life_loss=False), 
        lambda x: FrameStack(x, 4, False),
        ClipRewardEnv,
        EpisodicLifeEnv,
    ]
    envs = gym.make_vec(f'{env_id}NoFrameskip-v4', num_envs, wrappers=wrappers)
    envs = RecordEpisodeStatistics(envs)
    return envs
    

def img_preproc(imgs):
    return imgs.float() / 255.0 * 2.0 - 1.0

if __name__ == '__main__':
    envs = make_atari('Breakout', 16)
    obs, info = envs.reset()