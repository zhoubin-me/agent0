import gymnasium as gym
from gymnasium import spaces
import ale_py
import ale_py.roms
import numpy as np
import random
from collections import deque
from gymnasium.utils import seeding
import cv2

class AtariEnv(gym.Env):
    def __init__(self,
                 game: str,
                 max_frames=108_000,
                 frame_stack=4,
                 frame_skip=4):
        self.ale = ale_py.ALEInterface()
        self.ale.setFloat("repeat_action_probability", 0.0)  # Set deterministic actions
        self.ale.setInt("max_num_frames_per_episode", max_frames)
        self.seed_game()
        self.ale.loadROM(ale_py.roms.get_rom_path(game))
        self.action_set = self.ale.getMinimalActionSet()

        self.action_space = spaces.Discrete(len(self.action_set))
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(4, 84, 84),
            dtype=np.uint8)
        
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.firereset = ale_py.Action.FIRE in self.action_set
        self.screen_buffer = np.zeros((2, *self.ale.getScreenDims()), dtype=np.uint8)
        self.frame_buffer = deque([self._get_obs()]*frame_stack, maxlen=frame_stack)
        self.lives = self.ale.lives()
        self.score = 0

    def seed_game(self, seed=None):
        """Seeds the internal and ALE RNG."""
        ss = np.random.SeedSequence(seed)
        np_seed, ale_seed = ss.generate_state(n_words=2)
        self._np_random, seed = seeding.np_random(int(np_seed))
        self.ale.setInt("random_seed", np.int32(ale_seed))
        return np_seed, ale_seed

    def reset(self, seed=None, **kwargs):
        if self.ale.game_over() or self.ale.game_truncated():
            self.seed_game(seed)
            self.ale.reset_game()
            for _ in range(random.randint(2, 31)):
                self.ale.act(ale_py.Action.NOOP, 1.0)
                idx = self.ale.getEpisodeFrameNumber()
                self.ale.getScreenGrayscale(self.screen_buffer[idx % 2])
            self.score = 0

        
        if self.firereset:
            for a in range(3):
                self.ale.act(self.action_set[a], 1.0)

        if self.ale.game_over():
            self.reset()
        self.frame_buffer = deque([self._get_obs()]*self.frame_stack, maxlen=self.frame_stack)
        return self._get_frames(), self._get_info()
        
    def step(self, action):
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.ale.act(self.action_set[action], 1.0)
            idx = self.ale.getEpisodeFrameNumber()
            self.ale.getScreenGrayscale(self.screen_buffer[idx % 2])
            if self.ale.game_over():
                break
        self.score += reward
        reward_reshaped = np.sign(reward) * np.log(1 + np.abs(reward))
        self.frame_buffer.append(self._get_obs())

        terminal = self.ale.game_over()
        truncated = self.ale.game_truncated()
        return self._get_frames(), reward_reshaped, terminal, truncated, self._get_info()
        

    def _get_info(self):
        lifeloss = self.ale.lives() < self.lives
        self.lives = self.ale.lives()
        return {'score': self.score, 
                'lifeloss': lifeloss}

    def _get_obs(self):
        return cv2.resize(np.max(self.screen_buffer, axis=0), (84, 84))

    def _get_frames(self):
        return np.array(self.frame_buffer)

def make_atari(env_id, num_envs):
    def trunk():
        env = AtariEnv(env_id.lower())
        return env
    envs = gym.vector.AsyncVectorEnv([lambda: trunk() for _ in range(num_envs)])
    return envs

if __name__ == '__main__':
    envs = make_atari('breakout', 3)
    obs = envs.reset()
    breakpoint()
