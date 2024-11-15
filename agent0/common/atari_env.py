import gymnasium as gym
from gymnasium import spaces
import ale_py
import ale_py.roms
import numpy as np
import random
from collections import deque
from gymnasium.utils import seeding
import cv2
from lz4.block import compress

class LazyFrames:
    __slots__ = ("frame_shape", "dtype", "shape", "lz4_compress", "_frames")

    def __init__(self, frames: list, lz4_compress: bool = False):
        self.frame_shape = tuple(frames[0].shape)
        self.shape = (len(frames),) + self.frame_shape
        self.dtype = frames[0].dtype
        frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.stack(
            [self._check_decompress(f) for f in self._frames[int_or_slice]], axis=0
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            from lz4.block import decompress

            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class FrameStack(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        num_stack: int,
        lz4_compress: bool = False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, num_stack=num_stack, lz4_compress=lz4_compress
        )
        gym.ObservationWrapper.__init__(self, env)

        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        return LazyFrames(list(self.frames), self.lz4_compress)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(None), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.num_stack)]
        return self.observation(None), info


class AtariEnv(gym.Env):
    def __init__(self,
                 game: str,
                 max_frames=108_000, 
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
            shape=(84, 84),
            dtype=np.uint8)
        
        self.frame_skip = frame_skip
        self.firereset = ale_py.Action.FIRE in self.action_set

        self.screen_buffer = np.zeros((2, *self.ale.getScreenDims()), dtype=np.uint8)
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
            for _ in range(4):
                self.ale.act(ale_py.Action.FIRE, 1.0)

        if self.ale.game_over():
            self.reset()

        return self._get_obs(), self._get_info()
        
    def step(self, action):
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.ale.act(self.action_set[action], 1.0)
            idx = self.ale.getEpisodeFrameNumber()
            self.ale.getScreenGrayscale(self.screen_buffer[idx % 2])
            if self.ale.game_over():
                break
        self.score += reward

        terminal = self.ale.game_over()
        truncated = self.ale.game_truncated()
        return self._get_obs(), reward, terminal, truncated, self._get_info()
        

    def _get_info(self):
        lifeloss = self.ale.lives() < self.lives
        self.lives = self.ale.lives()
        return {'score': self.score, 
                'lifeloss': lifeloss}

    def _get_obs(self):
        return cv2.resize(np.max(self.screen_buffer, axis=0), (84, 84))



def make_atari(env_id, num_envs):
    def trunk():
        env = AtariEnv(env_id.lower())
        env = FrameStack(env, 4, lz4_compress=True)
        return env
    envs = gym.vector.AsyncVectorEnv([lambda: trunk() for _ in range(num_envs)])
    return envs

