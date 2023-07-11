import copy
from collections import deque, defaultdict

import cv2
import gym
import numpy as np
from gym import spaces
from lz4.block import compress

cv2.ocl.setUseOpenCL(False)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action: int):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30, noop_num=None):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = noop_num
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, force_reset=False, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done or force_reset:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


# noinspection PyArgumentList
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        info = {}
        done = None
        for i in range(self._skip):
            obs_, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs_
            if i == self._skip - 1:
                self._obs_buffer[1] = obs_
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class NormReward(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.log1p(reward) if reward >= 0 else -np.log1p(-reward)


class GaussianReward(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        return np.random.normal(float(reward), 0.25)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.frames_ = self.frames.copy()
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]),
                                                dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=0)

    def restore(self, state):
        self.ale.restoreSystemState(state)
        self.frames = self.frames_.copy()
        return np.concatenate(self.frames_, axis=0)

    def clone(self):
        self.frames_ = self.frames.copy()
        return self.ale.cloneSystemState()


class NStepEnv(gym.Wrapper):
    def __init__(self, env, n, discount):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.discount = discount
        self.tracker = deque(maxlen=n)
        self.last_obs = None

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        self.last_obs = ob
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.tracker.append((self.last_obs, action, reward, done, info))
        self.last_obs = ob

        r_discounted = 0
        done_discounted = False
        bad_transit = False
        for _, _, r, d, inf in reversed(self.tracker):
            r_discounted = r_discounted * self.discount * (1 - d) + r
            if d:
                done_discounted = True
            if 'counter' in inf:
                bad_transit = True

        info.update(
            prev_obs=self.tracker[0][0],
            prev_action=self.tracker[0][1],
            prev_reward=r_discounted,
            prev_done=done_discounted,
            prev_bad_transit=bad_transit,
        )

        return ob, reward, done, info


class StateCountEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.ep_counter = defaultdict(int)
        self.ep_len = 0
        self.obs_ = None
        self.max_count = 10

    def reset(self, **kwargs):
        if len(self.ep_counter) > 0 and max(self.ep_counter.values()) > self.max_count:
            kwargs.update(force_reset=True)
        ob = self.env.reset(**kwargs)
        self.ep_counter.clear()
        self.obs_ = ob
        self.ep_len = 0
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        key = hash((self.obs_.tobytes(), action))
        self.ep_counter[key] += 1
        self.ep_len += 1
        if self.ep_len > 100 and self.ep_counter[key] > self.max_count:
            info.update(counter=(self.max_count, self.ep_len))
            done = True
        self.obs_ = ob
        return ob, reward, done, info


class RewardStatEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.steps = 0
        self.real_reward = 0

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.steps += 1
        self.real_reward += reward
        if done:
            if self.was_real_done or 'counter' in info:
                info.update(real_reward=self.real_reward, steps=self.steps, real_done=self.was_real_done)
                self.steps = 0
                self.real_reward = 0
        return ob, reward, done, info


class EpRecordEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.cur_ep = []
        self.best_return = float('-inf')

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        self.ob = ob
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.cur_ep.append((compress(self.ob), action))
        if 'real_reward' in info and info['real_reward'] > self.best_return:
            self.best_return = info['real_reward']
            info.update(best_ep=copy.deepcopy(self.cur_ep))
            self.cur_ep = []
        return ob, reward, done, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


def _process_frame42(frame):
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = spaces.Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)




def make_dm_atari(game, episode_life=True, clip_rewards=True, frame_stack=4, transpose_image=True, norm_reward=False,
                   n_step=1, discount=0.99, scale=False, noop_num=None, seed=None, gaussian_reward=False,
                   state_count=False, record_best_ep=False):
    assert not (clip_rewards and norm_reward)
    env = gym.make(f'{game}NoFrameskip-v4')
    env = NoopResetEnv(env, noop_max=30, noop_num=noop_num)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if transpose_image:
        env = TransposeImage(env)
    if scale:
        env = ScaledFloatFrame(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if state_count:
        env = StateCountEnv(env)
    env = RewardStatEnv(env)
    if record_best_ep:
        env = EpRecordEnv(env)

    if clip_rewards:
        env = ClipRewardEnv(env)
    if norm_reward:
        env = NormReward(env)
    if gaussian_reward:
        env = GaussianReward(env)
    if n_step > 1:
        env = NStepEnv(env, n_step, discount)
    if seed is not None:
        env.seed(seed)
    return env

def make_atari(env_id, num_envs):
    from agent0.common.vec_env import ShmemVecEnv
    from gym.spaces import MultiDiscrete, Box
    envs = ShmemVecEnv([lambda : make_dm_atari(env_id) for _ in range(num_envs)], context='spawn')
    envs.action_space = MultiDiscrete([envs.action_space.n for _ in range(num_envs)])
    envs.observation_space = Box(low=0, high=255, shape=(1, *envs.observation_space.shape))
    return envs


if __name__ == '__main__':
    import time
    import tqdm
    envs = make_atari('Breakout', 16)
    obs = envs.reset()
    print(obs.shape)
    tic = time.time()
    for _ in tqdm.tqdm(range(100)):
        _, _, done, info = envs.step(envs.action_space.sample())
        if 'real_reward' in info:
            print(info)
    toc = time.time()
    print(toc - tic)
