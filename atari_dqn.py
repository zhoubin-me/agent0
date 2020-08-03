import gym
import os
import random
import time
import cv2
import copy
import numpy as np
import collections
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import argparse
from PIL import Image
from collections import deque, OrderedDict
from baselines.common.vec_env import ShmemVecEnv, VecEnvWrapper, DummyVecEnv
from prefetch_generator import BackgroundGenerator



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--game", type=str, default="Breakout")
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=9)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--exploration_steps", type=int, default=20000)
    parser.add_argument("--max_step", type=int, default=int(1e7))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_env", type=int, default=16)

    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args




def main():
    args = parse_arguments()
    game = args.game
    replay_size = args.replay_size
    max_step = args.max_step
    discount = args.discount
    lr = args.lr
    batch_size = args.batch_size
    exploration_steps = args.exploration_steps
    gpu_id = args.gpu_id
    num_env = args.num_env

    epoches = 100
    steps_per_epoch = max_step // epoches
    update_per_data = 8
    exploration_ratio=0.1



    # ckpt_path = "_".join(vars(args).values())

    try:
        os.makedirs(ckpt)
    except:
        pass

    # set_thread(5)
    random_seed(args.seed)
    device = torch.device(f'cuda:{gpu_id}')


    def make_env(game, episode_life=True, clip_rewards=True):
        env = make_atari(f'{game}NoFrameskip-v4')
        env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False, transpose_image=True)
        return env

    env_test = make_env(game, False, False)
    envs = ShmemVecEnv([lambda: make_env(game) for _ in range(num_env)], context='fork')

    state_shape = envs.observation_space.shape
    action_shape = envs.action_space.n

    epsilon_schedule = LinearSchedule(1.0, 0.01, int(epoches * exploration_ratio))
    model = ConvNet(4, action_shape).to(device)
    model_target = copy.deepcopy(model)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    R = np.zeros(num_env)
    Rs, Ls, Qs = [], [], []
    obs = envs.reset()
    replay = deque(maxlen=replay_size)

    for epoch in range(epoches):
        # Collect Samples
        R_data = []
        tic = time.time()
        epsilon = epsilon_schedule()
        steps = steps_per_epoch // num_env
        for step in tqdm(range(steps)):
            action_random = np.random.randint(0, action_shape, num_env)
            st = torch.from_numpy(np.array(obs)).float().to(device) / 255.0
            qs = model(st)
            qs_max, qs_argmax = qs.max(dim=-1)
            action_greedy = qs_argmax.tolist()
            Qs += qs_max.tolist()
            action = [act_grd if p > epsilon else act_rnd for p, act_rnd, act_grd in zip(np.random.rand(num_env), action_random, action_greedy)]

            obs_next, reward, done, info = envs.step(action)
            for entry in zip(obs, action, reward, obs_next, done):
                replay.append(entry)
            obs = obs_next
            R += np.array(reward)
            for idx, d in enumerate(done):
                if d:
                    Rs.append(R[idx])
                    R_data.append(R[idx])
                    R[idx] = 0

        toc = time.time()
        print(f"Epoch {epoch}, Data Collection Time: {(toc - tic) / 60} mins, Rs:[{np.mean(R_data):6.2f},{np.std(R_data):6.2f},{np.max(R_data):6.2f}], Speed: {steps_per_epoch / (toc - tic)}, Epsilon: {epsilon}")

        # Update model
        dataset = ImageDataset(replay)
        dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=True)
        prefetcher = DataPrefetcher(dataloader, device)
        data = prefetcher.next()
        steps = steps_per_epoch * update_per_data // batch_size - 1
        tic = time.time()

        for step in tqdm(range(steps)):
            states, actions, rewards, next_states, terminals = data
            states = states.float()/ 255.0
            next_states = next_states.float() / 255.0
            actions = actions.long()
            terminals = terminals.float()
            rewards = rewards.float()

            with torch.no_grad():
                q_next = model_target(next_states)
                q_next_online = model(next_states)
                q_next = q_next.gather(1, q_next_online.argmax(dim=-1).unsqueeze(-1)).squeeze(-1)
                q_target = rewards + discount * (1 - terminals) * q_next

            q = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            loss = F.smooth_l1_loss(q, q_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Ls.append(loss.item())
            try:
                data = prefetcher.next()
            except:
                prefetcher = DataPrefetcher(dataloader, device)
                data = prefetcher.next()

            if step % (steps // update_per_data) == 0:
                model_target.load_state_dict(model.state_dict())

        toc = time.time()
        print(f"Epoch {epoch}, Training Time: {(toc - tic) / 60} mins, Speed: {steps_per_epoch / (toc - tic)}, Loss: {np.mean(Ls[-steps:])}")

        # Testing model
        Rs_test = []
        R_test = 0
        tic = time.time()
        for e in tqdm(range(300)):
            ob = env_test.reset()
            while True:
                ob = torch.from_numpy(np.array(ob)).float().to(device).unsqueeze(0) / 255.0
                action = model(ob).argmax(dim=-1).item()
                ob_next, reward, done, info = env_test.step(action)
                ob = ob_next
                R_test += reward
                if done:
                    ob = env_test.reset()
                    Rs_test.append(R_test)
                    R_test = 0
                    break
        toc = time.time()
        print(f"Epoch {epoch}, Testing Time: {(toc - tic) / 60} mins, Rs:[{np.mean(Rs_test):6.2f},{np.std(Rs_test):6.2f},{np.max(Rs_test):6.2f}]")

        # toc = time.time()
        # Ts.append(toc - tic)

        # # Print Info
        # if step > 0 and step % 10000 == 0:
        #     mean_duration = np.mean(Ts[-10000:])
        #     mean_q = np.mean(Qs[-10000:])
        #     if len(Rs) == 0:
        #         max_R = float('nan')
        #     else:
        #         max_R = np.max(Rs[-10000:])

        #     print(f"[{step // 10000:4d}-{max_step // 10000:4d}] *10000  ||  Speed {1.0 / mean_duration:5.0f}  ||  RemHr {(max_step - step) * mean_duration / 3600.0:5.2f}  ||  Loss {np.mean(Ls[-100:]):10.7f}  ||  \
        #             Mean R {np.mean(Rs[-100:]):8.2f}  ||  Std R {np.std(Rs[-100:]):8.2f}  ||  Max R {max_R:5.0f}  ||  Mean Q {mean_q:5.2f}")

        # # Save model
        # if step % 200000 == 0:
        #     with open('{ckpt}/log.txt', 'w') as f:

        #         for x in Ls:
        #             f.write('Loss: ' + str(x) + '\n')

        #         for x in Rs:
        #             f.write('R: ' + str(x) + '\n')

        #         for x in Qs:
        #             f.write('Q: ' + str(x) + '\n')

        #     torch.save({'model': model.state_dict()}, f"{ckpt}/dqn_step{step:08d}.pth")



class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.preload()

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None

        with torch.cuda.stream(self.stream):
            self.next_data = (x.to(self.device, non_blocking=True) for x in self.next_data)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val

def random_seed(seed=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(np.random.randint(int(1e6)))


def set_thread(n):
    os.environ['OMP_NUM_THREADS'] = str(n)
    os.environ['MKL_NUM_THREADS'] = str(n)
    torch.set_num_threads(n)

def init(m, gain=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, gain)
        nn.init.zeros_(m.bias.data)

class ConvNet(nn.Module):
    def __init__(self, in_channels, action_dim):
        super(ConvNet, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU())

        self.v = nn.Linear(512, 1)
        self.p = nn.Linear(512, action_dim)

        self.convs.apply(lambda m: init(m, nn.init.calculate_gain('relu')))
        self.p.apply(lambda m: init(m, 0.01))
        self.v.apply(lambda m: init(m, 1.0))

    def forward(self, x):
        features = self.convs(x)
        values = self.v(features)
        advantange = self.p(features)
        q = values.expand_as(advantange) + (advantange - advantange.mean(1, keepdim=True).expand_as(advantange))
        return q


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, beta=None):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class ClipActionsWrapper(gym.Wrapper):
    def step(self, action):
        import numpy as np
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
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
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((shp[0] * k,) + shp[1:]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

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
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


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
        self.observation_space = gym.spaces.Box(0.0, 1.0, [1, 42, 42])

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
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)



def make_atari(env_id, max_episode_steps=None):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=False, scale=False, transpose_image=True):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if transpose_image:
        env = TransposeImage(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env







if __name__ == '__main__':
    main()
