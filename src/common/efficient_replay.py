import numpy as np


class NPReplay:
    def __init__(self, num_steps, num_envs, obs_shape, n_stack):

        self.imgs = np.zeros((num_steps, num_envs, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((num_steps, num_envs))
        self.rewards = np.zeros((num_steps, num_envs))
        self.terminals = np.zeros((num_steps, num_envs))

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.obs_shape = obs_shape
        self.n_stack = n_stack

        self.top = 0
        self.ptr = 0

    def add(self, imgs, actions, rewards, terminals):
        self.imgs[self.ptr] = np.array(imgs).squeeze()
        self.actions[self.ptr] = np.array(actions)
        self.rewards[self.ptr] = np.array(rewards)
        self.terminals[self.ptr] = np.array(terminals)

        self.ptr = (self.ptr + 1) % self.num_steps
        self.top = min(self.top + 1, self.num_steps)

    def sample_transit(self, step_idx, env_idx):
        if any(self.terminals[:, env_idx].take(np.arange(step_idx - self.n_stack + 1, step_idx))):
            return None
        else:
            imgs = self.imgs[:, env_idx].take(np.arange(step_idx - self.n_stack + 1, step_idx + 2), axis=0)
            action = self.actions[step_idx, env_idx]
            reward = self.rewards[step_idx, env_idx]
            terminal = self.terminals[step_idx, env_idx]
            return imgs, action, reward, terminal

    def sample_batch(self, batch_size):
        images = np.zeros((batch_size, self.n_stack + 1, *self.obs_shape))
        actions = np.zeros(batch_size)
        rewards = np.zeros(batch_size)
        terminals = np.zeros(batch_size)

        count = 0
        while count < batch_size:
            idx = np.random.randint(self.top * self.num_envs)
            transit = self.sample_transit(idx // self.num_envs, idx % self.num_envs)
            if transit is not None:
                img, act, rew, term = transit
                images[count] = img
                actions[count] = act
                rewards[count] = rew
                terminals[count] = term
                count += 1
        return images[:, :-1, :, :], actions, rewards, images[:, 1:, :, :], terminals


if __name__ == '__main__':
    from src.common.atari_wrappers import make_atari, wrap_deepmind
    from src.common.vec_env import ShmemVecEnv


    def make_env():
        env = make_atari('BreakoutNoFrameskip-v4')
        env = wrap_deepmind(env)
        return env


    envs = ShmemVecEnv([make_env for _ in range(10)])
    replay = NPReplay(1000, 10, (84, 84), 4)
    obs = envs.reset()
    print(obs.shape)
    for _ in range(100):
        actions = np.random.randint(0, envs.action_space.n, 10)
        obs_next, reward, done, info = envs.step(actions)
        replay.add(obs, actions, reward, done)

    for _ in range(10):
        images, actions, rewards, terminals = replay.sample_batch(32)
    print(images.shape)
    print(actions.shape)
