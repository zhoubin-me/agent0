import gym


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
            info.update(real_reward=self.real_reward, steps=self.steps)
            self.steps = 0
            self.real_reward = 0
        return ob, reward, done, info


def make_bullet_env(game, seed=None, **kwargs):
    env = gym.make(f"{game}PyBulletEnv-v0")
    env = RewardStatEnv(env)
    env.seed(seed)
    return env
