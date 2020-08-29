import pickle

import gym
import torch
import torchvision as tv
from PIL import Image
from torch.utils.data import Dataset

from agent0.common.utils import DataPrefetcher, DataLoaderX, parse_arguments
from agent0.nips_encoder.model import ModelEncoder
from agent0.nips_encoder.trainer import Config


class EncoderDataset(Dataset):
    def __init__(self, data, state_shape):
        self.data = data
        self.state_shape = state_shape

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        st, at, rt, dt, st_next = self.data[idx]
        return st, at, rt, dt, st_next


if __name__ == '__main__':
    cfg = Config()
    args = parse_arguments(cfg)
    cfg = Config(**vars(args))
    env = gym.make('{cfg.game}NoFrameskip-v4')

    transit = []
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        next_obs, reward, done, _ = env.step(action)
        transit.append((obs, action, reward, done, next_obs))
        obs = next_obs
        if done:
            break

    dataset = EncoderDataset(transit, env.observation_space.shape)
    data_loader = DataLoaderX(dataset, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=True)

    data_fetcher = DataPrefetcher(data_loader, torch.device('cuda:0'))
    model = ModelEncoder(env.action_space.n)
    with open(cfg.restore_checkpoint, 'rb') as f:
        data = pickle.load(f)
        model.load_state_dict(data['model'])

    model = model.cuda()

    outs = []
    while True:
        try:
            data = data_fetcher.next()
        except (StopIteration, AttributeError):
            break
        with torch.no_grad():
            st, at, rt, dt, st_next = data
            st = st.float().div(255.0).permute(0, 3, 1, 2)
            st_next = st_next.float().div(255.0).permute(0, 3, 1, 2)
            at = at.long()

            st_predict = model(st, at).detach().cpu()
            obs = torch.cat((st_next.cpu(), st_predict), dim=-1)
            outs.append(obs)
    outs_ = torch.cat(outs, dim=0)
    img_ = tv.utils.make_grid(outs_).mul(255.0).permute(1, 2, 0).byte().numpy()
    img = Image.fromarray(img_)
    img.save('out.png')
