import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from agent0.common.atari_wrappers import make_atari
from agent0.common.utils import DataLoaderX, DataPrefetcher, set_random_seed
import agent0.deepq.agent as agents
from agent0.deepq.config import ExpConfig
from agent0.deepq.replay import ReplayDataset, ReplayEnum

class Trainer:
    def __init__(self, cfg: ExpConfig, use_lp=False):
        self.cfg = cfg

        set_random_seed(cfg.seed)

        dummy_env = make_atari(cfg.env_id, 1)
        self.obs_shape = dummy_env.observation_space.shape[1:]
        self.act_dim = dummy_env.action_space[0].n
        dummy_env.close()
        
        try:
            self.learner = getattr(agents, f'{self.cfg.learner.algo.name.upper()}Learner')(cfg)
        except:
            raise NotImplementedError(f"No such learner for {self.cfg.learner.algo.name}")
        self.replay = ReplayDataset(cfg)
        if not use_lp:
            self.actors = agents.Actor(cfg)

        self.epsilon_fn = (
            lambda step: cfg.actor.min_eps
            if step > cfg.trainer.exploration_steps
            else (1.0 - step / cfg.trainer.exploration_steps) + cfg.actor.min_eps
        )

        self.frame_count = 0
        self.writer = SummaryWriter(cfg.logdir)
        self.num_transitions = cfg.actor.actor_steps * cfg.actor.num_envs
        self.Ls, self.Rs, self.Qs, self.FLs = [], [], [], []
        self.data_fetcher = None

    def get_data_fetcher(self):
        data_loader = DataLoaderX(
            self.replay,
            batch_size=self.cfg.learner.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        data_fetcher = DataPrefetcher(data_loader, self.cfg.device.value)
        return data_fetcher

    def step(self, transitions, returns, qmax):

        self.Qs.extend(qmax)
        self.Rs.extend(returns)
        self.replay.extend(transitions)
        self.frame_count += self.num_transitions

        # Start training at
        if len(self.replay) > self.cfg.trainer.training_start_steps:
            for _ in range(self.cfg.learner.learner_steps):
                try:
                    data = self.data_fetcher.next()
                except (StopIteration, AttributeError):
                    self.data_fetcher = self.get_data_fetcher()
                    data = self.data_fetcher.next()
                frames, actions, rewards, terminals, priorities, indices = map(lambda x: x.float(), data)
                probs = priorities / self.replay.priority.sum().item()
                weights = (self.replay.top * probs).pow(-self.replay.beta)
                weights = weights / weights.max().add(1e-8)
                data = frames, actions, rewards, terminals, weights, indices
                result = self.learner.train(data)
                q_loss = result['q_loss']
                fraction_loss = result['fraction_loss']
                indices = result['indices']

                if self.cfg.replay.policy == ReplayEnum.prioritize:
                    self.replay.update_priority(indices, priorities=q_loss)

                if q_loss is not None: self.Ls.append(q_loss.mean().item())
                if fraction_loss is not None: self.FLs.append(fraction_loss.mean().item())

        result = dict(
            frames=self.frame_count,
            fraction_loss=np.mean(self.FLs[-20:]) if len(self.FLs) > 0 else None,
            loss=np.mean(self.Ls[-20:]) if len(self.Ls) > 0 else None,
            return_train=np.mean(self.Rs[-20:]) if len(self.Rs) > 0 else None,
            return_train_max=np.max(self.Rs) if len(self.Rs) > 0 else None,
            qmax=np.mean(self.Qs[-100:]) if len(self.Qs) > 0 else None,
        )
        return result

    def run(self):
        trainer_steps = self.cfg.trainer.total_steps // self.num_transitions + 1

        with tqdm(range(trainer_steps)) as t:
            for _ in t:
                epsilon = self.epsilon_fn(self.frame_count)
                transitions, returns, qmax = self.actors.sample(epsilon, self.learner.model.state_dict())
                result = self.step(transitions, returns, qmax)
                msg = ""
                for k, v in result.items():
                    if v is None:
                        continue
                    self.writer.add_scalar(k, v, self.frame_count)
                    if k in ["frames", "loss", "qmax"] or "return" in k:
                        msg += f"{k}: {v:.2f} | "
                t.set_description(msg)

        self.actors.close()
