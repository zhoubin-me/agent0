import argparse
import json
import os
import time

import numpy as np
import ray
import torch
from ray import tune

from src.common.utils import LinearSchedule, pprint
from src.deepq.agent import default_hyperparams, Actor, Agent


def run(config=None, **kwargs):
    if config is not None:
        kwargs = default_hyperparams()
        for k, v in config.items():
            kwargs[k] = v
    else:
        args = default_hyperparams()
        for k, v in args.items():
            if k not in kwargs:
                kwargs[k] = v

    agent = Agent(**kwargs)

    try:
        os.mkdir(agent.ckpt_tune)
    except:
        pass

    epsilon_schedule = LinearSchedule(1.0, 0.01, int(agent.total_steps * agent.exploration_ratio))
    actors = [Actor.remote(rank=rank, **kwargs) for rank in range(agent.num_actors + 1)]
    tester = actors[-1]

    steps_per_epoch = agent.total_steps // agent.epoches
    actor_steps = steps_per_epoch // (agent.num_envs * agent.num_actors)

    # Warming Up
    sample_ops = [a.sample.remote(actor_steps, 1.0, agent.model.state_dict()) for a in actors]
    RRs, QQs, TRRs, LLs = [], [], [], []
    for local_replay, Rs, Qs, rank, fps in ray.get(sample_ops):
        if rank < agent.num_actors:
            agent.replay.extend(local_replay)
            RRs += Rs
            QQs += Qs
        else:
            TRRs += Rs
    pprint("Warming up Reward", RRs)
    pprint("Warming up Qmax  ", QQs)

    actor_fps, training_fps, iteration_fps, iteration_time, training_time = [], [], [], [], []
    epoch, steps = 0, 0
    tic = time.time()
    while True:
        # Sample data
        sampler_tic = time.time()
        done_id, sample_ops = ray.wait(sample_ops)
        data = ray.get(done_id)
        local_replay, Rs, Qs, rank, fps = data[0]
        if rank < agent.num_actors:
            # Actors
            agent.replay.extend(local_replay)
            epsilon = epsilon_schedule(len(local_replay))
            if epsilon == 0.01:
                epsilon = np.random.choice([0.01, 0.02, 0.05, 0.1], p=[0.7, 0.1, 0.1, 0.1])
            sample_ops.append(actors[rank].sample.remote(actor_steps, epsilon, agent.model.state_dict()))
            RRs += Rs
            QQs += Qs
            steps += len(local_replay)
            actor_fps.append(fps)
        else:
            # Tester
            sample_ops.append(tester.sample.remote(actor_steps, 0.01, agent.model.state_dict()))
            TRRs += Rs

        # Trainer
        trainer_tic = time.time()
        Ls = [agent.train_step() for _ in range(agent.agent_train_freq)]
        Ls = torch.stack(Ls).tolist()
        LLs += Ls

        toc = time.time()
        training_fps += [(agent.batch_size * agent.agent_train_freq) / (toc - trainer_tic)]
        iteration_fps += [len(local_replay) / (toc - sampler_tic)]
        iteration_time += [toc - sampler_tic]
        training_time += [toc - trainer_tic]
        # Logging and saving
        if (steps // steps_per_epoch) > epoch:
            epoch += 1

            # Start Testing at Epoch 10
            if epoch == 10:
                sample_ops.append(tester.sample.remote(actor_steps, 0.01, agent.model.state_dict()))

            # Logging every 10 epocoh
            if epoch % 10 == 1:
                speed = steps / (toc - tic)
                print("=" * 100)
                print(f"Epoch:{epoch:4d}\t Steps:{steps:8d}\t "
                      f"Updates:{agent.update_steps:4d}\t "
                      f"TimePast(min):{(toc - tic) / 60:8.2f}\t "
                      f"EstTimeRem(min):{(agent.total_steps - steps) / speed / 60:8.2f}\n"
                      f"AvgSpeedFPS:{speed:8.2f}\t "
                      f"Epsilon:{epsilon:6.4}")
                print('-' * 100)

                pprint("Training Reward   ", RRs[-1000:])
                pprint("Loss              ", LLs[-1000:])
                pprint("Qmax              ", QQs[-1000:])
                pprint("Test Reward       ", TRRs[-1000:])
                pprint("Training Speed    ", training_fps[-20:])
                pprint("Training Time     ", training_time[-20:])
                pprint("Iteration Time    ", iteration_time[-20:])
                pprint("Iteration FPS     ", iteration_fps[-20:])
                pprint("Actor FPS         ", actor_fps[-20:])

                print("=" * 100)
                print(" " * 100)

            if epoch % 50 == 1:
                torch.save({
                    'model': agent.model.state_dict(),
                    'optim': agent.optimizer.state_dict(),
                    'epoch': epoch,
                    'epsilon': epsilon,
                    'steps': steps,
                    'Rs': RRs,
                    'TRs': TRRs,
                    'Qs': QQs,
                    'Ls': LLs,
                    'time': toc - tic,
                    'params': kwargs,
                }, f'{agent.save_prefix}/{agent.game}_e{epoch:04d}.pth')

            if epoch > agent.epoches:
                print("Final Testing")
                FTRs = []
                ray.get([a.reset_envs.remote(False, False) for a in self.actors])
                datas = ray.get([a.sample.remote(self.actor_steps * 10, self.epsilon, self.agent.model.state_dict())
                         for a in self.actors])
                ray.get([a.close_envs.remote() for a in self.actors])
                for local_replay, Rs, Qs, rank, fps in datas:
                    FTRs += Rs
                torch.save({
                    'model': agent.model.state_dict(),
                    'optim': agent.optimizer.state_dict(),
                    'epoch': epoch,
                    'epsilon': epsilon,
                    'steps': steps,
                    'Rs': RRs,
                    'TRs': TRRs,
                    'FTRs': FTRs,
                    'Qs': QQs,
                    'Ls': LLs,
                    'time': toc - tic,
                    'params': kwargs,
                    'FTRs': TRs
                }, f'{agent.save_prefix}./{agent.game}_final.pth')



                if config is not None:
                    tune.report(final_test_rewards=np.mean(TRs))
                return


def parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(f"--{k}", type=type(v), default=v)
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return vars(args)


if __name__ == '__main__':
    params = default_hyperparams()
    kwargs = parse_arguments(params)
    ray.init(memory=20 * 2 ** 30, object_store_memory=100 * 2 ** 30)
    run(**kwargs)
