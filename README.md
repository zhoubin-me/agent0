# AgentZero

## Introduction

AgentZero is a Ray & PyTorch based light-weight Distributed Fast Reinforcement Learning Framework.


## Installation
```bash
git clone https://github.com/zmonoid/AgentZero
cd AgentZero
pip install -e .
```

Simple run
```
python -m agent0.deepq.run
```

If you have library missing issues:
```bash
sudo apt install libsm6 libglib2.0-0
```
This is the required library for OpenCV

## Hardware Requirements:
- A GPU
- RAM > 16G


## Speed Test
Hardware Setting:
 - CPU: AMD EPYC 7251 8-Core Processor
 - GPU: RTX 2080 Ti
 - RAM: 32 GB
 - Game: Breakout
 - Algorithm: Rainbow
 - FPS: 700+

Here FPS (frame per second) means frames collected and saved to replay buffer per second. With 4 frame skip, its FPS is 700X4=3600 in deepmind's word. 
Other implementations usually fall below 100 FPS after exploration. The bottle net is actually at data transferring from CPU to GPU.

## Run

Basic DQN:
```bash
python -m agent0.deepq.run
```

Specify game:
```bash
python -m agent0.deepq.run --game enduro
```
Or run over a list of games as defined in ```src/common/bench.py```
```
python -m agent0.deepq.run --game atari6
```
Specify algorithms:
```bash
# Our current implementation includes: c51, qr, mdqn
python -m agent0.deepq.run --algo c51
```

Run like in rainbow:
```bash
# exp_name will specify checkpoint directory under $HOME/ray_results
python -m agent0.deepq.run --double_q --dueling --noisy --priortize --n_step 3 --game atari47 --algo c51 --exp_name atari_rainbow
```

