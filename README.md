# AgentZero

## Introduction

AgentZero is a Ray & PyTorch based light-weight Distributed Fast Reinforcement Learning Framework.


## Installation
```bash
git clone https://github.com/zmonoid/AgentZero
```

## Requirements

#### Hardware Requirements:
- A GPU
- RAM > 16G

#### Package Requirements
 - gym[atari]
 - prefetch_generator
 - ray[tune]
 - torch
 - lz4
 - dataclasses
 - GitPython
 
If you have package version problems:
```bash
pip install -r requirements.txt
```
 
If you have import error on  ```ligGL.so.1```:
```bash
sudo apt install libgl1-mesa-glx
```
## Speed Test
Hardware Setting:
 - CPU: AMD EPYC 7251 8-Core Processor
 - GPU: RTX 2080 Ti
 - RAM: 32 GB

800FPS

## Run

Basic DQN:
```bash
python -m src.deepq.run
```

Specify game:
```bash
python -m src.deepq.run --game enduro
```
Or run over a list of games as defined in ```src/common/bench.py```
```
python -m src.deepq.run --game atari6
```
Specify algorithms:
```bash
# Our current implementation includes: c51, qr, mdqn
python -m src.deepq.run --algo c51
```

Run like in rainbow:
```bash
# exp_name will specify checkpoint directory under $HOME/ray_results
python -m src.deepq.run --double_q --dueling --noisy --priortize --n_step 3 --game atari47 --algo c51 --exp_name atari_rainbow
```

