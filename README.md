# AgentZero
A [Ray, PyTorch] based Fast RL Framework


## Installation
```bash
git clone https://github.com/zmonoid/AgentZero
```

## Requirements
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

Run like rainbow:
```bash
# exp_name will specify checkpoint directory under $HOME/ray_results
python -m src.deepq.run --double_q --dueling --noisy --priortize --n_step 3 --game atari6 --algo c51 --exp_name atari_rainbow
```

