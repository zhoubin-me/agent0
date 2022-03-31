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
# Our current implementation includes: c51, qr, mdqn, iqr, fqf
python -m agent0.deepq.run --algo c51
```

Run like in rainbow:
```bash
# exp_name will specify checkpoint directory under $HOME/ray_results
python -m agent0.deepq.run --double_q --dueling --noisy --priortize --n_step 3 --game atari47 --algo c51 --exp_name atari_rainbow
```

## Sample Run Result @10M Frames


![](imgs/Asterix.png)

![](imgs/BeamRider.png)

![](imgs/Breakout.png)

![](imgs/Enduro.png)

![](imgs/Qbert.png)

![](imgs/Seaquest.png)

![](imgs/SpaceInvaders.png)


| exp_name        | commit | algo | game          | test_ep_reward_mean | max     | min     | ckpt_frame |
|-----------------|--------|------|---------------|---------------------|---------|---------|------------|
| c51             | 3e3b70 | c51  | Asterix       | 30541.666666666668  | 39300.0 | 10700.0 | 10000640   |
| c51             | 3e3b70 | c51  | BeamRider     | 9890.166666666666   | 18472.0 | 4606.0  | 9600000    |
| c51             | 3e3b70 | c51  | Breakout      | 544.6666666666666   | 840.0   | 365.0   | 7040000    |
| c51             | 3e3b70 | c51  | Enduro        | 2833.9166666666665  | 5258.0  | 1914.0  | 7040000    |
| c51             | 3e3b70 | c51  | Qbert         | 16804.166666666668  | 20100.0 | 12275.0 | 10000640   |
| c51             | 3e3b70 | c51  | Seaquest      | 3171.6666666666665  | 4440.0  | 1940.0  | 9920000    |
| c51             | 3e3b70 | c51  | SpaceInvaders | 1154.5833333333333  | 1860.0  | 570.0   | 4160000    |
| double          | a6fca1 | dqn  | Asterix       | 5416.666666666667   | 7700.0  | 2700.0  | 8960000    |
| double          | a6fca1 | dqn  | BeamRider     | 4848.333333333333   | 7776.0  | 2160.0  | 1920000    |
| double          | a6fca1 | dqn  | Breakout      | 426.9166666666667   | 758.0   | 309.0   | 8000000    |
| double          | a6fca1 | dqn  | Enduro        | 1166.6666666666667  | 1391.0  | 796.0   | 5760000    |
| double          | a6fca1 | dqn  | Qbert         | 6852.083333333333   | 8750.0  | 4075.0  | 8960000    |
| double          | a6fca1 | dqn  | Seaquest      | 8249.166666666666   | 11740.0 | 5420.0  | 10000640   |
| double          | a6fca1 | dqn  | SpaceInvaders | 268.3333333333333   | 410.0   | 155.0   | 640000     |
| dqn             | 3e3b70 | dqn  | Asterix       | 4883.333333333333   | 7700.0  | 3600.0  | 9280000    |
| dqn             | 3e3b70 | dqn  | BeamRider     | 5947.5              | 9468.0  | 3000.0  | 9920000    |
| dqn             | 3e3b70 | dqn  | Breakout      | 431.0               | 835.0   | 145.0   | 9920000    |
| dqn             | 3e3b70 | dqn  | Enduro        | 1561.0833333333333  | 1977.0  | 1074.0  | 7360000    |
| dqn             | 3e3b70 | dqn  | Qbert         | 4358.333333333333   | 4700.0  | 4150.0  | 9600000    |
| dqn             | 3e3b70 | dqn  | Seaquest      | 9468.333333333334   | 14610.0 | 5140.0  | 9920000    |
| dqn             | 3e3b70 | dqn  | SpaceInvaders | 1398.75             | 2490.0  | 600.0   | 9920000    |
| dqn_norm_reward | a6fca1 | dqn  | Asterix       | 3037.5              | 5400.0  | 900.0   | 9920000    |
| dqn_norm_reward | a6fca1 | dqn  | BeamRider     | 5447.666666666667   | 8886.0  | 1692.0  | 6080000    |
| dqn_norm_reward | a6fca1 | dqn  | Breakout      | 385.0               | 425.0   | 280.0   | 8960000    |
| dqn_norm_reward | a6fca1 | dqn  | Enduro        | 1764.0              | 2262.0  | 1059.0  | 5120000    |
| dqn_norm_reward | a6fca1 | dqn  | Qbert         | 6479.166666666667   | 7825.0  | 3800.0  | 10000640   |
| dqn_norm_reward | a6fca1 | dqn  | Seaquest      | 7561.666666666667   | 15490.0 | 3820.0  | 9280000    |
| dqn_norm_reward | a6fca1 | dqn  | SpaceInvaders | 711.6666666666666   | 1375.0  | 550.0   | 2880000    |
| prioritized     | a6fca1 | dqn  | Asterix       | 2633.3333333333335  | 3300.0  | 1150.0  | 7680000    |
| prioritized     | a6fca1 | dqn  | BeamRider     | 4307.166666666667   | 7162.0  | 1380.0  | 8960000    |
| prioritized     | a6fca1 | dqn  | Breakout      | 385.75              | 420.0   | 325.0   | 7040000    |
| prioritized     | a6fca1 | dqn  | Enduro        | 426.25              | 478.0   | 386.0   | 2240000    |
| prioritized     | a6fca1 | dqn  | Qbert         | 1204.1666666666667  | 4550.0  | 400.0   | 2880000    |
| prioritized     | a6fca1 | dqn  | Seaquest      | 318.3333333333333   | 800.0   | 140.0   | 1600000    |
| qr              | 3e3b70 | qr   | Asterix       | 6661.111111111111   | 8700.0  | 4200.0  | 9280000    |
| qr              | 3e3b70 | qr   | BeamRider     | 7235.444444444444   | 11050.0 | 3140.0  | 10000640   |
| qr              | 3e3b70 | qr   | Breakout      | 452.44444444444446  | 789.0   | 338.0   | 10000640   |
| qr              | 3e3b70 | qr   | Enduro        | 1555.9444444444443  | 1992.0  | 1032.0  | 6720000    |
| qr              | 3e3b70 | qr   | Qbert         | 9709.722222222223   | 15875.0 | 3950.0  | 10000640   |
| qr              | 3e3b70 | qr   | Seaquest      | 16492.777777777777  | 28970.0 | 6830.0  | 9280000    |
| qr              | 3e3b70 | qr   | SpaceInvaders | 1215.8333333333333  | 2120.0  | 540.0   | 6080000    |
| rainbow         | 3e3b70 | c51  | Asterix       | 6183.333333333333   | 8900.0  | 3100.0  | 9920000    |
| rainbow         | 3e3b70 | c51  | BeamRider     | 8254.5              | 13758.0 | 4220.0  | 3840000    |
| rainbow         | 3e3b70 | c51  | Breakout      | 205.0               | 348.0   | 30.0    | 10000640   |
| rainbow         | 3e3b70 | c51  | Enduro        | 3204.3333333333335  | 4696.0  | 1673.0  | 6720000    |
| rainbow         | 3e3b70 | c51  | Qbert         | 22512.5             | 26675.0 | 15925.0 | 9600000    |
| rainbow         | 3e3b70 | c51  | Seaquest      | 7024.166666666667   | 12550.0 | 4080.0  | 7680000    |
| rainbow         | 3e3b70 | c51  | SpaceInvaders | 1187.0833333333333  | 2660.0  | 540.0   | 2880000    |
| rainbow_fqf     | b328c5 | fqf  | BeamRider     | 14261.84761904762   | 31040.0 | 3540.0  | 7680000    |
| rainbow_fqf     | b328c5 | fqf  | Breakout      | 395.7019230769231   | 430.0   | 147.0   | 10000640   |
| rainbow_fqf     | b328c5 | fqf  | Enduro        | 4564.542857142857   | 9479.0  | 1333.0  | 7680000    |
| rainbow_fqf     | b328c5 | fqf  | Qbert         | 15888.809523809523  | 23075.0 | 11750.0 | 10000640   |
| rainbow_fqf     | b328c5 | fqf  | Seaquest      | 33468.0             | 82550.0 | 4160.0  | 10000640   |
| rainbow_fqf     | b328c5 | fqf  | SpaceInvaders | 3415.3333333333335  | 10790.0 | 575.0   | 10000640   |

