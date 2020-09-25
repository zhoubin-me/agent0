apt update

apt install -y git tmux fish vim htop libgl1-mesa-glx wget unzip libglib2.0-0 curl

pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

curl https://rclone.org/install.sh | bash

git clone https://github.com/benelot/pybullet-gym.git

cd pybullet-gym

pip install -e .

cd ..

git clone https://zhoubinxyz@bitbucket.org/zhoubinxyz/agentzero.git

cd agentzero

pip install -e .

env CUDA_VISIBLE_DEVICES=0 python -m agent0.deepq.run --algo gmm --game atari6 --gpu_mult 1.0 &
env CUDA_VISIBLE_DEVICES=1 python -m agent0.deepq.run --algo gmm --game atari6 --gpu_mult 1.0 --reversed &
