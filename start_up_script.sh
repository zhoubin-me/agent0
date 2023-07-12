apt update

apt install -y git tmux fish vim htop libgl1-mesa-glx wget unzip libglib2.0-0 curl

git clone https://github.com/zhoubin-me/AgentZero
cd AgentZero
conda env create -f environment.yml
conda activate agent0
pip install tensorflow==2.8.4
pip install protobuf==3.20.3
pip install -e .