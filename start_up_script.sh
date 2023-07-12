apt update

apt install -y git tmux fish vim htop libgl1-mesa-glx wget unzip libglib2.0-0 curl unrar


curl https://rclone.org/install.sh | bash

git clone https://github.com/zhoubin-me/agent0.git

cd agent0

conda env create -n agent0 --file environment.yml

conda activate agent0

pip install -e .
