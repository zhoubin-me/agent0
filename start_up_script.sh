apt update

apt install -y git tmux fish vim htop libgl1-mesa-glx wget unzip libglib2.0-0 curl unrar

pip install torch torchvision torchaudio

curl https://rclone.org/install.sh | bash

git clone https://github.com/zhoubin-me/agent0.git

cd agent0

pip install -e .

wget http://www.atarimania.com/roms/Roms.rar

mkdir roms

unrar e Roms.rar roms/

python -m atari_py.import_roms roms
